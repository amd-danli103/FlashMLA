"""
Optimized Triton MLA Decode Kernels - Version 5.4

Key optimizations:
1. Fused Triton kernel for gather+dequant (significantly faster than PyTorch)
2. Process both KV scopes in single attention kernel
3. Minimize memory allocations
4. Autotuned block sizes for different configurations
5. Fixed 64-bit pointer arithmetic to avoid overflow with large KV caches
6. Correct handling of MODEL1 block-level layout
7. PyTorch fallback for large outputs to avoid int32 pointer overflow
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

LOG2E = tl.constexpr(1.4426950408889634)

# Block sizes for attention kernel
BLOCK_H = 16
BLOCK_N = 64
BLOCK_D = 128

# Constants for MODEL1 layout
MODEL1_D_QK = 512
MODEL1_D_NOPE = 448
MODEL1_D_ROPE = 64
MODEL1_TILE_SIZE = 64
MODEL1_NUM_TILES = 7
MODEL1_BYTES_PER_TOKEN_DATA = 576  # 448 nope + 128 rope
MODEL1_BYTES_PER_TOKEN_SCALE = 8   # 7 scales + 1 padding

# Constants for V32 layout
V32_D_QK = 576
V32_D_NOPE = 512
V32_D_ROPE = 64
V32_TILE_SIZE = 128
V32_NUM_TILES = 4
V32_BYTES_PER_TOKEN = 656

# ============================================================================
# Optimized Gather+Dequant Kernels
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_TK': 16}, num_warps=2),
        triton.Config({'BLOCK_TK': 32}, num_warps=4),
        triton.Config({'BLOCK_TK': 64}, num_warps=4),
        triton.Config({'BLOCK_TK': 128}, num_warps=8),
    ],
    key=['total_tokens', 'topk'],
)
@triton.jit
def _gather_dequant_model1_kernel(
    # KV cache is flattened to [num_blocks, bytes_per_block]
    KV_Cache,
    Indices,
    InvalidMask,
    Output,
    total_tokens,
    topk,
    num_blocks,
    block_size,
    stride_kv_block,  # Stride between blocks (in bytes)
    stride_idx_t, stride_idx_k,
    stride_mask_t, stride_mask_k,
    stride_out_t, stride_out_k, stride_out_d,
    BLOCK_TK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    BYTES_PER_TOKEN_DATA: tl.constexpr,
    BYTES_PER_TOKEN_SCALE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """
    Fused gather + dequant kernel for MODEL1 layout.

    MODEL1 block layout:
    [token0_nope_rope (576B)][token1_nope_rope (576B)]...[tokenN_nope_rope (576B)]
    [token0_scales (8B)][token1_scales (8B)]...[tokenN_scales (8B)]

    Per-token nope_rope layout (576 bytes):
    [nope (448 FP8)][rope (128 bytes = 64 bf16)]

    Per-token scales layout (8 bytes):
    [7 E8M0 scales][1 padding]
    """
    pid = tl.program_id(0)
    num_tk = total_tokens * topk

    offs_tk = pid * BLOCK_TK + tl.arange(0, BLOCK_TK)
    mask_tk = offs_tk < num_tk

    t_idx = offs_tk // topk
    k_idx = offs_tk % topk

    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k
    indices = tl.load(idx_ptrs, mask=mask_tk, other=0)

    mask_ptrs = InvalidMask + t_idx * stride_mask_t + k_idx * stride_mask_k
    is_invalid = tl.load(mask_ptrs, mask=mask_tk, other=True)

    valid_mask = mask_tk & ~is_invalid
    indices_clamped = tl.maximum(indices, 0)

    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    # Use 64-bit arithmetic to avoid overflow
    # Convert tensor values to int64, then multiply by scalar strides
    # The multiplication of int64 tensor with Python int produces int64 result
    block_idx_64 = block_idx.to(tl.int64)
    offset_in_block_64 = offset_in_block.to(tl.int64)

    # Base pointer for each block (block_idx_64 * stride_kv_block is int64)
    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block

    # Compute byte offsets within block
    # nope_rope data: offset_in_block * BYTES_PER_TOKEN_DATA
    # scales: block_size * BYTES_PER_TOKEN_DATA + offset_in_block * BYTES_PER_TOKEN_SCALE
    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
    scale_offset = block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE

    # Use 64-bit arithmetic for output pointers to avoid overflow with large topk
    # Convert tensor indices to int64 first, then multiply by scalar strides
    # The multiplication of int64 tensor with Python int produces int64 result
    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    out_base_ptrs = Output + t_idx_64 * stride_out_t + k_idx_64 * stride_out_k

    # Process 7 tiles of nope (each 64 FP8 elements)
    for tile_idx in range(7):
        tile_start = tile_idx * TILE_SIZE

        # Load scale for this tile (E8M0 format)
        scale_ptrs = kv_block_base + scale_offset + tile_idx
        scale_uint8 = tl.load(scale_ptrs, mask=valid_mask, other=127).to(tl.uint8)

        # E8M0 to float: 2^(val - 127)
        scale_exp = scale_uint8.to(tl.float32) - 127.0
        scale_f32 = tl.math.exp2(scale_exp)
        scale_bf16 = scale_f32.to(tl.bfloat16)

        offs_d = tl.arange(0, TILE_SIZE)

        # Load nope data (FP8)
        nope_ptrs = kv_block_base[:, None] + nope_rope_offset[:, None] + tile_start + offs_d[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)

        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        nope_bf16 = nope_fp8.to(tl.bfloat16)

        dequant = nope_bf16 * scale_bf16[:, None]
        dequant = tl.where(is_invalid[:, None], 0.0, dequant)

        out_ptrs = out_base_ptrs[:, None] + (tile_start + offs_d[None, :]) * stride_out_d
        tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process rope (64 bf16 values = 128 bytes)
    offs_rope = tl.arange(0, D_ROPE)
    rope_byte_start = D_NOPE  # rope starts after nope in per-token data

    rope_lo_ptrs = kv_block_base[:, None] + nope_rope_offset[:, None] + rope_byte_start + offs_rope[None, :] * 2
    rope_hi_ptrs = kv_block_base[:, None] + nope_rope_offset[:, None] + rope_byte_start + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where(is_invalid[:, None], 0.0, rope_bf16)

    out_ptrs = out_base_ptrs[:, None] + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk[:, None])

def gather_dequant_fp8_model1_triton(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    invalid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Triton implementation of gather+dequant for MODEL1 layout."""
    total_tokens, topk = indices.shape
    device = kv_cache_quantized.device
    num_blocks = kv_cache_quantized.shape[0]

    # View as uint8 and flatten to [num_blocks, bytes_per_block]
    kv_uint8 = kv_cache_quantized.view(torch.uint8)
    bytes_per_block = kv_uint8.shape[1] * kv_uint8.shape[2] * kv_uint8.shape[3]
    kv_flat = kv_uint8.reshape(num_blocks, bytes_per_block)

    # Get the actual stride between blocks (handles padding)
    stride_kv_block = kv_uint8.stride(0)

    output = torch.empty(total_tokens, topk, MODEL1_D_QK, dtype=torch.bfloat16, device=device)

    grid = lambda meta: (triton.cdiv(total_tokens * topk, meta['BLOCK_TK']),)

    _gather_dequant_model1_kernel[grid](
        kv_flat,
        indices,
        invalid_mask,
        output,
        total_tokens,
        topk,
        num_blocks,
        block_size,
        stride_kv_block,
        indices.stride(0), indices.stride(1),
        invalid_mask.stride(0), invalid_mask.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        D_NOPE=MODEL1_D_NOPE,
        D_ROPE=MODEL1_D_ROPE,
        BYTES_PER_TOKEN_DATA=MODEL1_BYTES_PER_TOKEN_DATA,
        BYTES_PER_TOKEN_SCALE=MODEL1_BYTES_PER_TOKEN_SCALE,
        TILE_SIZE=MODEL1_TILE_SIZE,
    )

    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_TK': 16}, num_warps=2),
        triton.Config({'BLOCK_TK': 32}, num_warps=4),
        triton.Config({'BLOCK_TK': 64}, num_warps=4),
        triton.Config({'BLOCK_TK': 128}, num_warps=8),
    ],
    key=['total_tokens', 'topk'],
)
@triton.jit
def _gather_dequant_v32_kernel(
    # KV cache with shape [num_blocks, block_size, bytes_per_token]
    KV_Cache,
    Indices,
    InvalidMask,
    Output,
    total_tokens,
    topk,
    num_blocks,
    block_size,
    stride_kv_block,
    stride_kv_token,
    stride_idx_t, stride_idx_k,
    stride_mask_t, stride_mask_k,
    stride_out_t, stride_out_k, stride_out_d,
    BLOCK_TK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """
    Fused gather + dequant kernel for V32 layout.

    V32 per-token layout (656 bytes):
    [nope (512 FP8)][scales (16 bytes = 4 f32)][rope (128 bytes = 64 bf16)]
    """
    pid = tl.program_id(0)
    num_tk = total_tokens * topk

    offs_tk = pid * BLOCK_TK + tl.arange(0, BLOCK_TK)
    mask_tk = offs_tk < num_tk

    t_idx = offs_tk // topk
    k_idx = offs_tk % topk

    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k
    indices = tl.load(idx_ptrs, mask=mask_tk, other=0)

    mask_ptrs = InvalidMask + t_idx * stride_mask_t + k_idx * stride_mask_k
    is_invalid = tl.load(mask_ptrs, mask=mask_tk, other=True)

    valid_mask = mask_tk & ~is_invalid
    indices_clamped = tl.maximum(indices, 0)

    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    # Use 64-bit arithmetic to avoid overflow
    # Convert tensor values to int64, then multiply by scalar strides
    # The multiplication of int64 tensor with Python int produces int64 result
    block_idx_64 = block_idx.to(tl.int64)
    offset_in_block_64 = offset_in_block.to(tl.int64)

    # Compute KV base pointers (all multiplications are int64 * Python int = int64)
    kv_base_ptrs = KV_Cache + block_idx_64 * stride_kv_block + offset_in_block_64 * stride_kv_token

    # Use 64-bit arithmetic for output pointers to avoid overflow with large topk
    # Convert tensor indices to int64 first, then multiply by scalar strides
    # The multiplication of int64 tensor with Python int produces int64 result
    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    out_base_ptrs = Output + t_idx_64 * stride_out_t + k_idx_64 * stride_out_k

    # Process 4 tiles of nope (each 128 FP8 elements)
    for tile_idx in range(NUM_TILES):
        tile_start = tile_idx * TILE_SIZE

        # Load scale (f32) - 4 bytes per scale
        scale_byte_offset = D_NOPE + tile_idx * 4
        scale_b0_ptrs = kv_base_ptrs + scale_byte_offset
        scale_b1_ptrs = kv_base_ptrs + scale_byte_offset + 1
        scale_b2_ptrs = kv_base_ptrs + scale_byte_offset + 2
        scale_b3_ptrs = kv_base_ptrs + scale_byte_offset + 3

        scale_b0 = tl.load(scale_b0_ptrs, mask=valid_mask, other=0)
        scale_b1 = tl.load(scale_b1_ptrs, mask=valid_mask, other=0)
        scale_b2 = tl.load(scale_b2_ptrs, mask=valid_mask, other=0)
        scale_b3 = tl.load(scale_b3_ptrs, mask=valid_mask, other=0)

        scale_uint32 = (scale_b0.to(tl.uint32) |
                       (scale_b1.to(tl.uint32) << 8) |
                       (scale_b2.to(tl.uint32) << 16) |
                       (scale_b3.to(tl.uint32) << 24))
        scale_f32 = scale_uint32.to(tl.float32, bitcast=True)

        # Process 128 elements in two chunks of 64
        for chunk in range(2):
            chunk_start = tile_start + chunk * 64
            offs_d = tl.arange(0, 64)

            nope_ptrs = kv_base_ptrs[:, None] + chunk_start + offs_d[None, :]
            nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)

            nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
            nope_f32 = nope_fp8.to(tl.float32)

            dequant = nope_f32 * scale_f32[:, None]
            dequant = tl.where(is_invalid[:, None], 0.0, dequant)

            out_ptrs = out_base_ptrs[:, None] + (chunk_start + offs_d[None, :]) * stride_out_d
            tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process rope (64 bf16 values = 128 bytes)
    rope_byte_offset = D_NOPE + NUM_TILES * 4
    offs_rope = tl.arange(0, D_ROPE)

    rope_lo_ptrs = kv_base_ptrs[:, None] + rope_byte_offset + offs_rope[None, :] * 2
    rope_hi_ptrs = kv_base_ptrs[:, None] + rope_byte_offset + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where(is_invalid[:, None], 0.0, rope_bf16)

    out_ptrs = out_base_ptrs[:, None] + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk[:, None])

def gather_dequant_fp8_v32_triton(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    invalid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Triton implementation of gather+dequant for V32 layout."""
    total_tokens, topk = indices.shape
    device = kv_cache_quantized.device
    num_blocks = kv_cache_quantized.shape[0]

    # View as uint8 - shape is [num_blocks, block_size, 1, bytes_per_token]
    kv_uint8 = kv_cache_quantized.view(torch.uint8)

    # Get strides
    stride_kv_block = kv_uint8.stride(0)
    stride_kv_token = kv_uint8.stride(1)

    output = torch.empty(total_tokens, topk, V32_D_QK, dtype=torch.bfloat16, device=device)

    grid = lambda meta: (triton.cdiv(total_tokens * topk, meta['BLOCK_TK']),)

    _gather_dequant_v32_kernel[grid](
        kv_uint8,
        indices,
        invalid_mask,
        output,
        total_tokens,
        topk,
        num_blocks,
        block_size,
        stride_kv_block, stride_kv_token,
        indices.stride(0), indices.stride(1),
        invalid_mask.stride(0), invalid_mask.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        D_NOPE=V32_D_NOPE,
        D_ROPE=V32_D_ROPE,
        TILE_SIZE=V32_TILE_SIZE,
        NUM_TILES=V32_NUM_TILES,
    )

    return output

# ============================================================================
# PyTorch Fallback Implementations
# ============================================================================

def gather_dequant_fp8_model1_fast(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    invalid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Optimized PyTorch MODEL1 gather+dequant."""
    d_qk = 512
    d_nope = 448
    d_rope = 64
    tile_size = 64
    num_tiles = 7
    bytes_per_token_data = 576
    bytes_per_token_scale = 8

    total_tokens, topk = indices.shape
    device = kv_cache_quantized.device
    num_blocks = kv_cache_quantized.shape[0]

    indices_clamped = indices.clamp(min=0)
    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    kv_uint8 = kv_cache_quantized.view(torch.uint8)
    bytes_per_block = kv_uint8.shape[1] * kv_uint8.shape[2] * kv_uint8.shape[3]
    kv_flat = kv_uint8.reshape(num_blocks, bytes_per_block)

    nope_rope_size = block_size * bytes_per_token_data
    nope_rope_view = kv_flat[:, :nope_rope_size].view(num_blocks, block_size, bytes_per_token_data)
    scales_view = kv_flat[:, nope_rope_size:nope_rope_size + block_size * bytes_per_token_scale].view(
        num_blocks, block_size, bytes_per_token_scale)

    flat_block_idx = block_idx.view(-1)
    flat_offset = offset_in_block.view(-1)

    gathered_nope_rope = nope_rope_view[flat_block_idx, flat_offset].view(total_tokens, topk, bytes_per_token_data)
    gathered_scales = scales_view[flat_block_idx, flat_offset].view(total_tokens, topk, bytes_per_token_scale)

    gathered_nope = gathered_nope_rope[..., :d_nope].view(torch.float8_e4m3fn)
    gathered_rope = gathered_nope_rope[..., d_nope:].contiguous().view(torch.bfloat16)
    gathered_scales = gathered_scales[..., :num_tiles].view(torch.float8_e8m0fnu)

    nope_bf16 = gathered_nope.to(torch.bfloat16)
    scales_bf16 = gathered_scales.to(torch.bfloat16)

    scales_expanded = scales_bf16.view(total_tokens, topk, num_tiles, 1).expand(
        total_tokens, topk, num_tiles, tile_size).reshape(total_tokens, topk, d_nope)

    output = torch.empty(total_tokens, topk, d_qk, dtype=torch.bfloat16, device=device)
    output[..., :d_nope] = nope_bf16 * scales_expanded
    output[..., d_nope:] = gathered_rope

    output[invalid_mask] = 0
    return output

def gather_dequant_fp8_v32_pytorch(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    invalid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """PyTorch V32 layout gather+dequant."""
    d_qk = 576
    d_nope = 512
    d_rope = 64
    tile_size = 128
    num_tiles = 4
    bytes_per_token = 656

    total_tokens, topk = indices.shape
    device = kv_cache_quantized.device
    num_blocks = kv_cache_quantized.shape[0]

    indices_clamped = torch.clamp(indices, min=0)
    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    kv_per_block = kv_cache_quantized.view(num_blocks, block_size, bytes_per_token)
    flat_block_idx = block_idx.view(-1)
    flat_offset = offset_in_block.view(-1)

    gathered_bytes = kv_per_block[flat_block_idx, flat_offset]
    gathered_bytes = gathered_bytes.view(total_tokens, topk, bytes_per_token)

    nope_fp8 = gathered_bytes[..., :d_nope].view(torch.float8_e4m3fn)
    scales = gathered_bytes[..., d_nope:d_nope + num_tiles * 4].contiguous().view(torch.float32)
    rope_bf16 = gathered_bytes[..., d_nope + num_tiles * 4:].contiguous().view(torch.bfloat16)

    output = torch.empty(total_tokens, topk, d_qk, dtype=torch.bfloat16, device=device)
    nope_f32 = nope_fp8.to(torch.float32)
    scales_expanded = scales.repeat_interleave(tile_size, dim=-1)
    output[..., :d_nope] = (nope_f32 * scales_expanded).to(torch.bfloat16)
    output[..., d_nope:] = rope_bf16

    output[invalid_mask] = 0
    return output

# ============================================================================
# Main Entry Points
# ============================================================================

def gather_dequant_fp8_model1(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    invalid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """MODEL1 layout gather+dequant - uses optimized Triton kernel with PyTorch fallback for large outputs."""
    total_tokens, topk = indices.shape
    d_qk = MODEL1_D_QK

    # Check if output would overflow int32 pointer arithmetic
    max_offset = (total_tokens - 1) * topk * d_qk + (topk - 1) * d_qk
    if max_offset > 2**31 - 1:
        # Use PyTorch fallback for large outputs
        return gather_dequant_fp8_model1_fast(kv_cache_quantized, indices, invalid_mask, block_size)

    return gather_dequant_fp8_model1_triton(kv_cache_quantized, indices, invalid_mask, block_size)

def gather_dequant_fp8_v32(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    invalid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """V32 layout gather+dequant - uses optimized Triton kernel with PyTorch fallback for large outputs."""
    total_tokens, topk = indices.shape
    d_qk = V32_D_QK

    # Check if output would overflow int32 pointer arithmetic
    # stride_out_t = topk * d_qk, max_offset = (total_tokens-1) * stride_out_t + (topk-1) * d_qk
    max_offset = (total_tokens - 1) * topk * d_qk + (topk - 1) * d_qk
    if max_offset > 2**31 - 1:
        # Use PyTorch fallback for large outputs
        return gather_dequant_fp8_v32_pytorch(kv_cache_quantized, indices, invalid_mask, block_size)

    return gather_dequant_fp8_v32_triton(kv_cache_quantized, indices, invalid_mask, block_size)

# ============================================================================
# Attention Kernel
# ============================================================================

@triton.jit
def _fused_sparse_decode_kernel_dual_scope(
    Q,
    KV_Main, Mask_Main,
    KV_Extra, Mask_Extra,
    AttnSink,
    Output, LSE,
    sm_scale,
    total_tokens,
    h_q,
    topk_main,
    topk_extra,
    d_qk,
    d_v,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kv_main_t, stride_kv_main_k, stride_kv_main_d,
    stride_mask_main_t, stride_mask_main_k,
    stride_kv_extra_t, stride_kv_extra_k, stride_kv_extra_d,
    stride_mask_extra_t, stride_mask_extra_k,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    HAS_EXTRA_KV: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused attention kernel processing both KV scopes."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    NEG_INF = float("-inf")
    POS_INF = float("+inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    q_base = Q + pid_t * stride_q_t

    kv_main_base = KV_Main + pid_t * stride_kv_main_t
    mask_main_base = Mask_Main + pid_t * stride_mask_main_t

    for n_start in range(0, topk_main, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk_main

        mask_ptrs = mask_main_base + offs_n * stride_mask_main_k
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True)
        valid = mask_n & ~invalid

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, d_qk, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d_qk

            q_ptrs = q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
            q_chunk = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

            k_ptrs = kv_main_base + offs_n[:, None] * stride_kv_main_k + offs_d[None, :] * stride_kv_main_d
            k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

            qk += tl.dot(q_chunk, tl.trans(k_chunk)).to(tl.float32)

        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, NEG_INF)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
        l_new = alpha * l_i + tl.sum(p, axis=1)
        p_bf16 = p.to(tl.bfloat16)

        offs_v = tl.arange(0, BLOCK_D)
        v_ptrs = kv_main_base + offs_n[:, None] * stride_kv_main_k + offs_v[None, :] * stride_kv_main_d
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_main_base + offs_n[:, None] * stride_kv_main_k + offs_v[None, :] * stride_kv_main_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_main_base + offs_n[:, None] * stride_kv_main_k + offs_v[None, :] * stride_kv_main_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_main_base + offs_n[:, None] * stride_kv_main_k + offs_v[None, :] * stride_kv_main_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        m_i = m_new
        l_i = l_new

    if HAS_EXTRA_KV:
        kv_extra_base = KV_Extra + pid_t * stride_kv_extra_t
        mask_extra_base = Mask_Extra + pid_t * stride_mask_extra_t

        for n_start in range(0, topk_extra, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < topk_extra

            mask_ptrs = mask_extra_base + offs_n * stride_mask_extra_k
            invalid = tl.load(mask_ptrs, mask=mask_n, other=True)
            valid = mask_n & ~invalid

            qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

            for d_start in range(0, d_qk, BLOCK_D):
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < d_qk

                q_ptrs = q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
                q_chunk = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

                k_ptrs = kv_extra_base + offs_n[:, None] * stride_kv_extra_k + offs_d[None, :] * stride_kv_extra_d
                k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

                qk += tl.dot(q_chunk, tl.trans(k_chunk)).to(tl.float32)

            qk = qk * sm_scale
            qk = tl.where(valid[None, :], qk, NEG_INF)

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
            p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
            l_new = alpha * l_i + tl.sum(p, axis=1)
            p_bf16 = p.to(tl.bfloat16)

            offs_v = tl.arange(0, BLOCK_D)
            v_ptrs = kv_extra_base + offs_n[:, None] * stride_kv_extra_k + offs_v[None, :] * stride_kv_extra_d
            v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
            acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

            offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
            v_ptrs = kv_extra_base + offs_n[:, None] * stride_kv_extra_k + offs_v[None, :] * stride_kv_extra_d
            v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
            acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

            offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
            v_ptrs = kv_extra_base + offs_n[:, None] * stride_kv_extra_k + offs_v[None, :] * stride_kv_extra_d
            v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
            acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

            offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
            v_ptrs = kv_extra_base + offs_n[:, None] * stride_kv_extra_k + offs_v[None, :] * stride_kv_extra_d
            v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
            acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

            m_i = m_new
            l_i = l_new

    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    is_lonely_q = (l_i == 0.0)

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    lse = tl.where(is_lonely_q, POS_INF, lse)

    tl.store(LSE + pid_t * stride_lse_t + offs_h * stride_lse_h, lse, mask=mask_h)

    o_base = Output + pid_t * stride_o_t
    offs_v = tl.arange(0, BLOCK_D)
    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d, acc_0.to(tl.bfloat16), mask=mask_h[:, None])
    offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d, acc_1.to(tl.bfloat16), mask=mask_h[:, None] & (offs_v[None, :] < d_v))
    offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d, acc_2.to(tl.bfloat16), mask=mask_h[:, None] & (offs_v[None, :] < d_v))
    offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d, acc_3.to(tl.bfloat16), mask=mask_h[:, None] & (offs_v[None, :] < d_v))

def _run_dual_scope_attention(q_reshaped, kv_main, mask_main, kv_extra, mask_extra,
                               d_v, sm_scale, total_tokens, h_q, topk_main, topk_extra, d_qk,
                               attn_sink=None):
    output = torch.empty((total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q_reshaped.device)
    lse = torch.empty((total_tokens, h_q), dtype=torch.float32, device=q_reshaped.device)

    grid = (total_tokens, triton.cdiv(h_q, BLOCK_H))
    HAS_EXTRA_KV = kv_extra is not None
    HAS_ATTN_SINK = attn_sink is not None

    # Optimization: Reuse existing tensors instead of allocating dummy tensors
    # When HAS_EXTRA_KV=False, the kernel never accesses kv_extra/mask_extra
    # When HAS_ATTN_SINK=False, the kernel never accesses attn_sink_tensor
    # So we can safely reuse kv_main/mask_main as placeholders to avoid allocator pressure
    if not HAS_EXTRA_KV:
        kv_extra = kv_main  # Reuse kv_main as placeholder (not accessed when HAS_EXTRA_KV=False)
        mask_extra = mask_main  # Reuse mask_main as placeholder
        topk_extra = 0

    # Reuse lse tensor as placeholder for attn_sink when not needed (lse is already float32)
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1]

    _fused_sparse_decode_kernel_dual_scope[grid](
        q_reshaped,
        kv_main, mask_main,
        kv_extra, mask_extra,
        attn_sink_tensor,
        output, lse,
        sm_scale, total_tokens, h_q, topk_main, topk_extra, d_qk, d_v,
        q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2),
        kv_main.stride(0), kv_main.stride(1), kv_main.stride(2),
        mask_main.stride(0), mask_main.stride(1),
        kv_extra.stride(0), kv_extra.stride(1), kv_extra.stride(2) if HAS_EXTRA_KV else 1,
        mask_extra.stride(0), mask_extra.stride(1) if HAS_EXTRA_KV else 1,
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_EXTRA_KV=HAS_EXTRA_KV,
        HAS_ATTN_SINK=HAS_ATTN_SINK,
        BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=1,
    )
    return output, lse

def triton_sparse_attn_decode(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized sparse attention decode with dual-scope kernel."""
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q

    def process_kv_scope(scope):
        assert scope.indices_in_kvcache is not None
        topk = scope.indices_in_kvcache.size(-1)
        block_size = scope.blocked_k.shape[1]
        invalid_mask = scope.indices_in_kvcache == -1
        if scope.topk_length is not None:
            invalid_mask = invalid_mask | (
                torch.arange(0, topk, device=q.device).view(1, 1, topk).broadcast_to(b, s_q, topk)
                >= scope.topk_length.view(b, 1, 1)
            )
        indices_reshaped = scope.indices_in_kvcache.reshape(total_tokens, topk)
        invalid_mask_reshaped = invalid_mask.reshape(total_tokens, topk)

        if scope.blocked_k_quantized is not None:
            if d_qk == 576:
                gathered_kv = gather_dequant_fp8_v32(scope.blocked_k_quantized, indices_reshaped, invalid_mask_reshaped, block_size)
            elif d_qk == 512:
                gathered_kv = gather_dequant_fp8_model1(scope.blocked_k_quantized, indices_reshaped, invalid_mask_reshaped, block_size)
            else:
                raise ValueError(f"Unsupported d_qk: {d_qk}")
        else:
            indices_clamped = torch.clamp(indices_reshaped, min=0)
            gathered_kv = scope.blocked_k.view(-1, d_qk).index_select(0, indices_clamped.view(-1)).view(total_tokens, topk, d_qk).to(torch.bfloat16)
            gathered_kv[invalid_mask_reshaped] = 0
        return gathered_kv, invalid_mask_reshaped

    gathered_kv_main, invalid_mask_main = process_kv_scope(kv_scope)
    topk_main = gathered_kv_main.shape[1]

    gathered_kv_extra = None
    invalid_mask_extra = None
    topk_extra = 0

    if extra_kv_scope is not None:
        gathered_kv_extra, invalid_mask_extra = process_kv_scope(extra_kv_scope)
        topk_extra = gathered_kv_extra.shape[1]

    # Use nan_to_num for efficient in-place NaN cleaning (avoids zeros_like allocation)
    gathered_kv_main = torch.nan_to_num(gathered_kv_main, nan=0.0)
    if gathered_kv_extra is not None:
        gathered_kv_extra = torch.nan_to_num(gathered_kv_extra, nan=0.0)

    q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk)

    # Ensure q_reshaped is contiguous (input q may have non-standard layout)
    if not q_reshaped.is_contiguous():
        q_reshaped = q_reshaped.contiguous()

    # Assert contiguity for tensors that should always be contiguous
    assert gathered_kv_main.is_contiguous(), "gathered_kv_main should be contiguous"
    assert invalid_mask_main.is_contiguous(), "invalid_mask_main should be contiguous"
    if gathered_kv_extra is not None:
        assert gathered_kv_extra.is_contiguous(), "gathered_kv_extra should be contiguous"
        assert invalid_mask_extra.is_contiguous(), "invalid_mask_extra should be contiguous"

    total_topk = topk_main + topk_extra

    if total_topk <= 8192:
        output, lse = _run_dual_scope_attention(
            q_reshaped, gathered_kv_main, invalid_mask_main,
            gathered_kv_extra, invalid_mask_extra,
            d_v, sm_scale, total_tokens, h_q, topk_main, topk_extra, d_qk,
            attn_sink=attn_sink
        )
    else:
        if gathered_kv_extra is not None:
            gathered_kv = torch.cat([gathered_kv_main, gathered_kv_extra], dim=1)
            invalid_mask_reshaped = torch.cat([invalid_mask_main, invalid_mask_extra], dim=1)
        else:
            gathered_kv = gathered_kv_main
            invalid_mask_reshaped = invalid_mask_main

        attn_weight = q_reshaped.float() @ gathered_kv.float().transpose(-1, -2)
        attn_weight *= sm_scale
        attn_weight[invalid_mask_reshaped.unsqueeze(1).broadcast_to(total_tokens, h_q, total_topk)] = float("-inf")
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
        output = (attn_weight @ gathered_kv.float()[..., :d_v]).to(torch.bfloat16)

        output = output.view(b, s_q, h_q, d_v)
        lse = lse.view(b, s_q, h_q)
        if attn_sink is not None:
            output = output.float()
            output *= (1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))).unsqueeze(-1)
            output = output.to(torch.bfloat16)
        lonely_q_mask = (lse == float("-inf"))
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(b, s_q, h_q, d_v)] = 0.0
        lse[lonely_q_mask] = float("+inf")
        return output, lse.transpose(1, 2)

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)
