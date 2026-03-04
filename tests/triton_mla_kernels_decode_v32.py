"""
Triton MLA Decode Kernels for V3.2 (d_qk=576).

This module contains V3.2-specific gather+dequant kernels and the main
sparse attention decode entry point for V3.2.

Optimized version with:
- Triton cache persistence
- Batched scale loading (preload all 4 scales)
- Unrolled tile processing
- Fixed kernel for small workloads
- Extended autotune configurations
"""

import os
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

from triton_mla_kernels_decode_common import (
    _get_workload_size_category,
    run_unified_attention,
    run_chunked_attention_triton,
    slice_kv_scope_for_tokens,
    compute_token_ranges,
)

# Enable Triton autotune cache persistence
TRITON_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".triton_cache")
os.makedirs(TRITON_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", TRITON_CACHE_DIR)

# Constants for V32 layout
V32_D_QK = 576          # Total query/key dimension
V32_D_NOPE = 512        # Non-positional embedding dimension
V32_D_ROPE = 64         # Rotary positional embedding dimension
V32_TILE_SIZE = 128     # Size of each tile for processing
V32_NUM_TILES = 4       # Number of tiles (D_NOPE / TILE_SIZE = 512 / 128 = 4)
V32_BYTES_PER_TOKEN = 656  # Total bytes per token in KV cache

# Derived constants for V32
V32_CHUNK_SIZE = 64     # Each tile is processed in 2 chunks of 64 elements
V32_SCALE_BYTES = 4     # Each scale is 4 bytes (float32)
V32_NUM_SCALES = 4      # Number of scales (one per tile)
V32_TOTAL_SCALE_BYTES = V32_NUM_SCALES * V32_SCALE_BYTES  # 16 bytes total for scales

# Numeric constants
BF16_MAX = 65504.0      # Maximum value for bfloat16 (used for clamping)

# Performance tuning thresholds (empirically determined)
# These thresholds balance kernel launch overhead vs. computation efficiency
#
# V32_USE_FUSED_THRESHOLD: Use 1D fused kernel below this element count
#   Rationale: Single kernel launch reduces overhead for small/medium workloads
#   Value 150K determined by benchmarking on typical production workloads
V32_USE_FUSED_THRESHOLD = 150000
#
# V32_USE_FIXED_KERNEL_THRESHOLD: Use fixed BLOCK_TK=128 kernel below this
#   Rationale: Avoids autotune overhead for small workloads where fixed config
#   performs well. Value 32K balances autotune benefit vs. overhead
V32_USE_FIXED_KERNEL_THRESHOLD = 32768



# ============================================================================
# V32 Gather+Dequant Kernels - Optimized with Batched Scale Loading
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_TK': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_TK': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_TK': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_TK': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_TK': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_TK': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_TK': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_TK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_TK': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TK': 256}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_TK': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_TK': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TK': 512}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_TK': 512}, num_warps=8, num_stages=2),
    ],
    key=['total_tokens', 'topk', 'workload_size_cat'],
)
@triton.jit
def _gather_dequant_v32_kernel(
    KV_Cache,
    Indices,
    TopkLength,
    OutputKV,
    OutputMask,
    total_tokens,
    topk,
    num_blocks,
    block_size,
    workload_size_cat,
    k_offset,
    s_q,
    stride_kv_block,
    stride_kv_token,
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_k, stride_out_d,
    stride_mask_t, stride_mask_k,
    BLOCK_TK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
):
    """Optimized gather + dequant kernel with batched scale loading for V32 layout."""
    pid = tl.program_id(0)
    num_tk = total_tokens * topk

    offs_tk = pid * BLOCK_TK + tl.arange(0, BLOCK_TK)
    mask_tk = offs_tk < num_tk

    t_idx = offs_tk // topk
    k_idx = offs_tk % topk

    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k
    indices = tl.load(idx_ptrs, mask=mask_tk, other=-1)

    is_invalid = indices == -1

    if HAS_TOPK_LENGTH:
        batch_idx = t_idx // s_q
        topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk)
        is_invalid = is_invalid | (k_idx >= topk_len)

    mask_out_ptrs = OutputMask + t_idx * stride_mask_t + (k_idx + k_offset) * stride_mask_k
    tl.store(mask_out_ptrs, is_invalid, mask=mask_tk)

    valid_mask = mask_tk & ~is_invalid
    indices_clamped = tl.maximum(indices, 0)

    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    block_idx_64 = block_idx.to(tl.int64)
    offset_in_block_64 = offset_in_block.to(tl.int64)

    kv_base_ptrs = KV_Cache + block_idx_64 * stride_kv_block + offset_in_block_64 * stride_kv_token

    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64)
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64)
    out_base_ptrs = OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64

    # Preload all 4 scales at once (each scale is 4 bytes float32)
    scale_base = D_NOPE

    # Load scale 0 (4 bytes)
    scale_b0_0 = tl.load(kv_base_ptrs + scale_base, mask=valid_mask, other=0)
    scale_b1_0 = tl.load(kv_base_ptrs + scale_base + 1, mask=valid_mask, other=0)
    scale_b2_0 = tl.load(kv_base_ptrs + scale_base + 2, mask=valid_mask, other=0)
    scale_b3_0 = tl.load(kv_base_ptrs + scale_base + 3, mask=valid_mask, other=0)
    scale_uint32_0 = (scale_b0_0.to(tl.uint32) | (scale_b1_0.to(tl.uint32) << 8) |
                      (scale_b2_0.to(tl.uint32) << 16) | (scale_b3_0.to(tl.uint32) << 24))
    scale_f32_0 = scale_uint32_0.to(tl.float32, bitcast=True)

    # Load scale 1
    scale_b0_1 = tl.load(kv_base_ptrs + scale_base + 4, mask=valid_mask, other=0)
    scale_b1_1 = tl.load(kv_base_ptrs + scale_base + 5, mask=valid_mask, other=0)
    scale_b2_1 = tl.load(kv_base_ptrs + scale_base + 6, mask=valid_mask, other=0)
    scale_b3_1 = tl.load(kv_base_ptrs + scale_base + 7, mask=valid_mask, other=0)
    scale_uint32_1 = (scale_b0_1.to(tl.uint32) | (scale_b1_1.to(tl.uint32) << 8) |
                      (scale_b2_1.to(tl.uint32) << 16) | (scale_b3_1.to(tl.uint32) << 24))
    scale_f32_1 = scale_uint32_1.to(tl.float32, bitcast=True)

    # Load scale 2
    scale_b0_2 = tl.load(kv_base_ptrs + scale_base + 8, mask=valid_mask, other=0)
    scale_b1_2 = tl.load(kv_base_ptrs + scale_base + 9, mask=valid_mask, other=0)
    scale_b2_2 = tl.load(kv_base_ptrs + scale_base + 10, mask=valid_mask, other=0)
    scale_b3_2 = tl.load(kv_base_ptrs + scale_base + 11, mask=valid_mask, other=0)
    scale_uint32_2 = (scale_b0_2.to(tl.uint32) | (scale_b1_2.to(tl.uint32) << 8) |
                      (scale_b2_2.to(tl.uint32) << 16) | (scale_b3_2.to(tl.uint32) << 24))
    scale_f32_2 = scale_uint32_2.to(tl.float32, bitcast=True)

    # Load scale 3
    scale_b0_3 = tl.load(kv_base_ptrs + scale_base + 12, mask=valid_mask, other=0)
    scale_b1_3 = tl.load(kv_base_ptrs + scale_base + 13, mask=valid_mask, other=0)
    scale_b2_3 = tl.load(kv_base_ptrs + scale_base + 14, mask=valid_mask, other=0)
    scale_b3_3 = tl.load(kv_base_ptrs + scale_base + 15, mask=valid_mask, other=0)
    scale_uint32_3 = (scale_b0_3.to(tl.uint32) | (scale_b1_3.to(tl.uint32) << 8) |
                      (scale_b2_3.to(tl.uint32) << 16) | (scale_b3_3.to(tl.uint32) << 24))
    scale_f32_3 = scale_uint32_3.to(tl.float32, bitcast=True)

    offs_d = tl.arange(0, 64)  # CHUNK_SIZE: each tile processed in 2 chunks of 64

    # Pre-compute base pointers for optimization
    tile_base = kv_base_ptrs[:, None]
    out_base = out_base_ptrs[:, None]
    valid_mask_2d = valid_mask[:, None]
    is_invalid_2d = is_invalid[:, None]
    mask_tk_2d = mask_tk[:, None]

    # Dequantization: clamp to BF16_MAX (65504.0) to handle NaN/Inf values
    # Tile 0, chunk 0
    nope_ptrs = tile_base + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 0, chunk 1
    nope_ptrs = tile_base + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 1, chunk 0
    nope_ptrs = tile_base + TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 1, chunk 1
    nope_ptrs = tile_base + TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 2, chunk 0
    nope_ptrs = tile_base + 2*TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (2*TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 2, chunk 1
    nope_ptrs = tile_base + 2*TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (2*TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 3, chunk 0
    nope_ptrs = tile_base + 3*TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (3*TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 3, chunk 1
    nope_ptrs = tile_base + 3*TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (3*TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Process rope (bytes after scales: D_NOPE + NUM_TILES * 4 = 512 + 16 = 528)
    rope_byte_offset = D_NOPE + 16  # Skip 4 scales * 4 bytes = 16 bytes
    offs_rope = tl.arange(0, D_ROPE)

    rope_lo_ptrs = tile_base + rope_byte_offset + offs_rope[None, :] * 2
    rope_hi_ptrs = tile_base + rope_byte_offset + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where((rope_bf16 != rope_bf16) | is_invalid_2d, 0.0,
                         tl.maximum(tl.minimum(rope_bf16, 65504.0), -65504.0))

    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk_2d)


@triton.jit
def _gather_dequant_v32_kernel_fixed_128(
    KV_Cache,
    Indices,
    TopkLength,
    OutputKV,
    OutputMask,
    total_tokens,
    topk,
    num_blocks,
    block_size,
    k_offset,
    s_q,
    stride_kv_block,
    stride_kv_token,
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_k, stride_out_d,
    stride_mask_t, stride_mask_k,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
):
    """Fixed-config gather kernel with BLOCK_TK=128 for V32."""
    BLOCK_TK: tl.constexpr = 128
    pid = tl.program_id(0)
    num_tk = total_tokens * topk

    offs_tk = pid * BLOCK_TK + tl.arange(0, BLOCK_TK)
    mask_tk = offs_tk < num_tk

    t_idx = offs_tk // topk
    k_idx = offs_tk % topk

    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k
    indices = tl.load(idx_ptrs, mask=mask_tk, other=-1)

    is_invalid = indices == -1

    if HAS_TOPK_LENGTH:
        batch_idx = t_idx // s_q
        topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk)
        is_invalid = is_invalid | (k_idx >= topk_len)

    mask_out_ptrs = OutputMask + t_idx * stride_mask_t + (k_idx + k_offset) * stride_mask_k
    tl.store(mask_out_ptrs, is_invalid, mask=mask_tk)

    valid_mask = mask_tk & ~is_invalid
    indices_clamped = tl.maximum(indices, 0)

    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    block_idx_64 = block_idx.to(tl.int64)
    offset_in_block_64 = offset_in_block.to(tl.int64)

    kv_base_ptrs = KV_Cache + block_idx_64 * stride_kv_block + offset_in_block_64 * stride_kv_token

    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64)
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64)
    out_base_ptrs = OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64

    # Preload all 4 scales
    scale_base = D_NOPE

    scale_b0_0 = tl.load(kv_base_ptrs + scale_base, mask=valid_mask, other=0)
    scale_b1_0 = tl.load(kv_base_ptrs + scale_base + 1, mask=valid_mask, other=0)
    scale_b2_0 = tl.load(kv_base_ptrs + scale_base + 2, mask=valid_mask, other=0)
    scale_b3_0 = tl.load(kv_base_ptrs + scale_base + 3, mask=valid_mask, other=0)
    scale_uint32_0 = (scale_b0_0.to(tl.uint32) | (scale_b1_0.to(tl.uint32) << 8) |
                      (scale_b2_0.to(tl.uint32) << 16) | (scale_b3_0.to(tl.uint32) << 24))
    scale_f32_0 = scale_uint32_0.to(tl.float32, bitcast=True)

    scale_b0_1 = tl.load(kv_base_ptrs + scale_base + 4, mask=valid_mask, other=0)
    scale_b1_1 = tl.load(kv_base_ptrs + scale_base + 5, mask=valid_mask, other=0)
    scale_b2_1 = tl.load(kv_base_ptrs + scale_base + 6, mask=valid_mask, other=0)
    scale_b3_1 = tl.load(kv_base_ptrs + scale_base + 7, mask=valid_mask, other=0)
    scale_uint32_1 = (scale_b0_1.to(tl.uint32) | (scale_b1_1.to(tl.uint32) << 8) |
                      (scale_b2_1.to(tl.uint32) << 16) | (scale_b3_1.to(tl.uint32) << 24))
    scale_f32_1 = scale_uint32_1.to(tl.float32, bitcast=True)

    scale_b0_2 = tl.load(kv_base_ptrs + scale_base + 8, mask=valid_mask, other=0)
    scale_b1_2 = tl.load(kv_base_ptrs + scale_base + 9, mask=valid_mask, other=0)
    scale_b2_2 = tl.load(kv_base_ptrs + scale_base + 10, mask=valid_mask, other=0)
    scale_b3_2 = tl.load(kv_base_ptrs + scale_base + 11, mask=valid_mask, other=0)
    scale_uint32_2 = (scale_b0_2.to(tl.uint32) | (scale_b1_2.to(tl.uint32) << 8) |
                      (scale_b2_2.to(tl.uint32) << 16) | (scale_b3_2.to(tl.uint32) << 24))
    scale_f32_2 = scale_uint32_2.to(tl.float32, bitcast=True)

    scale_b0_3 = tl.load(kv_base_ptrs + scale_base + 12, mask=valid_mask, other=0)
    scale_b1_3 = tl.load(kv_base_ptrs + scale_base + 13, mask=valid_mask, other=0)
    scale_b2_3 = tl.load(kv_base_ptrs + scale_base + 14, mask=valid_mask, other=0)
    scale_b3_3 = tl.load(kv_base_ptrs + scale_base + 15, mask=valid_mask, other=0)
    scale_uint32_3 = (scale_b0_3.to(tl.uint32) | (scale_b1_3.to(tl.uint32) << 8) |
                      (scale_b2_3.to(tl.uint32) << 16) | (scale_b3_3.to(tl.uint32) << 24))
    scale_f32_3 = scale_uint32_3.to(tl.float32, bitcast=True)

    offs_d = tl.arange(0, 64)  # CHUNK_SIZE: each tile processed in 2 chunks of 64

    # Pre-compute base pointers for optimization
    tile_base = kv_base_ptrs[:, None]
    out_base = out_base_ptrs[:, None]
    valid_mask_2d = valid_mask[:, None]
    is_invalid_2d = is_invalid[:, None]
    mask_tk_2d = mask_tk[:, None]

    # Dequantization: clamp to BF16_MAX (65504.0) to handle NaN/Inf values
    # Tile 0, chunk 0
    nope_ptrs = tile_base + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 0, chunk 1
    nope_ptrs = tile_base + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 1, chunk 0
    nope_ptrs = tile_base + TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 1, chunk 1
    nope_ptrs = tile_base + TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 2, chunk 0
    nope_ptrs = tile_base + 2*TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (2*TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 2, chunk 1
    nope_ptrs = tile_base + 2*TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (2*TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 3, chunk 0
    nope_ptrs = tile_base + 3*TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (3*TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 3, chunk 1
    nope_ptrs = tile_base + 3*TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (3*TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Process rope
    rope_byte_offset = D_NOPE + 16  # Skip 4 scales * 4 bytes = 16 bytes
    offs_rope = tl.arange(0, D_ROPE)

    rope_lo_ptrs = tile_base + rope_byte_offset + offs_rope[None, :] * 2
    rope_hi_ptrs = tile_base + rope_byte_offset + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where((rope_bf16 != rope_bf16) | is_invalid_2d, 0.0,
                         tl.maximum(tl.minimum(rope_bf16, 65504.0), -65504.0))

    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk_2d)


# ============================================================================
# V32 Wrapper Functions
# ============================================================================

def gather_dequant_fp8_v32(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    output_kv: torch.Tensor,
    output_mask: torch.Tensor,
    k_offset: int = 0,
    topk_length: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> bool:
    """Unified V32 gather+dequant with optional topk_length mask."""
    total_tokens, topk = indices.shape
    num_blocks = kv_cache_quantized.shape[0]

    kv_uint8 = kv_cache_quantized.view(torch.uint8)
    stride_kv_block = kv_uint8.stride(0)
    stride_kv_token = kv_uint8.stride(1)
    workload_size_cat = _get_workload_size_category(total_tokens, topk)

    grid = lambda meta: (triton.cdiv(total_tokens * topk, meta['BLOCK_TK']),)

    topk_length_tensor = topk_length if topk_length is not None else output_mask[:1, 0]
    has_topk_length = topk_length is not None

    _gather_dequant_v32_kernel[grid](
        kv_uint8, indices, topk_length_tensor,
        output_kv, output_mask,
        total_tokens, topk, num_blocks, block_size,
        workload_size_cat, k_offset, s_q,
        stride_kv_block, stride_kv_token,
        indices.stride(0), indices.stride(1),
        output_kv.stride(0), output_kv.stride(1), output_kv.stride(2),
        output_mask.stride(0), output_mask.stride(1),
        D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
        TILE_SIZE=V32_TILE_SIZE,
        HAS_TOPK_LENGTH=has_topk_length,
    )
    return True



# ============================================================================
# V32 1D Grid Fused Gather+Dequant Kernel (Optimized - No Empty Blocks)
# Single kernel launch with 1D grid: (num_main_pids + num_extra_pids,)
# ============================================================================

@triton.jit
def _gather_dequant_v32_1d_fused_kernel(
    # Main KV cache
    KV_Cache_Main,
    Indices_Main,
    TopkLength_Main,
    # Extra KV cache
    KV_Cache_Extra,
    Indices_Extra,
    TopkLength_Extra,
    # Output
    OutputKV,
    OutputMask,
    # Dimensions
    total_tokens,
    topk_main,
    topk_extra,
    num_blocks_main,
    num_blocks_extra,
    block_size_main,
    block_size_extra,
    s_q,
    # Strides for main
    stride_kv_block_main,
    stride_kv_token_main,
    stride_idx_t_main, stride_idx_k_main,
    # Strides for extra
    stride_kv_block_extra,
    stride_kv_token_extra,
    stride_idx_t_extra, stride_idx_k_extra,
    # Output strides
    stride_out_t, stride_out_k, stride_out_d,
    stride_mask_t, stride_mask_k,
    # Grid info
    num_main_pids,
    # Constexpr
    BLOCK_TK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_TOPK_LENGTH_MAIN: tl.constexpr,
    HAS_TOPK_LENGTH_EXTRA: tl.constexpr,
):
    """1D fused gather kernel for V32 - single launch, no empty blocks.

    Grid: (num_main_pids + num_extra_pids,)
    - pid < num_main_pids: process main cache
    - pid >= num_main_pids: process extra cache
    """
    pid = tl.program_id(0)

    # Determine if this is main or extra processing
    is_main_pid = pid < num_main_pids

    # Select parameters based on pid
    if is_main_pid:
        local_pid = pid
        topk = topk_main
        k_offset = 0
        num_tk = total_tokens * topk_main
        KV_Cache = KV_Cache_Main
        Indices = Indices_Main
        TopkLength = TopkLength_Main
        block_size = block_size_main
        stride_kv_block = stride_kv_block_main
        stride_kv_token = stride_kv_token_main
        stride_idx_t = stride_idx_t_main
        stride_idx_k = stride_idx_k_main
    else:
        local_pid = pid - num_main_pids
        topk = topk_extra
        k_offset = topk_main
        num_tk = total_tokens * topk_extra
        KV_Cache = KV_Cache_Extra
        Indices = Indices_Extra
        TopkLength = TopkLength_Extra
        block_size = block_size_extra
        stride_kv_block = stride_kv_block_extra
        stride_kv_token = stride_kv_token_extra
        stride_idx_t = stride_idx_t_extra
        stride_idx_k = stride_idx_k_extra

    # Compute element indices for this block
    offs_tk = local_pid * BLOCK_TK + tl.arange(0, BLOCK_TK)
    mask_tk = offs_tk < num_tk

    t_idx = offs_tk // topk
    k_idx = offs_tk % topk

    # Load indices
    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k
    indices = tl.load(idx_ptrs, mask=mask_tk, other=-1)

    is_invalid = indices == -1

    # Handle topk_length
    batch_idx = t_idx // s_q
    if is_main_pid:
        if HAS_TOPK_LENGTH_MAIN:
            topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk)
            is_invalid = is_invalid | (k_idx >= topk_len)
    else:
        if HAS_TOPK_LENGTH_EXTRA:
            topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk)
            is_invalid = is_invalid | (k_idx >= topk_len)

    # Store mask
    mask_out_ptrs = OutputMask + t_idx * stride_mask_t + (k_idx + k_offset) * stride_mask_k
    tl.store(mask_out_ptrs, is_invalid, mask=mask_tk)

    valid_mask = mask_tk & ~is_invalid
    indices_clamped = tl.maximum(indices, 0)

    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    block_idx_64 = block_idx.to(tl.int64)
    offset_in_block_64 = offset_in_block.to(tl.int64)

    kv_base_ptrs = KV_Cache + block_idx_64 * stride_kv_block + offset_in_block_64 * stride_kv_token

    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64)
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64)
    out_base_ptrs = OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64

    # Preload all 4 scales
    scale_base = D_NOPE
    scale_ptrs_0 = kv_base_ptrs + scale_base
    scale_uint8_0 = tl.load(scale_ptrs_0, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_1 = tl.load(scale_ptrs_0 + 1, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_2 = tl.load(scale_ptrs_0 + 2, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_3 = tl.load(scale_ptrs_0 + 3, mask=valid_mask, other=127).to(tl.uint8)

    # E8M0 scale format: exponent-only, bias=127, scale = 2^(uint8 - 127)
    scale_f32_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0)
    scale_f32_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0)
    scale_f32_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0)
    scale_f32_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0)

    offs_d = tl.arange(0, 64)  # CHUNK_SIZE: each tile processed in 2 chunks of 64

    tile_base = kv_base_ptrs[:, None]
    out_base = out_base_ptrs[:, None]
    valid_mask_2d = valid_mask[:, None]
    is_invalid_2d = is_invalid[:, None]
    mask_tk_2d = mask_tk[:, None]

    # Dequantization: clamp to BF16_MAX (65504.0) to handle NaN/Inf values
    # Tile 0, chunk 0
    nope_ptrs = tile_base + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 0, chunk 1
    nope_ptrs = tile_base + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 1, chunk 0
    nope_ptrs = tile_base + TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 1, chunk 1
    nope_ptrs = tile_base + TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 2, chunk 0
    nope_ptrs = tile_base + 2*TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (2*TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 2, chunk 1
    nope_ptrs = tile_base + 2*TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (2*TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 3, chunk 0
    nope_ptrs = tile_base + 3*TILE_SIZE + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (3*TILE_SIZE + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Tile 3, chunk 1
    nope_ptrs = tile_base + 3*TILE_SIZE + 64 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_f32 = nope_fp8.to(tl.float32)
    dequant = nope_f32 * scale_f32_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid_2d, 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (3*TILE_SIZE + 64 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk_2d)

    # Process rope (bytes after scales: D_NOPE + NUM_TILES * 4 = 512 + 16 = 528)
    rope_byte_offset = D_NOPE + 16  # Skip 4 scales * 4 bytes = 16 bytes
    offs_rope = tl.arange(0, D_ROPE)

    rope_lo_ptrs = tile_base + rope_byte_offset + offs_rope[None, :] * 2
    rope_hi_ptrs = tile_base + rope_byte_offset + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask_2d, other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where((rope_bf16 != rope_bf16) | is_invalid_2d, 0.0,
                         tl.maximum(tl.minimum(rope_bf16, 65504.0), -65504.0))

    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk_2d)


def truly_fused_gather_dequant_fp8_v32(
    kv_cache_main, indices_main, block_size_main, topk_length_main,
    kv_cache_extra, indices_extra, block_size_extra, topk_length_extra,
    output_kv, output_mask, s_q=1,
):
    """Truly fused V32 gather - single kernel launch with 1D grid (no empty blocks)."""
    total_tokens, topk_main = indices_main.shape
    topk_extra = indices_extra.shape[1]
    b = total_tokens // s_q

    kv_uint8_main = kv_cache_main.view(torch.uint8)
    stride_kv_block_main = kv_uint8_main.stride(0)
    stride_kv_token_main = kv_uint8_main.stride(1)
    num_blocks_main = kv_cache_main.shape[0]

    kv_uint8_extra = kv_cache_extra.view(torch.uint8)
    stride_kv_block_extra = kv_uint8_extra.stride(0)
    stride_kv_token_extra = kv_uint8_extra.stride(1)
    num_blocks_extra = kv_cache_extra.shape[0]

    has_topk_length_main = topk_length_main is not None
    has_topk_length_extra = topk_length_extra is not None

    # Always use int32 tensors for topk_length
    if has_topk_length_main:
        topk_length_main_tensor = topk_length_main
    else:
        topk_length_main_tensor = torch.full((b,), topk_main, dtype=torch.int32, device=indices_main.device)

    if has_topk_length_extra:
        topk_length_extra_tensor = topk_length_extra
    else:
        topk_length_extra_tensor = torch.full((b,), topk_extra, dtype=torch.int32, device=indices_extra.device)

    stride_idx_t_main, stride_idx_k_main = indices_main.stride(0), indices_main.stride(1)
    stride_idx_t_extra, stride_idx_k_extra = indices_extra.stride(0), indices_extra.stride(1)
    stride_out_t, stride_out_k, stride_out_d = output_kv.stride(0), output_kv.stride(1), output_kv.stride(2)
    stride_mask_t, stride_mask_k = output_mask.stride(0), output_mask.stride(1)

    BLOCK_TK = 128

    # Calculate grid sizes - 1D grid with exact number of needed blocks
    num_elements_main = total_tokens * topk_main
    num_elements_extra = total_tokens * topk_extra
    num_main_pids = triton.cdiv(num_elements_main, BLOCK_TK)
    num_extra_pids = triton.cdiv(num_elements_extra, BLOCK_TK)

    # 1D grid: (num_main_pids + num_extra_pids,) - no empty blocks!
    grid = (num_main_pids + num_extra_pids,)

    _gather_dequant_v32_1d_fused_kernel[grid](
        kv_uint8_main, indices_main, topk_length_main_tensor,
        kv_uint8_extra, indices_extra, topk_length_extra_tensor,
        output_kv, output_mask,
        total_tokens, topk_main, topk_extra,
        num_blocks_main, num_blocks_extra,
        block_size_main, block_size_extra,
        s_q,
        stride_kv_block_main, stride_kv_token_main,
        stride_idx_t_main, stride_idx_k_main,
        stride_kv_block_extra, stride_kv_token_extra,
        stride_idx_t_extra, stride_idx_k_extra,
        stride_out_t, stride_out_k, stride_out_d,
        stride_mask_t, stride_mask_k,
        num_main_pids,
        BLOCK_TK=BLOCK_TK,
        D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
        TILE_SIZE=V32_TILE_SIZE,
        HAS_TOPK_LENGTH_MAIN=has_topk_length_main,
        HAS_TOPK_LENGTH_EXTRA=has_topk_length_extra,
        num_warps=8, num_stages=2,
    )
    return True


def fused_gather_dequant_fp8_v32(
    kv_cache_main, indices_main, block_size_main, topk_length_main,
    kv_cache_extra, indices_extra, block_size_extra, topk_length_extra,
    output_kv, output_mask, s_q=1,
):
    """Fused V32 gather - uses 1D fused kernel for small/medium workloads."""
    total_tokens, topk_main = indices_main.shape
    topk_extra = indices_extra.shape[1]
    total_elements = total_tokens * (topk_main + topk_extra)

    # Use 1D fused kernel for workloads below threshold (single kernel launch, no empty blocks)
    USE_FUSED_THRESHOLD = V32_USE_FUSED_THRESHOLD

    if total_elements < USE_FUSED_THRESHOLD:
        return truly_fused_gather_dequant_fp8_v32(
            kv_cache_main, indices_main, block_size_main, topk_length_main,
            kv_cache_extra, indices_extra, block_size_extra, topk_length_extra,
            output_kv, output_mask, s_q,
        )

    # For larger workloads, use two separate kernel launches with autotuning

    kv_uint8_main = kv_cache_main.view(torch.uint8)
    stride_kv_block_main = kv_uint8_main.stride(0)
    stride_kv_token_main = kv_uint8_main.stride(1)
    num_blocks_main = kv_cache_main.shape[0]

    kv_uint8_extra = kv_cache_extra.view(torch.uint8)
    stride_kv_block_extra = kv_uint8_extra.stride(0)
    stride_kv_token_extra = kv_uint8_extra.stride(1)
    num_blocks_extra = kv_cache_extra.shape[0]

    has_topk_length_main = topk_length_main is not None
    has_topk_length_extra = topk_length_extra is not None
    topk_length_main_tensor = topk_length_main if has_topk_length_main else output_mask[:1, 0]
    topk_length_extra_tensor = topk_length_extra if has_topk_length_extra else output_mask[:1, 0]

    stride_idx_t_main, stride_idx_k_main = indices_main.stride(0), indices_main.stride(1)
    stride_idx_t_extra, stride_idx_k_extra = indices_extra.stride(0), indices_extra.stride(1)
    stride_out_t, stride_out_k, stride_out_d = output_kv.stride(0), output_kv.stride(1), output_kv.stride(2)
    stride_mask_t, stride_mask_k = output_mask.stride(0), output_mask.stride(1)

    total_elements_main = total_tokens * topk_main
    total_elements_extra = total_tokens * topk_extra

    USE_FIXED_KERNEL_THRESHOLD = V32_USE_FIXED_KERNEL_THRESHOLD

    if total_elements_main < USE_FIXED_KERNEL_THRESHOLD:
        grid_main = (triton.cdiv(total_elements_main, 128),)
        _gather_dequant_v32_kernel_fixed_128[grid_main](
            kv_uint8_main, indices_main, topk_length_main_tensor,
            output_kv, output_mask,
            total_tokens, topk_main, num_blocks_main, block_size_main,
            0, s_q, stride_kv_block_main, stride_kv_token_main,
            stride_idx_t_main, stride_idx_k_main,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
            TILE_SIZE=V32_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_main,
            num_warps=8, num_stages=2,
        )
    else:
        workload_main = _get_workload_size_category(total_tokens, topk_main)
        grid_main = lambda meta: (triton.cdiv(total_elements_main, meta['BLOCK_TK']),)
        _gather_dequant_v32_kernel[grid_main](
            kv_uint8_main, indices_main, topk_length_main_tensor,
            output_kv, output_mask,
            total_tokens, topk_main, num_blocks_main, block_size_main,
            workload_main, 0, s_q, stride_kv_block_main, stride_kv_token_main,
            stride_idx_t_main, stride_idx_k_main,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
            TILE_SIZE=V32_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_main,
        )

    if total_elements_extra < USE_FIXED_KERNEL_THRESHOLD:
        grid_extra = (triton.cdiv(total_elements_extra, 128),)
        _gather_dequant_v32_kernel_fixed_128[grid_extra](
            kv_uint8_extra, indices_extra, topk_length_extra_tensor,
            output_kv, output_mask,
            total_tokens, topk_extra, num_blocks_extra, block_size_extra,
            topk_main, s_q, stride_kv_block_extra, stride_kv_token_extra,
            stride_idx_t_extra, stride_idx_k_extra,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
            TILE_SIZE=V32_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_extra,
            num_warps=8, num_stages=2,
        )
    else:
        workload_extra = _get_workload_size_category(total_tokens, topk_extra)
        grid_extra = lambda meta: (triton.cdiv(total_elements_extra, meta['BLOCK_TK']),)
        _gather_dequant_v32_kernel[grid_extra](
            kv_uint8_extra, indices_extra, topk_length_extra_tensor,
            output_kv, output_mask,
            total_tokens, topk_extra, num_blocks_extra, block_size_extra,
            workload_extra, topk_main, s_q, stride_kv_block_extra, stride_kv_token_extra,
            stride_idx_t_extra, stride_idx_k_extra,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
            TILE_SIZE=V32_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_extra,
        )
    return True


# ============================================================================
# V32 Main Entry Point
# ============================================================================

def triton_sparse_attn_decode_v32(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention decode for V3.2 (d_qk=576)."""
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == V32_D_QK, f"Expected d_qk={V32_D_QK} for V3.2, got {d_qk}"
    total_tokens = b * s_q

    topk_main = kv_scope.indices_in_kvcache.size(-1)
    topk_extra = extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0
    total_topk = topk_main + topk_extra

    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk)

    if len(token_ranges) == 1:
        return _triton_sparse_attn_decode_v32_impl(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
        )

    outputs = []
    lses = []

    for start_t, end_t in token_ranges:
        chunk_tokens = end_t - start_t
        q_chunk = q.reshape(total_tokens, h_q, d_qk)[start_t:end_t]
        q_input = q_chunk.reshape(chunk_tokens, 1, h_q, d_qk)
        chunk_kv_scope = slice_kv_scope_for_tokens(kv_scope, start_t, end_t, s_q)
        chunk_extra_kv_scope = slice_kv_scope_for_tokens(extra_kv_scope, start_t, end_t, s_q)

        chunk_out, chunk_lse = _triton_sparse_attn_decode_v32_impl(
            q_input, chunk_kv_scope, chunk_extra_kv_scope, sm_scale, d_v, attn_sink
        )

        outputs.append(chunk_out.reshape(chunk_tokens, h_q, d_v))
        lses.append(chunk_lse.reshape(chunk_tokens, h_q))

    output = torch.cat(outputs, dim=0).reshape(b, s_q, h_q, d_v)
    lse = torch.cat(lses, dim=0).reshape(b, s_q, h_q).transpose(1, 2)

    return output, lse


def _triton_sparse_attn_decode_v32_impl(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal implementation of sparse attention decode for V3.2.

    Assumes KV cache is always FP8 quantized (blocked_k_quantized is not None).
    """
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q

    topk_main = kv_scope.indices_in_kvcache.size(-1)
    topk_extra = extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0
    total_topk = topk_main + topk_extra

    gathered_kv = torch.empty(total_tokens, total_topk, d_qk, dtype=torch.bfloat16, device=q.device)
    invalid_mask = torch.empty(total_tokens, total_topk, dtype=torch.bool, device=q.device)

    block_size_main = kv_scope.blocked_k.shape[1]
    indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)

    if extra_kv_scope is not None:
        # Fused gather for both main and extra scope
        block_size_extra = extra_kv_scope.blocked_k.shape[1]
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)
        fused_gather_dequant_fp8_v32(
            kv_scope.blocked_k_quantized, indices_main, block_size_main, kv_scope.topk_length,
            extra_kv_scope.blocked_k_quantized, indices_extra, block_size_extra, extra_kv_scope.topk_length,
            gathered_kv, invalid_mask, s_q)
    else:
        # Single gather for main scope only
        gather_dequant_fp8_v32(
            kv_scope.blocked_k_quantized, indices_main, block_size_main,
            gathered_kv, invalid_mask, 0, kv_scope.topk_length, s_q)

    q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk)

    if not q_reshaped.is_contiguous():
        q_reshaped = q_reshaped.contiguous()

    if total_topk <= 65536:
        output, lse = run_unified_attention(
            q_reshaped, gathered_kv, invalid_mask,
            d_v, sm_scale, total_tokens, h_q, total_topk, d_qk,
            attn_sink=attn_sink
        )
    else:
        output, lse = run_chunked_attention_triton(
            q_reshaped, gathered_kv, invalid_mask,
            d_v, sm_scale, total_tokens, h_q, total_topk, d_qk,
            attn_sink=attn_sink, chunk_size=32768
        )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)
