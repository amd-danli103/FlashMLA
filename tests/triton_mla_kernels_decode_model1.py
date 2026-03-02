"""
Triton MLA Decode Kernels for MODEL1 (d_qk=512).

This module contains MODEL1-specific gather+dequant kernels and the main
sparse attention decode entry point for MODEL1.
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

# Constants for MODEL1 layout
MODEL1_D_QK = 512
MODEL1_D_NOPE = 448
MODEL1_D_ROPE = 64
MODEL1_TILE_SIZE = 64
MODEL1_NUM_TILES = 7
MODEL1_BYTES_PER_TOKEN_DATA = 576  # 448 nope + 128 rope
MODEL1_BYTES_PER_TOKEN_SCALE = 8   # 7 scales + 1 padding


# ============================================================================
# MODEL1 Gather+Dequant Kernels - Optimized with Batched Scale Loading
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
def _gather_dequant_model1_kernel(
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
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_k, stride_out_d,
    stride_mask_t, stride_mask_k,
    BLOCK_TK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    BYTES_PER_TOKEN_DATA: tl.constexpr,
    BYTES_PER_TOKEN_SCALE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
):
    """Optimized gather + dequant kernel with batched scale loading."""
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

    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block

    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
    scale_base_offset = block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE

    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64)
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64)
    out_base_ptrs = OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64

    # Load all 7 scales at once - each scale is at scale_base_offset + tile_idx
    scale_ptrs_0 = kv_block_base + scale_base_offset
    scale_ptrs_1 = kv_block_base + scale_base_offset + 1
    scale_ptrs_2 = kv_block_base + scale_base_offset + 2
    scale_ptrs_3 = kv_block_base + scale_base_offset + 3
    scale_ptrs_4 = kv_block_base + scale_base_offset + 4
    scale_ptrs_5 = kv_block_base + scale_base_offset + 5
    scale_ptrs_6 = kv_block_base + scale_base_offset + 6

    scale_uint8_0 = tl.load(scale_ptrs_0, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_1 = tl.load(scale_ptrs_1, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_2 = tl.load(scale_ptrs_2, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_3 = tl.load(scale_ptrs_3, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_4 = tl.load(scale_ptrs_4, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_5 = tl.load(scale_ptrs_5, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_6 = tl.load(scale_ptrs_6, mask=valid_mask, other=127).to(tl.uint8)

    # Convert all scales to bf16
    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16)

    offs_d = tl.arange(0, TILE_SIZE)

    # Pre-compute base pointers for optimization
    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None]
    out_base = out_base_ptrs[:, None]

    # Process tile 0
    nope_ptrs = tile_base + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 1
    tile_start_1 = TILE_SIZE
    nope_ptrs = tile_base + tile_start_1 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_1 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 2
    tile_start_2 = 2 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_2 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_2 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 3
    tile_start_3 = 3 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_3 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_3 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 4
    tile_start_4 = 4 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_4 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_4[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_4 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 5
    tile_start_5 = 5 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_5 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_5[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_5 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 6
    tile_start_6 = 6 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_6 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_6[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_6 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process rope
    offs_rope = tl.arange(0, D_ROPE)
    rope_byte_start = D_NOPE

    rope_lo_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2
    rope_hi_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where((rope_bf16 != rope_bf16) | is_invalid[:, None], 0.0,
                         tl.maximum(tl.minimum(rope_bf16, 65504.0), -65504.0))

    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk[:, None])


@triton.jit
def _gather_dequant_model1_kernel_fixed_128(
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
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_k, stride_out_d,
    stride_mask_t, stride_mask_k,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    BYTES_PER_TOKEN_DATA: tl.constexpr,
    BYTES_PER_TOKEN_SCALE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
):
    """Fixed-config gather kernel with BLOCK_TK=128 and batched scale loading."""
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

    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block

    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
    scale_base_offset = block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE

    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64)
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64)
    out_base_ptrs = OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64

    # Load all 7 scales at once
    scale_ptrs_0 = kv_block_base + scale_base_offset
    scale_ptrs_1 = kv_block_base + scale_base_offset + 1
    scale_ptrs_2 = kv_block_base + scale_base_offset + 2
    scale_ptrs_3 = kv_block_base + scale_base_offset + 3
    scale_ptrs_4 = kv_block_base + scale_base_offset + 4
    scale_ptrs_5 = kv_block_base + scale_base_offset + 5
    scale_ptrs_6 = kv_block_base + scale_base_offset + 6

    scale_uint8_0 = tl.load(scale_ptrs_0, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_1 = tl.load(scale_ptrs_1, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_2 = tl.load(scale_ptrs_2, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_3 = tl.load(scale_ptrs_3, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_4 = tl.load(scale_ptrs_4, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_5 = tl.load(scale_ptrs_5, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_6 = tl.load(scale_ptrs_6, mask=valid_mask, other=127).to(tl.uint8)

    # Convert all scales to bf16
    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16)

    offs_d = tl.arange(0, TILE_SIZE)

    # Pre-compute base pointers for optimization
    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None]
    out_base = out_base_ptrs[:, None]

    # Process tile 0
    nope_ptrs = tile_base + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 1
    tile_start_1 = TILE_SIZE
    nope_ptrs = tile_base + tile_start_1 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_1 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 2
    tile_start_2 = 2 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_2 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_2 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 3
    tile_start_3 = 3 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_3 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_3 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 4
    tile_start_4 = 4 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_4 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_4[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_4 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 5
    tile_start_5 = 5 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_5 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_5[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_5 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 6
    tile_start_6 = 6 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_6 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_6[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_6 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process rope
    offs_rope = tl.arange(0, D_ROPE)
    rope_byte_start = D_NOPE

    rope_lo_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2
    rope_hi_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where((rope_bf16 != rope_bf16) | is_invalid[:, None], 0.0,
                         tl.maximum(tl.minimum(rope_bf16, 65504.0), -65504.0))

    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk[:, None])


# ============================================================================
# MODEL1 Wrapper Functions
# ============================================================================

def gather_dequant_fp8_model1(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    output_kv: torch.Tensor,
    output_mask: torch.Tensor,
    k_offset: int = 0,
    topk_length: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> bool:
    """Unified MODEL1 gather+dequant with optional topk_length mask."""
    total_tokens, topk = indices.shape
    num_blocks = kv_cache_quantized.shape[0]

    kv_uint8 = kv_cache_quantized.view(torch.uint8)
    bytes_per_block = kv_uint8.shape[1] * kv_uint8.shape[2] * kv_uint8.shape[3]
    kv_flat = kv_uint8.reshape(num_blocks, bytes_per_block)

    stride_kv_block = kv_uint8.stride(0)
    workload_size_cat = _get_workload_size_category(total_tokens, topk)

    grid = lambda meta: (triton.cdiv(total_tokens * topk, meta['BLOCK_TK']),)

    topk_length_tensor = topk_length if topk_length is not None else output_mask[:1, 0]
    has_topk_length = topk_length is not None

    _gather_dequant_model1_kernel[grid](
        kv_flat, indices, topk_length_tensor,
        output_kv, output_mask,
        total_tokens, topk, num_blocks, block_size,
        workload_size_cat, k_offset, s_q, stride_kv_block,
        indices.stride(0), indices.stride(1),
        output_kv.stride(0), output_kv.stride(1), output_kv.stride(2),
        output_mask.stride(0), output_mask.stride(1),
        D_NOPE=MODEL1_D_NOPE, D_ROPE=MODEL1_D_ROPE,
        BYTES_PER_TOKEN_DATA=MODEL1_BYTES_PER_TOKEN_DATA,
        BYTES_PER_TOKEN_SCALE=MODEL1_BYTES_PER_TOKEN_SCALE,
        TILE_SIZE=MODEL1_TILE_SIZE,
        HAS_TOPK_LENGTH=has_topk_length,
    )
    return True



# ============================================================================
# Truly Fused Gather Kernel - 2D Grid Approach
# ============================================================================

# ============================================================================
# MODEL1 1D Grid Fused Gather+Dequant Kernel (Optimized - No Empty Blocks)
# Single kernel launch with 1D grid: (num_main_pids + num_extra_pids,)
# ============================================================================

@triton.jit
def _gather_dequant_model1_1d_fused_kernel(
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
    stride_idx_t_main, stride_idx_k_main,
    # Strides for extra
    stride_kv_block_extra,
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
    BYTES_PER_TOKEN_DATA: tl.constexpr,
    BYTES_PER_TOKEN_SCALE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_TOPK_LENGTH_MAIN: tl.constexpr,
    HAS_TOPK_LENGTH_EXTRA: tl.constexpr,
):
    """1D fused gather kernel - single launch, no empty blocks.

    Grid: (num_main_pids + num_extra_pids,)
    - pid < num_main_pids: process main cache
    - pid >= num_main_pids: process extra cache

    This eliminates empty blocks when main/extra topk differ significantly.
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

    # Handle topk_length - need to handle both cases
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

    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block

    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
    scale_base_offset = block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE

    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64)
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64)
    out_base_ptrs = OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64

    # Load all 7 scales
    scale_ptrs_0 = kv_block_base + scale_base_offset
    scale_uint8_0 = tl.load(scale_ptrs_0, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_1 = tl.load(scale_ptrs_0 + 1, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_2 = tl.load(scale_ptrs_0 + 2, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_3 = tl.load(scale_ptrs_0 + 3, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_4 = tl.load(scale_ptrs_0 + 4, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_5 = tl.load(scale_ptrs_0 + 5, mask=valid_mask, other=127).to(tl.uint8)
    scale_uint8_6 = tl.load(scale_ptrs_0 + 6, mask=valid_mask, other=127).to(tl.uint8)

    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16)

    offs_d = tl.arange(0, TILE_SIZE)

    # Pre-compute base pointers for optimization
    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None]
    out_base = out_base_ptrs[:, None]

    # Process tile 0
    nope_ptrs = tile_base + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_0[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 1
    tile_start_1 = TILE_SIZE
    nope_ptrs = tile_base + tile_start_1 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_1[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_1 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 2
    tile_start_2 = 2 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_2 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_2[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_2 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 3
    tile_start_3 = 3 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_3 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_3[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_3 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 4
    tile_start_4 = 4 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_4 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_4[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_4 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 5
    tile_start_5 = 5 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_5 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_5[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_5 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process tile 6
    tile_start_6 = 6 * TILE_SIZE
    nope_ptrs = tile_base + tile_start_6 + offs_d[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    nope_bf16 = nope_fp8.to(tl.bfloat16)
    dequant = nope_bf16 * scale_bf16_6[:, None]
    dequant = tl.where((dequant != dequant) | is_invalid[:, None], 0.0,
                       tl.maximum(tl.minimum(dequant, 65504.0), -65504.0))
    out_ptrs = out_base + (tile_start_6 + offs_d[None, :]) * stride_out_d
    tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    # Process rope
    offs_rope = tl.arange(0, D_ROPE)
    rope_byte_start = D_NOPE
    rope_lo_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2
    rope_hi_ptrs = tile_base + rope_byte_start + offs_rope[None, :] * 2 + 1
    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)
    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where((rope_bf16 != rope_bf16) | is_invalid[:, None], 0.0,
                         tl.maximum(tl.minimum(rope_bf16, 65504.0), -65504.0))
    out_ptrs = out_base + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk[:, None])



def _prepare_kv_cache_flat(kv_cache):
    """Helper to prepare KV cache for gather operations.

    Returns: (kv_flat, num_blocks, stride_kv_block)
    """
    kv_uint8 = kv_cache.view(torch.uint8)
    num_blocks = kv_cache.shape[0]
    bytes_per_block = kv_uint8.shape[1] * kv_uint8.shape[2] * kv_uint8.shape[3]
    kv_flat = kv_uint8.reshape(num_blocks, bytes_per_block)
    stride_kv_block = kv_uint8.stride(0)
    return kv_flat, num_blocks, stride_kv_block


def truly_fused_gather_dequant_fp8_model1(
    kv_cache_main, indices_main, block_size_main, topk_length_main,
    kv_cache_extra, indices_extra, block_size_extra, topk_length_extra,
    output_kv, output_mask, s_q=1,
):
    """Truly fused MODEL1 gather - single kernel launch with 1D grid (no empty blocks)."""
    total_tokens, topk_main = indices_main.shape
    topk_extra = indices_extra.shape[1]
    b = total_tokens // s_q  # batch size

    kv_flat_main, num_blocks_main, stride_kv_block_main = _prepare_kv_cache_flat(kv_cache_main)
    kv_flat_extra, num_blocks_extra, stride_kv_block_extra = _prepare_kv_cache_flat(kv_cache_extra)

    has_topk_length_main = topk_length_main is not None
    has_topk_length_extra = topk_length_extra is not None

    # Always use int32 tensors for topk_length to avoid type mismatch in Triton
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

    _gather_dequant_model1_1d_fused_kernel[grid](
        kv_flat_main, indices_main, topk_length_main_tensor,
        kv_flat_extra, indices_extra, topk_length_extra_tensor,
        output_kv, output_mask,
        total_tokens, topk_main, topk_extra,
        num_blocks_main, num_blocks_extra,
        block_size_main, block_size_extra,
        s_q,
        stride_kv_block_main,
        stride_idx_t_main, stride_idx_k_main,
        stride_kv_block_extra,
        stride_idx_t_extra, stride_idx_k_extra,
        stride_out_t, stride_out_k, stride_out_d,
        stride_mask_t, stride_mask_k,
        num_main_pids,
        BLOCK_TK=BLOCK_TK,
        D_NOPE=MODEL1_D_NOPE, D_ROPE=MODEL1_D_ROPE,
        BYTES_PER_TOKEN_DATA=MODEL1_BYTES_PER_TOKEN_DATA,
        BYTES_PER_TOKEN_SCALE=MODEL1_BYTES_PER_TOKEN_SCALE,
        TILE_SIZE=MODEL1_TILE_SIZE,
        HAS_TOPK_LENGTH_MAIN=has_topk_length_main,
        HAS_TOPK_LENGTH_EXTRA=has_topk_length_extra,
        num_warps=8, num_stages=2,
    )
    return True


def fused_gather_dequant_fp8_model1(
    kv_cache_main, indices_main, block_size_main, topk_length_main,
    kv_cache_extra, indices_extra, block_size_extra, topk_length_extra,
    output_kv, output_mask, s_q=1,
):
    """Fused MODEL1 gather - uses 1D fused kernel for small workloads, two kernels for large."""
    has_topk_length_main = topk_length_main is not None
    has_topk_length_extra = topk_length_extra is not None

    total_tokens, topk_main = indices_main.shape
    topk_extra = indices_extra.shape[1]
    total_elements = total_tokens * (topk_main + topk_extra)

    # Use fused 2D grid kernel only for small workloads where kernel launch overhead matters
    # For large workloads, the two-kernel approach is more efficient
    USE_FUSED_THRESHOLD = 150000  # ~150K elements - balanced for 1D grid performance

    # Now we can use fused kernel even when topk_length settings differ
    # because we fixed the type mismatch issue by always using int32 tensors
    use_fused = total_elements < USE_FUSED_THRESHOLD

    if use_fused:
        return truly_fused_gather_dequant_fp8_model1(
            kv_cache_main, indices_main, block_size_main, topk_length_main,
            kv_cache_extra, indices_extra, block_size_extra, topk_length_extra,
            output_kv, output_mask, s_q,
        )

    # Use original two-kernel approach for large workloads
    kv_flat_main, num_blocks_main, stride_kv_block_main = _prepare_kv_cache_flat(kv_cache_main)
    kv_flat_extra, num_blocks_extra, stride_kv_block_extra = _prepare_kv_cache_flat(kv_cache_extra)

    topk_length_main_tensor = topk_length_main if has_topk_length_main else output_mask[:1, 0]
    topk_length_extra_tensor = topk_length_extra if has_topk_length_extra else output_mask[:1, 0]

    stride_idx_t_main, stride_idx_k_main = indices_main.stride(0), indices_main.stride(1)
    stride_idx_t_extra, stride_idx_k_extra = indices_extra.stride(0), indices_extra.stride(1)
    stride_out_t, stride_out_k, stride_out_d = output_kv.stride(0), output_kv.stride(1), output_kv.stride(2)
    stride_mask_t, stride_mask_k = output_mask.stride(0), output_mask.stride(1)

    total_elements_main = total_tokens * topk_main
    total_elements_extra = total_tokens * topk_extra

    USE_FIXED_KERNEL_THRESHOLD = 32768

    if total_elements_main < USE_FIXED_KERNEL_THRESHOLD:
        grid_main = (triton.cdiv(total_elements_main, 128),)
        _gather_dequant_model1_kernel_fixed_128[grid_main](
            kv_flat_main, indices_main, topk_length_main_tensor,
            output_kv, output_mask,
            total_tokens, topk_main, num_blocks_main, block_size_main,
            0, s_q, stride_kv_block_main,
            stride_idx_t_main, stride_idx_k_main,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=MODEL1_D_NOPE, D_ROPE=MODEL1_D_ROPE,
            BYTES_PER_TOKEN_DATA=MODEL1_BYTES_PER_TOKEN_DATA,
            BYTES_PER_TOKEN_SCALE=MODEL1_BYTES_PER_TOKEN_SCALE,
            TILE_SIZE=MODEL1_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_main,
            num_warps=8, num_stages=2,
        )
    else:
        workload_main = _get_workload_size_category(total_tokens, topk_main)
        grid_main = lambda meta: (triton.cdiv(total_elements_main, meta['BLOCK_TK']),)
        _gather_dequant_model1_kernel[grid_main](
            kv_flat_main, indices_main, topk_length_main_tensor,
            output_kv, output_mask,
            total_tokens, topk_main, num_blocks_main, block_size_main,
            workload_main, 0, s_q, stride_kv_block_main,
            stride_idx_t_main, stride_idx_k_main,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=MODEL1_D_NOPE, D_ROPE=MODEL1_D_ROPE,
            BYTES_PER_TOKEN_DATA=MODEL1_BYTES_PER_TOKEN_DATA,
            BYTES_PER_TOKEN_SCALE=MODEL1_BYTES_PER_TOKEN_SCALE,
            TILE_SIZE=MODEL1_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_main,
        )

    if total_elements_extra < USE_FIXED_KERNEL_THRESHOLD:
        grid_extra = (triton.cdiv(total_elements_extra, 128),)
        _gather_dequant_model1_kernel_fixed_128[grid_extra](
            kv_flat_extra, indices_extra, topk_length_extra_tensor,
            output_kv, output_mask,
            total_tokens, topk_extra, num_blocks_extra, block_size_extra,
            topk_main, s_q, stride_kv_block_extra,
            stride_idx_t_extra, stride_idx_k_extra,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=MODEL1_D_NOPE, D_ROPE=MODEL1_D_ROPE,
            BYTES_PER_TOKEN_DATA=MODEL1_BYTES_PER_TOKEN_DATA,
            BYTES_PER_TOKEN_SCALE=MODEL1_BYTES_PER_TOKEN_SCALE,
            TILE_SIZE=MODEL1_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_extra,
            num_warps=8, num_stages=2,
        )
    else:
        workload_extra = _get_workload_size_category(total_tokens, topk_extra)
        grid_extra = lambda meta: (triton.cdiv(total_elements_extra, meta['BLOCK_TK']),)
        _gather_dequant_model1_kernel[grid_extra](
            kv_flat_extra, indices_extra, topk_length_extra_tensor,
            output_kv, output_mask,
            total_tokens, topk_extra, num_blocks_extra, block_size_extra,
            workload_extra, topk_main, s_q, stride_kv_block_extra,
            stride_idx_t_extra, stride_idx_k_extra,
            stride_out_t, stride_out_k, stride_out_d,
            stride_mask_t, stride_mask_k,
            D_NOPE=MODEL1_D_NOPE, D_ROPE=MODEL1_D_ROPE,
            BYTES_PER_TOKEN_DATA=MODEL1_BYTES_PER_TOKEN_DATA,
            BYTES_PER_TOKEN_SCALE=MODEL1_BYTES_PER_TOKEN_SCALE,
            TILE_SIZE=MODEL1_TILE_SIZE,
            HAS_TOPK_LENGTH=has_topk_length_extra,
        )
    return True

def triton_sparse_attn_decode_model1(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention decode for MODEL1 (d_qk=512)."""
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == MODEL1_D_QK, f"Expected d_qk={MODEL1_D_QK} for MODEL1, got {d_qk}"
    total_tokens = b * s_q

    topk_main = kv_scope.indices_in_kvcache.size(-1)
    topk_extra = extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0
    total_topk = topk_main + topk_extra

    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk)

    if len(token_ranges) == 1:
        return _triton_sparse_attn_decode_model1_impl(
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

        chunk_out, chunk_lse = _triton_sparse_attn_decode_model1_impl(
            q_input, chunk_kv_scope, chunk_extra_kv_scope, sm_scale, d_v, attn_sink
        )

        outputs.append(chunk_out.reshape(chunk_tokens, h_q, d_v))
        lses.append(chunk_lse.reshape(chunk_tokens, h_q))

    output = torch.cat(outputs, dim=0).reshape(b, s_q, h_q, d_v)
    lse = torch.cat(lses, dim=0).reshape(b, s_q, h_q).transpose(1, 2)

    return output, lse


def _triton_sparse_attn_decode_model1_impl(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal implementation of sparse attention decode for MODEL1."""
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

    use_fused = (kv_scope.blocked_k_quantized is not None and
                 extra_kv_scope is not None and
                 extra_kv_scope.blocked_k_quantized is not None)

    if use_fused:
        block_size_extra = extra_kv_scope.blocked_k.shape[1]
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)
        fused_gather_dequant_fp8_model1(
            kv_scope.blocked_k_quantized, indices_main, block_size_main, kv_scope.topk_length,
            extra_kv_scope.blocked_k_quantized, indices_extra, block_size_extra, extra_kv_scope.topk_length,
            gathered_kv, invalid_mask, s_q)
    else:
        if kv_scope.blocked_k_quantized is not None:
            gather_dequant_fp8_model1(
                kv_scope.blocked_k_quantized, indices_main, block_size_main,
                gathered_kv, invalid_mask, 0, kv_scope.topk_length, s_q)
        else:
            scope_invalid_mask = indices_main == -1
            if kv_scope.topk_length is not None:
                topk_length_mask = (
                    torch.arange(0, topk_main, device=q.device).view(1, 1, topk_main).broadcast_to(b, s_q, topk_main)
                    >= kv_scope.topk_length.view(b, 1, 1)
                ).reshape(total_tokens, topk_main)
                scope_invalid_mask = scope_invalid_mask | topk_length_mask
            invalid_mask[:, :topk_main] = scope_invalid_mask
            indices_clamped = torch.clamp(indices_main, min=0)
            kv_gathered = kv_scope.blocked_k.view(-1, d_qk).index_select(0, indices_clamped.view(-1)).view(total_tokens, topk_main, d_qk).to(torch.bfloat16)
            kv_gathered[scope_invalid_mask] = 0
            gathered_kv[:, :topk_main, :] = kv_gathered

        if extra_kv_scope is not None:
            block_size_extra = extra_kv_scope.blocked_k.shape[1]
            indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)

            if extra_kv_scope.blocked_k_quantized is not None:
                gather_dequant_fp8_model1(
                    extra_kv_scope.blocked_k_quantized, indices_extra, block_size_extra,
                    gathered_kv, invalid_mask, topk_main, extra_kv_scope.topk_length, s_q)
            else:
                scope_invalid_mask = indices_extra == -1
                if extra_kv_scope.topk_length is not None:
                    topk_length_mask = (
                        torch.arange(0, topk_extra, device=q.device).view(1, 1, topk_extra).broadcast_to(b, s_q, topk_extra)
                        >= extra_kv_scope.topk_length.view(b, 1, 1)
                    ).reshape(total_tokens, topk_extra)
                    scope_invalid_mask = scope_invalid_mask | topk_length_mask
                invalid_mask[:, topk_main:] = scope_invalid_mask
                indices_clamped = torch.clamp(indices_extra, min=0)
                kv_gathered = extra_kv_scope.blocked_k.view(-1, d_qk).index_select(0, indices_clamped.view(-1)).view(total_tokens, topk_extra, d_qk).to(torch.bfloat16)
                kv_gathered[scope_invalid_mask] = 0
                gathered_kv[:, topk_main:, :] = kv_gathered

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
