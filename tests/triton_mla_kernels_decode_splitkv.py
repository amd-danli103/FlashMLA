"""
Split-KV Triton MLA Decode Kernels - Optimized Version 3 with Autotune

Key optimizations:
1. Autotune support for BLOCK_H, num_warps
2. Two-phase approach: partial kernel + combine kernel
3. Better memory access patterns

Supports:
- MODEL1 (d_qk=512)
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

# ============================================================================
# Constants
# ============================================================================
MODEL1_D_QK = 512
MODEL1_D_V = 512
MODEL1_D_NOPE = 448
MODEL1_D_ROPE = 64
MODEL1_TILE_SIZE = 64
MODEL1_BYTES_PER_TOKEN_DATA = 576
MODEL1_BYTES_PER_TOKEN_SCALE = 8

# Default split size
SPLIT_K_SIZE = 128


# Autotune disabled for debugging
@triton.jit
def _splitkv_partial_kernel_v3(
    Q, KV_Cache, Indices, TopkLength,
    PartialO, PartialLSE,
    sm_scale, total_tokens, h_q, topk, num_blocks, block_size, s_q,
    num_splits,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kv_block,
    stride_idx_t, stride_idx_k,
    stride_po_split, stride_po_t, stride_po_h, stride_po_d,
    stride_plse_split, stride_plse_t, stride_plse_h,
    HAS_TOPK_LENGTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """Split-KV partial kernel - computes partial O and LSE for each split."""
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8
    NEG_INF = float("-inf")

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_split = tl.program_id(2)
    pid_t_64 = pid_t.to(tl.int64)

    kv_start = pid_split * BLOCK_N
    kv_end = tl.minimum(kv_start + BLOCK_N, topk)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64

    batch_idx = pid_t // s_q
    offs_tile = tl.arange(0, TILE_SIZE)

    q_0 = tl.load(q_base + offs_h[:, None] * stride_q_h + offs_tile[None, :] * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_1 = tl.load(q_base + offs_h[:, None] * stride_q_h + (TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_2 = tl.load(q_base + offs_h[:, None] * stride_q_h + (2*TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_3 = tl.load(q_base + offs_h[:, None] * stride_q_h + (3*TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_4 = tl.load(q_base + offs_h[:, None] * stride_q_h + (4*TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_5 = tl.load(q_base + offs_h[:, None] * stride_q_h + (5*TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_6 = tl.load(q_base + offs_h[:, None] * stride_q_h + (6*TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_7 = tl.load(q_base + offs_h[:, None] * stride_q_h + (7*TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)

    has_work = kv_start < topk

    if has_work:
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_end

        idx_ptrs = Indices + pid_t * stride_idx_t + offs_n * stride_idx_k
        indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

        is_invalid = indices == -1
        if HAS_TOPK_LENGTH:
            topk_len = tl.load(TopkLength + batch_idx)
            is_invalid = is_invalid | (offs_n >= topk_len)

        valid = mask_n & ~is_invalid
        indices_clamped = tl.maximum(indices, 0)

        block_idx = indices_clamped // block_size
        offset_in_block = indices_clamped % block_size

        # Clamp block_idx to valid range to prevent out-of-bounds access
        block_idx_clamped = tl.minimum(block_idx, num_blocks - 1)
        block_idx_64 = block_idx_clamped.to(tl.int64)
        offset_in_block_64 = offset_in_block.to(tl.int64)

        # Cast stride to int64 to prevent overflow for large block indices
        stride_kv_block_64 = tl.cast(stride_kv_block, tl.int64)
        kv_block_base = KV_Cache + block_idx_64 * stride_kv_block_64

        # Use int64 for all offset calculations
        BYTES_PER_TOKEN_DATA_64: tl.constexpr = 576
        BYTES_PER_TOKEN_SCALE_64: tl.constexpr = 8
        block_size_64 = tl.cast(block_size, tl.int64)
        nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA_64
        scale_base_offset = block_size_64 * BYTES_PER_TOKEN_DATA_64 + offset_in_block_64 * BYTES_PER_TOKEN_SCALE_64

        valid_2d = valid[:, None]

        scale_ptrs = kv_block_base + scale_base_offset
        scale_uint8_0 = tl.load(scale_ptrs, mask=valid, other=127).to(tl.uint8)
        scale_uint8_1 = tl.load(scale_ptrs + 1, mask=valid, other=127).to(tl.uint8)
        scale_uint8_2 = tl.load(scale_ptrs + 2, mask=valid, other=127).to(tl.uint8)
        scale_uint8_3 = tl.load(scale_ptrs + 3, mask=valid, other=127).to(tl.uint8)
        scale_uint8_4 = tl.load(scale_ptrs + 4, mask=valid, other=127).to(tl.uint8)
        scale_uint8_5 = tl.load(scale_ptrs + 5, mask=valid, other=127).to(tl.uint8)
        scale_uint8_6 = tl.load(scale_ptrs + 6, mask=valid, other=127).to(tl.uint8)

        scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16)

        tile_base = kv_block_base[:, None] + nope_rope_offset[:, None]

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        nope_ptrs = tile_base + offs_tile[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        kv_0 = (nope_fp8.to(tl.bfloat16) * scale_bf16_0[:, None]).to(tl.bfloat16)
        kv_0 = tl.where(valid_2d, kv_0, 0.0)
        qk += tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)

        nope_ptrs = tile_base + TILE_SIZE + offs_tile[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        kv_1 = (nope_fp8.to(tl.bfloat16) * scale_bf16_1[:, None]).to(tl.bfloat16)
        kv_1 = tl.where(valid_2d, kv_1, 0.0)
        qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)

        nope_ptrs = tile_base + 2*TILE_SIZE + offs_tile[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        kv_2 = (nope_fp8.to(tl.bfloat16) * scale_bf16_2[:, None]).to(tl.bfloat16)
        kv_2 = tl.where(valid_2d, kv_2, 0.0)
        qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)

        nope_ptrs = tile_base + 3*TILE_SIZE + offs_tile[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        kv_3 = (nope_fp8.to(tl.bfloat16) * scale_bf16_3[:, None]).to(tl.bfloat16)
        kv_3 = tl.where(valid_2d, kv_3, 0.0)
        qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)

        nope_ptrs = tile_base + 4*TILE_SIZE + offs_tile[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        kv_4 = (nope_fp8.to(tl.bfloat16) * scale_bf16_4[:, None]).to(tl.bfloat16)
        kv_4 = tl.where(valid_2d, kv_4, 0.0)
        qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)

        nope_ptrs = tile_base + 5*TILE_SIZE + offs_tile[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        kv_5 = (nope_fp8.to(tl.bfloat16) * scale_bf16_5[:, None]).to(tl.bfloat16)
        kv_5 = tl.where(valid_2d, kv_5, 0.0)
        qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)

        nope_ptrs = tile_base + 6*TILE_SIZE + offs_tile[None, :]
        nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
        nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
        kv_6 = (nope_fp8.to(tl.bfloat16) * scale_bf16_6[:, None]).to(tl.bfloat16)
        kv_6 = tl.where(valid_2d, kv_6, 0.0)
        qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)

        rope_ptrs = tile_base + D_NOPE + offs_tile[None, :] * 2
        rope_lo = tl.load(rope_ptrs, mask=valid_2d, other=0).to(tl.uint16)
        rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
        kv_7 = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
        kv_7 = tl.where(valid_2d, kv_7, 0.0)
        qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)

        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, NEG_INF)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
        l_new = alpha * l_i + tl.sum(p, axis=1)
        p_bf16 = p.to(tl.bfloat16)

        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
        acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
        acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
        acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
        acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)

        m_i = m_new
        l_i = l_new

    partial_lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    partial_lse = tl.where(l_i == 0.0, NEG_INF, partial_lse)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc_0 = acc_0 / l_safe[:, None]
    acc_1 = acc_1 / l_safe[:, None]
    acc_2 = acc_2 / l_safe[:, None]
    acc_3 = acc_3 / l_safe[:, None]
    acc_4 = acc_4 / l_safe[:, None]
    acc_5 = acc_5 / l_safe[:, None]
    acc_6 = acc_6 / l_safe[:, None]
    acc_7 = acc_7 / l_safe[:, None]

    stride_po_split_64 = tl.cast(stride_po_split, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
    po_base = PartialO + pid_split * stride_po_split_64 + pid_t_64 * stride_po_t_64

    partial_row_ptrs = po_base + offs_h[:, None] * stride_po_h
    tl.store(partial_row_ptrs + offs_tile[None, :] * stride_po_d, acc_0, mask=mask_h[:, None])
    tl.store(partial_row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_1, mask=mask_h[:, None])
    tl.store(partial_row_ptrs + (2*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_2, mask=mask_h[:, None])
    tl.store(partial_row_ptrs + (3*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_3, mask=mask_h[:, None])
    tl.store(partial_row_ptrs + (4*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_4, mask=mask_h[:, None])
    tl.store(partial_row_ptrs + (5*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_5, mask=mask_h[:, None])
    tl.store(partial_row_ptrs + (6*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_6, mask=mask_h[:, None])
    tl.store(partial_row_ptrs + (7*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_7, mask=mask_h[:, None])

    stride_plse_split_64 = tl.cast(stride_plse_split, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)
    plse_base = PartialLSE + pid_split * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
    plse_ptrs = plse_base + offs_h * stride_plse_h
    tl.store(plse_ptrs, partial_lse, mask=mask_h)


# Autotune disabled for debugging
@triton.jit
def _combine_kernel_v3(
    PartialO, PartialLSE, AttnSink,
    Output, LSE,
    total_tokens, h_q, d_v, num_splits,
    stride_po_split, stride_po_t, stride_po_h, stride_po_d,
    stride_plse_split, stride_plse_t, stride_plse_h,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    """Combine partial O and LSE from all splits."""
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    POS_INF = float("+inf")

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    stride_plse_split_64 = tl.cast(stride_plse_split, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)

    global_max_lse = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    for split_idx in range(MAX_SPLITS):
        if split_idx < num_splits:
            plse_base = PartialLSE + split_idx * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
            plse_ptrs = plse_base + offs_h * stride_plse_h
            partial_lse = tl.load(plse_ptrs, mask=mask_h, other=NEG_INF)
            global_max_lse = tl.maximum(global_max_lse, partial_lse)

    all_invalid = (global_max_lse == NEG_INF)
    global_max_lse = tl.where(all_invalid, 0.0, global_max_lse)

    sum_exp = tl.zeros([BLOCK_H], dtype=tl.float32)
    for split_idx in range(MAX_SPLITS):
        if split_idx < num_splits:
            plse_base = PartialLSE + split_idx * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
            plse_ptrs = plse_base + offs_h * stride_plse_h
            partial_lse = tl.load(plse_ptrs, mask=mask_h, other=NEG_INF)
            exp_diff = tl.where(partial_lse == NEG_INF, 0.0, tl.math.exp2((partial_lse - global_max_lse) * LOG2E))
            sum_exp += exp_diff

    global_lse = global_max_lse + tl.math.log2(tl.where(sum_exp == 0.0, 1.0, sum_exp)) / LOG2E
    is_lonely_q = all_invalid

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    offs_d = tl.arange(0, BLOCK_D)

    stride_po_split_64 = tl.cast(stride_po_split, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)

    for split_idx in range(MAX_SPLITS):
        if split_idx < num_splits:
            plse_base = PartialLSE + split_idx * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
            plse_ptrs = plse_base + offs_h * stride_plse_h
            partial_lse = tl.load(plse_ptrs, mask=mask_h, other=NEG_INF)

            weight = tl.where(partial_lse == NEG_INF, 0.0, tl.math.exp2((partial_lse - global_lse) * LOG2E))

            po_base = PartialO + split_idx * stride_po_split_64 + pid_t_64 * stride_po_t_64
            po_row_ptrs = po_base + offs_h[:, None] * stride_po_h

            po_0 = tl.load(po_row_ptrs + offs_d[None, :] * stride_po_d, mask=mask_h[:, None], other=0.0)
            po_1 = tl.load(po_row_ptrs + (BLOCK_D + offs_d[None, :]) * stride_po_d, mask=mask_h[:, None], other=0.0)
            po_2 = tl.load(po_row_ptrs + (2*BLOCK_D + offs_d[None, :]) * stride_po_d, mask=mask_h[:, None], other=0.0)
            po_3 = tl.load(po_row_ptrs + (3*BLOCK_D + offs_d[None, :]) * stride_po_d, mask=mask_h[:, None], other=0.0)

            acc_0 += weight[:, None] * po_0
            acc_1 += weight[:, None] * po_1
            acc_2 += weight[:, None] * po_2
            acc_3 += weight[:, None] * po_3

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_lse = tl.math.exp2((attn_sink_vals - global_lse) * LOG2E)
        denominator = 1.0 + exp_attn_sink_minus_lse
        denominator = tl.where(is_lonely_q, 1.0, denominator)
        output_scale = 1.0 / denominator
        acc_0 = acc_0 * output_scale[:, None]
        acc_1 = acc_1 * output_scale[:, None]
        acc_2 = acc_2 * output_scale[:, None]
        acc_3 = acc_3 * output_scale[:, None]

    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0)
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1)
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2)
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3)
    final_lse = tl.where(is_lonely_q, POS_INF, global_lse)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64
    out_row_ptrs = o_base + offs_h[:, None] * stride_o_h

    tl.store(out_row_ptrs + offs_d[None, :] * stride_o_d, acc_0.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(out_row_ptrs + (BLOCK_D + offs_d[None, :]) * stride_o_d, acc_1.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(out_row_ptrs + (2*BLOCK_D + offs_d[None, :]) * stride_o_d, acc_2.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(out_row_ptrs + (3*BLOCK_D + offs_d[None, :]) * stride_o_d, acc_3.to(tl.bfloat16), mask=mask_h[:, None])

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    lse_ptrs = LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h
    tl.store(lse_ptrs, final_lse, mask=mask_h)


def splitkv_sparse_attn_decode_model1(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    sm_scale: float,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split-KV sparse attention decode for MODEL1.

    Uses a chunked two-phase approach to work around ROCm memory issues:
    1. Process topk in chunks to avoid large intermediate tensors
    2. For each chunk: gather KV data, then compute attention
    3. Combine results using online softmax

    This approach avoids the ROCm-specific issue where combining
    Q loading with indirect KV access in a single kernel causes crashes.
    """
    from triton_mla_kernels_decode_model1 import gather_dequant_fp8_model1
    from triton_mla_kernels_decode_common import run_unified_attention

    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]
    d_v = MODEL1_D_V
    device = q.device

    # Chunk size to keep intermediate tensors manageable
    # Each chunk processes up to MAX_CHUNK_TOPK tokens
    MAX_CHUNK_TOPK = 4096  # ~1GB for gathered_kv per chunk

    if topk <= MAX_CHUNK_TOPK:
        # Small topk: single pass
#        print(f"[DEBUG SplitKV] single pass: total_tokens={total_tokens}, h_q={h_q}, topk={topk}")

        # Use zeros to avoid ROCm memory issues
        gathered_kv = torch.zeros(total_tokens, topk, d_qk, dtype=torch.bfloat16, device=device)
        invalid_mask = torch.zeros(total_tokens, topk, dtype=torch.bool, device=device)

        gather_dequant_fp8_model1(
            kv_cache, indices, block_size,
            gathered_kv, invalid_mask, 0, topk_length, s_q
        )

        q_reshaped = q.to(torch.bfloat16)
        if not q_reshaped.is_contiguous():
            q_reshaped = q_reshaped.contiguous()

        output, lse = run_unified_attention(
            q_reshaped, gathered_kv, invalid_mask,
            d_v, sm_scale, total_tokens, h_q, topk, d_qk,
            attn_sink=attn_sink
        )

        return output, lse

    # Large topk: chunked processing with online softmax
#    print(f"[DEBUG SplitKV] chunked: total_tokens={total_tokens}, h_q={h_q}, topk={topk}, chunk_size={MAX_CHUNK_TOPK}")

    num_chunks = (topk + MAX_CHUNK_TOPK - 1) // MAX_CHUNK_TOPK

    # Initialize accumulators for online softmax
    acc_output = torch.zeros(total_tokens, h_q, d_v, dtype=torch.float32, device=device)
    acc_lse = torch.full((total_tokens, h_q), float('-inf'), dtype=torch.float32, device=device)

    q_reshaped = q.to(torch.bfloat16)
    if not q_reshaped.is_contiguous():
        q_reshaped = q_reshaped.contiguous()

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * MAX_CHUNK_TOPK
        chunk_end = min(chunk_start + MAX_CHUNK_TOPK, topk)
        chunk_topk = chunk_end - chunk_start

        # Slice indices for this chunk
        chunk_indices = indices[:, chunk_start:chunk_end].contiguous()

        # Adjust topk_length for this chunk
        chunk_topk_length = None
        if topk_length is not None:
            chunk_topk_length = torch.clamp(topk_length - chunk_start, 0, chunk_topk)

        # Gather KV for this chunk (use zeros to avoid ROCm issues)
        gathered_kv = torch.zeros(total_tokens, chunk_topk, d_qk, dtype=torch.bfloat16, device=device)
        invalid_mask = torch.zeros(total_tokens, chunk_topk, dtype=torch.bool, device=device)

        gather_dequant_fp8_model1(
            kv_cache, chunk_indices, block_size,
            gathered_kv, invalid_mask, 0, chunk_topk_length, s_q
        )

        # Compute attention for this chunk (no attn_sink for intermediate chunks)
        chunk_output, chunk_lse = run_unified_attention(
            q_reshaped, gathered_kv, invalid_mask,
            d_v, sm_scale, total_tokens, h_q, chunk_topk, d_qk,
            attn_sink=None
        )

        # Online softmax combination with proper handling of inf values
        # new_lse = log(exp(acc_lse) + exp(chunk_lse))
        # new_output = (exp(acc_lse - new_lse) * acc_output + exp(chunk_lse - new_lse) * chunk_output)

        # Handle edge cases:
        # 1. +inf in LSE means "lonely query" (no valid tokens) - treat as contributing nothing
        # 2. -inf in LSE means uninitialized accumulator - treat as contributing nothing
        # 3. Both invalid: result should be +inf (lonely) with zero output
        # 4. One invalid: result should be the valid value with its output
        # 5. Both finite: standard online softmax

        # Detect invalid LSE values (both +inf and -inf mean no contribution)
        acc_is_invalid = torch.isinf(acc_lse)  # Both +inf and -inf
        chunk_is_invalid = torch.isinf(chunk_lse)  # Both +inf and -inf
        both_invalid = acc_is_invalid & chunk_is_invalid

        # For numerical stability, replace inf with a very small/large finite value temporarily
        # This avoids inf - inf = nan
        SAFE_REPLACEMENT = -1e30  # Use a very negative value for invalid entries
        acc_lse_safe = torch.where(acc_is_invalid, torch.full_like(acc_lse, SAFE_REPLACEMENT), acc_lse)
        chunk_lse_safe = torch.where(chunk_is_invalid, torch.full_like(chunk_lse, SAFE_REPLACEMENT), chunk_lse)

        max_lse = torch.maximum(acc_lse_safe, chunk_lse_safe)
        exp_acc = torch.exp(acc_lse_safe - max_lse)
        exp_chunk = torch.exp(chunk_lse_safe - max_lse)

        # When original value was invalid (inf), the weight should be 0
        exp_acc = torch.where(acc_is_invalid, torch.zeros_like(exp_acc), exp_acc)
        exp_chunk = torch.where(chunk_is_invalid, torch.zeros_like(exp_chunk), exp_chunk)

        sum_exp = exp_acc + exp_chunk
        # Avoid division by zero when both are invalid
        sum_exp_safe = torch.where(both_invalid, torch.ones_like(sum_exp), sum_exp)

        # Update output
        new_output = (exp_acc.unsqueeze(-1) * acc_output +
                      exp_chunk.unsqueeze(-1) * chunk_output.float()) / sum_exp_safe.unsqueeze(-1)
        # When both are invalid, output should be zero
        acc_output = torch.where(both_invalid.unsqueeze(-1), torch.zeros_like(new_output), new_output)

        # Update lse
        new_lse = max_lse + torch.log(sum_exp_safe)
        # When both are invalid, lse should be +inf (lonely query convention)
        acc_lse = torch.where(both_invalid, torch.full_like(new_lse, float('+inf')), new_lse)

        # Clean up chunk tensors
        del gathered_kv, invalid_mask, chunk_output, chunk_lse

    # Apply attn_sink at the end if needed
    # attn_sink is a per-head value in log-space that adds to the softmax denominator
    # It represents the contribution of "sink" tokens to the attention normalization
    #
    # In the kernel, attn_sink is used as:
    #   exp_attn_sink_minus_m = exp((attn_sink - m_i) * LOG2E)
    #   denominator = l_i + exp_attn_sink_minus_m
    #   output_scale = 1.0 / denominator
    #   output = acc * output_scale
    #   lse = m_i + log(l_i)  # LSE does NOT include attn_sink
    #
    # So we need to scale the output but NOT modify the LSE
    if attn_sink is not None:
        # attn_sink shape: [h_q], acc_lse shape: [total_tokens, h_q]
        # acc_lse = m_i + log(l_i) where l_i is the sum of exp(scores - m_i)
        # We need to compute: output_scale = l_i / (l_i + exp(attn_sink - m_i))
        #                                  = 1 / (1 + exp(attn_sink - m_i) / l_i)
        #                                  = 1 / (1 + exp(attn_sink - m_i - log(l_i)))
        #                                  = 1 / (1 + exp(attn_sink - lse))

        # Broadcast attn_sink to match acc_lse shape
        sink_vals = attn_sink.unsqueeze(0).expand_as(acc_lse)  # [total_tokens, h_q]

        # Detect lonely queries (acc_lse is +inf means no valid tokens)
        is_lonely = torch.isinf(acc_lse) & (acc_lse > 0)

        # For lonely queries, output is already 0 and should stay 0
        # For non-lonely queries, compute the scale

        # Replace +inf in acc_lse with a large finite value to avoid inf - inf = nan
        acc_lse_safe = torch.where(is_lonely, torch.full_like(acc_lse, 1e30), acc_lse)

        # Compute exp(attn_sink - lse) with numerical stability
        diff = sink_vals - acc_lse_safe

        # Clamp to avoid overflow
        diff_clamped = torch.clamp(diff, min=-100, max=100)
        exp_diff = torch.exp(diff_clamped)

        # Handle special cases for sink_vals:
        # - If sink_vals is +inf, exp_diff should be inf -> scale = 0
        # - If sink_vals is -inf, exp_diff should be 0 -> scale = 1
        sink_is_pos_inf = torch.isinf(sink_vals) & (sink_vals > 0)
        sink_is_neg_inf = torch.isinf(sink_vals) & (sink_vals < 0)

        exp_diff = torch.where(sink_is_pos_inf, torch.full_like(exp_diff, float('inf')), exp_diff)
        exp_diff = torch.where(sink_is_neg_inf, torch.zeros_like(exp_diff), exp_diff)

        # scale = 1 / (1 + exp_diff)
        denominator = 1.0 + exp_diff
        scale = 1.0 / denominator

        # Handle inf case: if exp_diff is inf, scale should be 0
        scale = torch.where(torch.isinf(exp_diff), torch.zeros_like(scale), scale)

        # For lonely queries, scale doesn't matter since output is already 0
        # But set scale to 1 to avoid any potential issues
        scale = torch.where(is_lonely, torch.ones_like(scale), scale)

        # Scale the output
        acc_output = acc_output * scale.unsqueeze(-1)

        # LSE is NOT modified - it stays as m_i + log(l_i)

    return acc_output.to(torch.bfloat16), acc_lse


def _splitkv_single_pass(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    sm_scale: float,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-pass SplitKV for small topk."""
    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]
    d_v = MODEL1_D_V
    device = q.device

    kv_uint8 = kv_cache.view(torch.uint8)
    num_blocks = kv_cache.shape[0]
    bytes_per_block = kv_uint8.shape[1] * kv_uint8.shape[2] * kv_uint8.shape[3]
    kv_flat = kv_uint8.reshape(num_blocks, bytes_per_block)
    stride_kv_block = kv_uint8.stride(0)
#    print(f"[DEBUG SplitKV] kv_uint8 shape={kv_uint8.shape}, strides={kv_uint8.stride()}")
#    print(f"[DEBUG SplitKV] kv_flat shape={kv_flat.shape}, strides={kv_flat.stride()}")
#    print(f"[DEBUG SplitKV] stride_kv_block={stride_kv_block}, bytes_per_block={bytes_per_block}")

    num_splits = (topk + SPLIT_K_SIZE - 1) // SPLIT_K_SIZE

#    print(f"[DEBUG SplitKV single-pass] topk={topk}, h_q={h_q}, num_splits={num_splits}")
#    print(f"[DEBUG SplitKV single-pass] indices shape={indices.shape}, kv_cache shape={kv_cache.shape}")
#    print(f"[DEBUG SplitKV single-pass] indices min={indices.min().item()}, max={indices.max().item()}")
#    print(f"[DEBUG SplitKV single-pass] num_blocks={num_blocks}, block_size={block_size}")

    # Preprocess indices: replace -1 with 0 and create a mask
    # This ensures all pointer calculations are valid
    invalid_mask = (indices == -1)
    indices_safe = indices.clone()
    indices_safe[invalid_mask] = 0  # Replace -1 with 0 for safe pointer calculation

    # Validate indices are within bounds
    max_valid_index = num_blocks * block_size - 1
    indices_max = indices_safe.max().item()
    if indices_max > max_valid_index:
        print(f"[WARNING] indices max ({indices_max}) > max_valid_index ({max_valid_index})")
        # Clamp indices to valid range
        indices_safe = torch.clamp(indices_safe, 0, max_valid_index)
#        print(f"[DEBUG] Clamped indices to valid range")

    # Use safe indices for kernel
    indices = indices_safe
    partial_o = torch.empty(num_splits, total_tokens, h_q, d_v, dtype=torch.float32, device=device)
    partial_lse = torch.empty(num_splits, total_tokens, h_q, dtype=torch.float32, device=device)

    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    if not indices.is_contiguous():
        indices = indices.contiguous()

    stride_q_t, stride_q_h, stride_q_d = q.stride()
    stride_idx_t, stride_idx_k = indices.stride()
    stride_po_split, stride_po_t, stride_po_h, stride_po_d = partial_o.stride()
    stride_plse_split, stride_plse_t, stride_plse_h = partial_lse.stride()
    stride_o_t, stride_o_h, stride_o_d = output.stride()
    stride_lse_t, stride_lse_h = lse.stride()

    HAS_TOPK_LENGTH = topk_length is not None
    HAS_ATTN_SINK = attn_sink is not None

    topk_length_tensor = topk_length if HAS_TOPK_LENGTH else lse[:1]
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1]

    if num_splits <= 4:
        MAX_SPLITS = 4
    elif num_splits <= 8:
        MAX_SPLITS = 8
    elif num_splits <= 16:
        MAX_SPLITS = 16
    elif num_splits <= 32:
        MAX_SPLITS = 32
    elif num_splits <= 64:
        MAX_SPLITS = 64
    elif num_splits <= 128:
        MAX_SPLITS = 128
    else:
        MAX_SPLITS = 256

    # Fixed BLOCK_H since autotune is disabled
    BLOCK_H_PARTIAL = 16
    n_h_blocks = (h_q + BLOCK_H_PARTIAL - 1) // BLOCK_H_PARTIAL
    grid_partial = (total_tokens, n_h_blocks, num_splits)

#    print(f"[DEBUG] Launching partial kernel: grid={grid_partial}")
    _splitkv_partial_kernel_v3[grid_partial](
        q, kv_flat, indices, topk_length_tensor,
        partial_o, partial_lse,
        sm_scale, total_tokens, h_q, topk, num_blocks, block_size, s_q,
        num_splits,
        stride_q_t, stride_q_h, stride_q_d,
        stride_kv_block,
        stride_idx_t, stride_idx_k,
        stride_po_split, stride_po_t, stride_po_h, stride_po_d,
        stride_plse_split, stride_plse_t, stride_plse_h,
        HAS_TOPK_LENGTH=HAS_TOPK_LENGTH,
        BLOCK_H=BLOCK_H_PARTIAL,
        BLOCK_N=SPLIT_K_SIZE,
        TILE_SIZE=MODEL1_TILE_SIZE,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()
#    print(f"[DEBUG] Partial kernel completed successfully")

    # Fixed BLOCK_H for combine kernel
    BLOCK_H_COMBINE = 16
    n_h_blocks_combine = (h_q + BLOCK_H_COMBINE - 1) // BLOCK_H_COMBINE
    grid_combine = (total_tokens, n_h_blocks_combine)

    _combine_kernel_v3[grid_combine](
        partial_o, partial_lse, attn_sink_tensor,
        output, lse,
        total_tokens, h_q, d_v, num_splits,
        stride_po_split, stride_po_t, stride_po_h, stride_po_d,
        stride_plse_split, stride_plse_t, stride_plse_h,
        stride_o_t, stride_o_h, stride_o_d,
        stride_lse_t, stride_lse_h,
        HAS_ATTN_SINK=HAS_ATTN_SINK,
        BLOCK_H=BLOCK_H_COMBINE,
        BLOCK_D=128,
        MAX_SPLITS=MAX_SPLITS,
        num_warps=4,
        num_stages=1,
    )

    return output, lse


def _splitkv_chunked(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    sm_scale: float,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
    max_topk_per_chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked SplitKV for large topk - processes topk in chunks and combines results."""
    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]
    d_v = MODEL1_D_V
    device = q.device

    # Calculate number of chunks
    num_chunks = (topk + max_topk_per_chunk - 1) // max_topk_per_chunk

#    print(f"[DEBUG SplitKV chunked] topk={topk}, h_q={h_q}, num_chunks={num_chunks}, max_topk_per_chunk={max_topk_per_chunk}")

    # Process each chunk and accumulate results using online softmax
    acc_output = None
    acc_lse = None

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * max_topk_per_chunk
        chunk_end = min(chunk_start + max_topk_per_chunk, topk)
        chunk_topk = chunk_end - chunk_start

        # Slice indices for this chunk
        chunk_indices = indices[:, chunk_start:chunk_end].contiguous()

        # Adjust topk_length for this chunk if needed
        chunk_topk_length = None
        if topk_length is not None:
            # Adjust topk_length: subtract chunk_start, clamp to [0, chunk_topk]
            chunk_topk_length = torch.clamp(topk_length - chunk_start, 0, chunk_topk)

        # Process this chunk (no attn_sink for intermediate chunks)
        chunk_output, chunk_lse = _splitkv_single_pass(
            q, kv_cache, chunk_indices, block_size, sm_scale,
            chunk_topk_length, None, s_q  # No attn_sink for chunks
        )

        # Combine with accumulated results using online softmax
        if acc_output is None:
            acc_output = chunk_output.float()
            acc_lse = chunk_lse
        else:
            # Online softmax combination
            # new_lse = log(exp(acc_lse) + exp(chunk_lse))
            # new_output = (exp(acc_lse - new_lse) * acc_output + exp(chunk_lse - new_lse) * chunk_output)
            max_lse = torch.maximum(acc_lse, chunk_lse)
            exp_acc = torch.exp(acc_lse - max_lse)
            exp_chunk = torch.exp(chunk_lse - max_lse)
            sum_exp = exp_acc + exp_chunk

            # Update output: weighted average
            acc_output = (exp_acc.unsqueeze(-1) * acc_output + exp_chunk.unsqueeze(-1) * chunk_output.float()) / sum_exp.unsqueeze(-1)

            # Update lse
            acc_lse = max_lse + torch.log(sum_exp)

    # Apply attn_sink at the end if needed
    if attn_sink is not None:
        # attn_sink shape: [total_tokens, h_q, d_v]
        # Combine with accumulated output using online softmax
        # attn_sink has implicit lse of 0 (weight of 1)
        sink_lse = torch.zeros_like(acc_lse)
        max_lse = torch.maximum(acc_lse, sink_lse)
        exp_acc = torch.exp(acc_lse - max_lse)
        exp_sink = torch.exp(sink_lse - max_lse)
        sum_exp = exp_acc + exp_sink

        acc_output = (exp_acc.unsqueeze(-1) * acc_output + exp_sink.unsqueeze(-1) * attn_sink.float()) / sum_exp.unsqueeze(-1)
        acc_lse = max_lse + torch.log(sum_exp)

    return acc_output.to(torch.bfloat16), acc_lse

