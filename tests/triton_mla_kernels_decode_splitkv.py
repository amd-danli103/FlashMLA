"""
Split-KV Triton MLA Decode Kernels

This module implements Split-KV optimization for sparse attention decode:
1. Split topk into chunks (SPLIT_K_SIZE = 64)
2. Each chunk computes partial O and partial LSE in parallel
3. A combine kernel merges partial results using LSE-based weighted sum

This approach better utilizes SMs for large batch/topk workloads.

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

# Split-KV chunk size (matches CUDA implementation)
SPLIT_K_SIZE = 64


# ============================================================================
# Split-KV Kernel: Compute partial O and LSE for each chunk
# ============================================================================
@triton.jit
def _splitkv_partial_kernel(
    Q, KV_Cache, Indices, TopkLength, AttnSink,
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
    """
    Compute partial attention output and LSE for a chunk of KV tokens.

    Grid: (total_tokens, num_h_blocks, num_splits)
    Each program handles one token, one head block, one KV chunk.

    Output:
    - PartialO: Normalized partial output (O / l_local)
    - PartialLSE: Log-sum-exp = m + log(l)
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8
    NEG_INF = float("-inf")

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_split = tl.program_id(2)
    pid_t_64 = pid_t.to(tl.int64)

    # Compute KV range for this split
    kv_start = pid_split * BLOCK_N
    kv_end = tl.minimum(kv_start + BLOCK_N, topk)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    # Initialize accumulators
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

    # Load Q tiles
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

    # Early exit if this split has no work
    if kv_start >= topk:
        # Store -inf LSE and zero output for empty splits
        stride_po_split_64 = tl.cast(stride_po_split, tl.int64)
        stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
        po_base = PartialO + pid_split * stride_po_split_64 + pid_t_64 * stride_po_t_64
        row_ptrs = po_base + offs_h[:, None] * stride_po_h
        zeros = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
        tl.store(row_ptrs + offs_tile[None, :] * stride_po_d, zeros, mask=mask_h[:, None])
        tl.store(row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_po_d, zeros, mask=mask_h[:, None])
        tl.store(row_ptrs + (2*TILE_SIZE + offs_tile[None, :]) * stride_po_d, zeros, mask=mask_h[:, None])
        tl.store(row_ptrs + (3*TILE_SIZE + offs_tile[None, :]) * stride_po_d, zeros, mask=mask_h[:, None])
        tl.store(row_ptrs + (4*TILE_SIZE + offs_tile[None, :]) * stride_po_d, zeros, mask=mask_h[:, None])
        tl.store(row_ptrs + (5*TILE_SIZE + offs_tile[None, :]) * stride_po_d, zeros, mask=mask_h[:, None])
        tl.store(row_ptrs + (6*TILE_SIZE + offs_tile[None, :]) * stride_po_d, zeros, mask=mask_h[:, None])
        tl.store(row_ptrs + (7*TILE_SIZE + offs_tile[None, :]) * stride_po_d, zeros, mask=mask_h[:, None])

        stride_plse_split_64 = tl.cast(stride_plse_split, tl.int64)
        stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)
        plse_base = PartialLSE + pid_split * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
        plse_ptrs = plse_base + offs_h * stride_plse_h
        tl.store(plse_ptrs, tl.full([BLOCK_H], NEG_INF, dtype=tl.float32), mask=mask_h)
        return

    # Process KV tokens in this chunk
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

    block_idx_64 = block_idx.to(tl.int64)
    offset_in_block_64 = offset_in_block.to(tl.int64)

    kv_block_base = KV_Cache + block_idx_64 * stride_kv_block
    nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
    scale_base_offset = block_size * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE

    valid_2d = valid[:, None]

    # Load scales (uint8 values)
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

    # Tile 0: nope (FP8)
    nope_ptrs = tile_base + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_0 = (nope_fp8.to(tl.bfloat16) * scale_bf16_0[:, None]).to(tl.bfloat16)
    kv_0 = tl.where(valid_2d, kv_0, 0.0)
    qk += tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)

    # Tile 1
    nope_ptrs = tile_base + TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_1 = (nope_fp8.to(tl.bfloat16) * scale_bf16_1[:, None]).to(tl.bfloat16)
    kv_1 = tl.where(valid_2d, kv_1, 0.0)
    qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)

    # Tile 2
    nope_ptrs = tile_base + 2*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_2 = (nope_fp8.to(tl.bfloat16) * scale_bf16_2[:, None]).to(tl.bfloat16)
    kv_2 = tl.where(valid_2d, kv_2, 0.0)
    qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)

    # Tile 3
    nope_ptrs = tile_base + 3*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_3 = (nope_fp8.to(tl.bfloat16) * scale_bf16_3[:, None]).to(tl.bfloat16)
    kv_3 = tl.where(valid_2d, kv_3, 0.0)
    qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)

    # Tile 4
    nope_ptrs = tile_base + 4*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_4 = (nope_fp8.to(tl.bfloat16) * scale_bf16_4[:, None]).to(tl.bfloat16)
    kv_4 = tl.where(valid_2d, kv_4, 0.0)
    qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)

    # Tile 5
    nope_ptrs = tile_base + 5*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_5 = (nope_fp8.to(tl.bfloat16) * scale_bf16_5[:, None]).to(tl.bfloat16)
    kv_5 = tl.where(valid_2d, kv_5, 0.0)
    qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)

    # Tile 6
    nope_ptrs = tile_base + 6*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_6 = (nope_fp8.to(tl.bfloat16) * scale_bf16_6[:, None]).to(tl.bfloat16)
    kv_6 = tl.where(valid_2d, kv_6, 0.0)
    qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)

    # Tile 7: rope (BF16)
    rope_ptrs = tile_base + D_NOPE + offs_tile[None, :] * 2
    rope_lo = tl.load(rope_ptrs, mask=valid_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
    kv_7 = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
    kv_7 = tl.where(valid_2d, kv_7, 0.0)
    qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)

    # Apply softmax scale and mask
    qk = qk * sm_scale
    qk = tl.where(valid[None, :], qk, NEG_INF)

    # Compute local softmax
    m_ij = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
    p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
    l_new = alpha * l_i + tl.sum(p, axis=1)
    p_bf16 = p.to(tl.bfloat16)

    # Update accumulators (unnormalized: acc = sum(exp(qk - m) * v))
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

    # Compute partial LSE = m + log(l)
    partial_lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E

    # Normalize partial output: O_normalized = acc / l
    # This way, combine kernel can do: O_final = sum(O_normalized_j * exp(lse_j - global_lse))
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc_0 = acc_0 / l_safe[:, None]
    acc_1 = acc_1 / l_safe[:, None]
    acc_2 = acc_2 / l_safe[:, None]
    acc_3 = acc_3 / l_safe[:, None]
    acc_4 = acc_4 / l_safe[:, None]
    acc_5 = acc_5 / l_safe[:, None]
    acc_6 = acc_6 / l_safe[:, None]
    acc_7 = acc_7 / l_safe[:, None]

    # Store partial O (normalized)
    stride_po_split_64 = tl.cast(stride_po_split, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
    po_base = PartialO + pid_split * stride_po_split_64 + pid_t_64 * stride_po_t_64

    row_ptrs = po_base + offs_h[:, None] * stride_po_h
    tl.store(row_ptrs + offs_tile[None, :] * stride_po_d, acc_0, mask=mask_h[:, None])
    tl.store(row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_1, mask=mask_h[:, None])
    tl.store(row_ptrs + (2*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_2, mask=mask_h[:, None])
    tl.store(row_ptrs + (3*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_3, mask=mask_h[:, None])
    tl.store(row_ptrs + (4*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_4, mask=mask_h[:, None])
    tl.store(row_ptrs + (5*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_5, mask=mask_h[:, None])
    tl.store(row_ptrs + (6*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_6, mask=mask_h[:, None])
    tl.store(row_ptrs + (7*TILE_SIZE + offs_tile[None, :]) * stride_po_d, acc_7, mask=mask_h[:, None])

    # Store partial LSE
    stride_plse_split_64 = tl.cast(stride_plse_split, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)
    plse_base = PartialLSE + pid_split * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
    plse_ptrs = plse_base + offs_h * stride_plse_h
    # For empty chunks (l_i == 0), store -inf
    partial_lse = tl.where(l_i == 0.0, NEG_INF, partial_lse)
    tl.store(plse_ptrs, partial_lse, mask=mask_h)


# ============================================================================
# Combine Kernel: Merge partial results
# ============================================================================
@triton.jit
def _combine_kernel(
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
    """
    Combine partial O and LSE from all splits.

    Grid: (total_tokens, num_h_blocks)

    Each partial O is normalized (O_j / l_j), and partial LSE = m_j + log(l_j).
    Final O = sum(O_j * exp(lse_j - global_lse)) where global_lse = log(sum(exp(lse_j)))
    """
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

    # First pass: find global max LSE
    global_max_lse = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    for split_idx in range(MAX_SPLITS):
        if split_idx < num_splits:
            plse_base = PartialLSE + split_idx * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
            plse_ptrs = plse_base + offs_h * stride_plse_h
            partial_lse = tl.load(plse_ptrs, mask=mask_h, other=NEG_INF)
            global_max_lse = tl.maximum(global_max_lse, partial_lse)

    # Handle case where all LSEs are -inf (no valid tokens)
    all_invalid = (global_max_lse == NEG_INF)
    global_max_lse = tl.where(all_invalid, 0.0, global_max_lse)

    # Second pass: compute sum of exp(lse - max)
    sum_exp = tl.zeros([BLOCK_H], dtype=tl.float32)
    for split_idx in range(MAX_SPLITS):
        if split_idx < num_splits:
            plse_base = PartialLSE + split_idx * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
            plse_ptrs = plse_base + offs_h * stride_plse_h
            partial_lse = tl.load(plse_ptrs, mask=mask_h, other=NEG_INF)
            exp_diff = tl.where(partial_lse == NEG_INF, 0.0, tl.math.exp2((partial_lse - global_max_lse) * LOG2E))
            sum_exp += exp_diff

    # Compute global LSE = max + log(sum_exp)
    global_lse = global_max_lse + tl.math.log2(tl.where(sum_exp == 0.0, 1.0, sum_exp)) / LOG2E
    is_lonely_q = all_invalid

    # Initialize output accumulators
    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    stride_po_split_64 = tl.cast(stride_po_split, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
    offs_d = tl.arange(0, BLOCK_D)

    # Third pass: weighted sum of partial outputs
    for split_idx in range(MAX_SPLITS):
        if split_idx < num_splits:
            # Load partial LSE for this split
            plse_base = PartialLSE + split_idx * stride_plse_split_64 + pid_t_64 * stride_plse_t_64
            plse_ptrs = plse_base + offs_h * stride_plse_h
            partial_lse = tl.load(plse_ptrs, mask=mask_h, other=NEG_INF)

            # Compute weight: exp(partial_lse - global_lse)
            weight = tl.where(partial_lse == NEG_INF, 0.0, tl.math.exp2((partial_lse - global_lse) * LOG2E))

            # Load and accumulate partial outputs
            po_base = PartialO + split_idx * stride_po_split_64 + pid_t_64 * stride_po_t_64
            row_ptrs = po_base + offs_h[:, None] * stride_po_h

            # Tile 0
            po_0 = tl.load(row_ptrs + offs_d[None, :] * stride_po_d, mask=mask_h[:, None], other=0.0)
            acc_0 += weight[:, None] * po_0

            # Tile 1
            po_1 = tl.load(row_ptrs + (BLOCK_D + offs_d[None, :]) * stride_po_d, mask=mask_h[:, None], other=0.0)
            acc_1 += weight[:, None] * po_1

            # Tile 2
            po_2 = tl.load(row_ptrs + (2*BLOCK_D + offs_d[None, :]) * stride_po_d, mask=mask_h[:, None], other=0.0)
            acc_2 += weight[:, None] * po_2

            # Tile 3
            po_3 = tl.load(row_ptrs + (3*BLOCK_D + offs_d[None, :]) * stride_po_d, mask=mask_h[:, None], other=0.0)
            acc_3 += weight[:, None] * po_3

    # Apply attn_sink if present
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

    # Handle lonely queries
    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0)
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1)
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2)
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3)
    final_lse = tl.where(is_lonely_q, POS_INF, global_lse)

    # Store output
    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64
    row_ptrs = o_base + offs_h[:, None] * stride_o_h

    tl.store(row_ptrs + offs_d[None, :] * stride_o_d, acc_0.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(row_ptrs + (BLOCK_D + offs_d[None, :]) * stride_o_d, acc_1.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(row_ptrs + (2*BLOCK_D + offs_d[None, :]) * stride_o_d, acc_2.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(row_ptrs + (3*BLOCK_D + offs_d[None, :]) * stride_o_d, acc_3.to(tl.bfloat16), mask=mask_h[:, None])

    # Store LSE
    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    lse_ptrs = LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h
    tl.store(lse_ptrs, final_lse, mask=mask_h)


# ============================================================================
# Python wrapper for Split-KV attention
# ============================================================================
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
    """
    Split-KV sparse attention decode for MODEL1.

    Args:
        q: Query tensor [total_tokens, h_q, d_qk]
        kv_cache: Quantized KV cache
        indices: KV indices [total_tokens, topk]
        block_size: Block size for KV cache
        sm_scale: Softmax scale
        topk_length: Optional per-batch topk length [b]
        attn_sink: Optional attention sink values [h_q]
        s_q: Sequence length per batch

    Returns:
        output: [total_tokens, h_q, d_v]
        lse: [total_tokens, h_q]
    """
    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]
    d_v = MODEL1_D_V
    device = q.device

    # Convert KV cache to uint8 view (same as fused kernel)
    kv_uint8 = kv_cache.view(torch.uint8)
    num_blocks = kv_cache.shape[0]
    stride_kv_block = kv_uint8.stride(0)
    kv_flat = kv_uint8.reshape(num_blocks, -1)

    # Compute number of splits
    num_splits = (topk + SPLIT_K_SIZE - 1) // SPLIT_K_SIZE

    # Allocate partial buffers
    partial_o = torch.empty(num_splits, total_tokens, h_q, d_v, dtype=torch.float32, device=device)
    partial_lse = torch.empty(num_splits, total_tokens, h_q, dtype=torch.float32, device=device)

    # Output buffers
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    # Ensure Q is contiguous bfloat16
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    if not indices.is_contiguous():
        indices = indices.contiguous()

    # Get strides
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

    # Choose BLOCK_H based on h_q
    BLOCK_H = 16 if h_q <= 64 else 32
    num_h_blocks = (h_q + BLOCK_H - 1) // BLOCK_H

    # Launch split-KV kernel
    grid_splitkv = (total_tokens, num_h_blocks, num_splits)
    _splitkv_partial_kernel[grid_splitkv](
        q, kv_flat, indices, topk_length_tensor, attn_sink_tensor,
        partial_o, partial_lse,
        sm_scale, total_tokens, h_q, topk, num_blocks, block_size, s_q,
        num_splits,
        stride_q_t, stride_q_h, stride_q_d,
        stride_kv_block,
        stride_idx_t, stride_idx_k,
        stride_po_split, stride_po_t, stride_po_h, stride_po_d,
        stride_plse_split, stride_plse_t, stride_plse_h,
        HAS_TOPK_LENGTH=HAS_TOPK_LENGTH,
        BLOCK_H=BLOCK_H,
        BLOCK_N=SPLIT_K_SIZE,
        TILE_SIZE=MODEL1_TILE_SIZE,
        num_warps=4,
        num_stages=1,
    )

    # Determine MAX_SPLITS for combine kernel
    if num_splits <= 4:
        MAX_SPLITS = 4
    elif num_splits <= 8:
        MAX_SPLITS = 8
    elif num_splits <= 16:
        MAX_SPLITS = 16
    elif num_splits <= 32:
        MAX_SPLITS = 32
    else:
        MAX_SPLITS = 64

    # Launch combine kernel
    grid_combine = (total_tokens, num_h_blocks)
    _combine_kernel[grid_combine](
        partial_o, partial_lse, attn_sink_tensor,
        output, lse,
        total_tokens, h_q, d_v, num_splits,
        stride_po_split, stride_po_t, stride_po_h, stride_po_d,
        stride_plse_split, stride_plse_t, stride_plse_h,
        stride_o_t, stride_o_h, stride_o_d,
        stride_lse_t, stride_lse_h,
        HAS_ATTN_SINK=HAS_ATTN_SINK,
        BLOCK_H=BLOCK_H,
        BLOCK_D=128,
        MAX_SPLITS=MAX_SPLITS,
        num_warps=4,
        num_stages=1,
    )

    return output, lse
