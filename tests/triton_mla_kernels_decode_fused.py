"""
Fused Gather+Dequant+Attention Kernel for MODEL1 (d_qk=512)

This module implements a fused kernel that combines:
1. Gather: Load KV from sparse indices
2. Dequant: FP8 to BF16 dequantization
3. Attention: Compute attention scores and output

Benefits for workloads without extra scope:
- Eliminates intermediate buffer (gathered_kv) write/read
- Reduces kernel launch overhead (1 kernel instead of 2)
- Better cache utilization

Supports:
- MODEL1 (d_qk=512): 7 tiles of 64, uint8 scales
- All configs: with/without topk_length, with/without attn_sink

OPTIMIZED VERSION: Reduced code duplication in dual-scope kernel by using
a helper function for KV block processing.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

# ============================================================================
# Constants for MODEL1 layout
# ============================================================================
MODEL1_D_QK = 512
MODEL1_D_NOPE = 448
MODEL1_D_ROPE = 64
MODEL1_D_V = 512
MODEL1_TILE_SIZE = 64
MODEL1_NUM_TILES = 7
MODEL1_BYTES_PER_TOKEN_DATA = 576  # 448 nope + 128 rope
MODEL1_BYTES_PER_TOKEN_SCALE = 8   # 7 scales + 1 padding


# ============================================================================
# Helper: Process KV block and compute QK scores + accumulator update
# This is the core computation shared by both single and dual scope kernels
# ============================================================================
@triton.jit
def _process_kv_block_and_update_acc(
    # KV cache parameters
    kv_block_base,
    nope_rope_offset,
    scale_base_offset,
    valid,
    valid_2d,
    # Query tiles
    q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7,
    # Accumulators (passed by reference via return)
    acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7,
    # Softmax state
    m_i, l_i,
    # Other parameters
    offs_tile,
    sm_scale,
    # Constants
    TILE_SIZE: tl.constexpr,
    D_NOPE: tl.constexpr,
    LOG2E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Process one block of KV tokens: load, dequantize, compute QK, update accumulators.

    This helper function encapsulates the core computation that is repeated for
    both MAIN and EXTRA scopes, eliminating code duplication.

    Returns updated accumulators and softmax state.
    """
    NEG_INF = float("-inf")

    # Load scales
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

    # Tile 1: nope (FP8)
    nope_ptrs = tile_base + TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_1 = (nope_fp8.to(tl.bfloat16) * scale_bf16_1[:, None]).to(tl.bfloat16)
    kv_1 = tl.where(valid_2d, kv_1, 0.0)
    qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)

    # Tile 2: nope (FP8)
    nope_ptrs = tile_base + 2*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_2 = (nope_fp8.to(tl.bfloat16) * scale_bf16_2[:, None]).to(tl.bfloat16)
    kv_2 = tl.where(valid_2d, kv_2, 0.0)
    qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)

    # Tile 3: nope (FP8)
    nope_ptrs = tile_base + 3*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_3 = (nope_fp8.to(tl.bfloat16) * scale_bf16_3[:, None]).to(tl.bfloat16)
    kv_3 = tl.where(valid_2d, kv_3, 0.0)
    qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)

    # Tile 4: nope (FP8)
    nope_ptrs = tile_base + 4*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_4 = (nope_fp8.to(tl.bfloat16) * scale_bf16_4[:, None]).to(tl.bfloat16)
    kv_4 = tl.where(valid_2d, kv_4, 0.0)
    qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)

    # Tile 5: nope (FP8)
    nope_ptrs = tile_base + 5*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_5 = (nope_fp8.to(tl.bfloat16) * scale_bf16_5[:, None]).to(tl.bfloat16)
    kv_5 = tl.where(valid_2d, kv_5, 0.0)
    qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)

    # Tile 6: nope (FP8)
    nope_ptrs = tile_base + 6*TILE_SIZE + offs_tile[None, :]
    nope_uint8 = tl.load(nope_ptrs, mask=valid_2d, other=0)
    nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
    kv_6 = (nope_fp8.to(tl.bfloat16) * scale_bf16_6[:, None]).to(tl.bfloat16)
    kv_6 = tl.where(valid_2d, kv_6, 0.0)
    qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)

    # Tile 7: rope (BF16)
    rope_lo_ptrs = tile_base + D_NOPE + offs_tile[None, :] * 2
    rope_hi_ptrs = tile_base + D_NOPE + offs_tile[None, :] * 2 + 1
    rope_lo = tl.load(rope_lo_ptrs, mask=valid_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_2d, other=0).to(tl.uint16)
    rope_uint16 = rope_lo | (rope_hi << 8)
    kv_7 = rope_uint16.to(tl.bfloat16, bitcast=True)
    kv_7 = tl.where(valid_2d, kv_7, 0.0)
    qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)

    # Apply softmax scale and mask
    qk = qk * sm_scale
    qk = tl.where(valid[None, :], qk, NEG_INF)

    # Online softmax update
    m_ij = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
    p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
    l_new = alpha * l_i + tl.sum(p, axis=1)
    p_bf16 = p.to(tl.bfloat16)

    # Update accumulators
    acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
    acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
    acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
    acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
    acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
    acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
    acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
    acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)

    return acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_new, l_new


# ============================================================================
# MODEL1 Fused Gather+Dequant+Attention Kernel (Single Scope)
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 256}, num_warps=8, num_stages=1),
    ],
    key=["total_tokens", "h_q", "topk"],
)
@triton.jit
def _fused_gather_attn_model1_kernel(
    Q, KV_Cache, Indices, TopkLength, AttnSink,
    Output, LSE,
    sm_scale, total_tokens, h_q, topk, num_blocks, block_size, s_q,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kv_block,
    stride_idx_t, stride_idx_k,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    HAS_TOPK_LENGTH: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused gather+dequant+attention kernel for MODEL1."""
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    D_ROPE: tl.constexpr = 64
    TILE_SIZE: tl.constexpr = 64
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

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

    for n_start in range(0, topk, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk

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

        # Use helper function for KV processing
        acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = \
            _process_kv_block_and_update_acc(
                kv_block_base, nope_rope_offset, scale_base_offset,
                valid, valid_2d,
                q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7,
                acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7,
                m_i, l_i,
                offs_tile, sm_scale,
                TILE_SIZE, D_NOPE, LOG2E, BLOCK_H, BLOCK_N,
            )

    # Finalize
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
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse = tl.where(is_lonely_q, float("+inf"), lse)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64

    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_tile[None, :] * stride_o_d,
             acc_0.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_1.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (2*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_2.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (3*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_3.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (4*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_4.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (5*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_5.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (6*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_6.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (7*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_7.to(tl.bfloat16), mask=mask_h[:, None])

    lse_ptrs = LSE + pid_t * stride_lse_t + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_h)


def _get_block_n(topk: int) -> int:
    """Select BLOCK_N based on topk size."""
    if topk <= 64:
        return 64
    elif topk <= 128:
        return 128
    elif topk <= 256:
        return 256
    else:
        return 256


def fused_gather_attn_decode_model1(
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
    Fused gather+dequant+attention for MODEL1.

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
        output: Attention output [total_tokens, h_q, d_v]
        lse: Log-sum-exp values [total_tokens, h_q]
    """
    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]
    d_v = MODEL1_D_V
    device = q.device

    kv_uint8 = kv_cache.view(torch.uint8)
    num_blocks = kv_cache.shape[0]
    stride_kv_block = kv_uint8.stride(0)
    kv_flat = kv_uint8.reshape(num_blocks, -1)

    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    if not indices.is_contiguous():
        indices = indices.contiguous()

    topk_length_tensor = topk_length if topk_length is not None else lse[:1, 0]
    attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

    grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))

    _fused_gather_attn_model1_kernel[grid](
        q, kv_flat, indices, topk_length_tensor, attn_sink_tensor,
        output, lse,
        sm_scale, total_tokens, h_q, topk, num_blocks, block_size, s_q,
        q.stride(0), q.stride(1), q.stride(2),
        stride_kv_block,
        indices.stride(0), indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_TOPK_LENGTH=topk_length is not None,
        HAS_ATTN_SINK=attn_sink is not None,
    )

    return output, lse


# ============================================================================
# MODEL1 Dual-Scope Fused Gather+Dequant+Attention Kernel (OPTIMIZED)
# Uses helper function to eliminate code duplication
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 256}, num_warps=8, num_stages=1),
    ],
    key=["total_tokens", "h_q", "topk_main", "topk_extra"],
)
@triton.jit
def _fused_gather_attn_model1_dual_scope_kernel(
    Q,
    KV_Cache_Main, Indices_Main, TopkLength_Main,
    KV_Cache_Extra, Indices_Extra, TopkLength_Extra,
    AttnSink,
    Output, LSE,
    sm_scale, total_tokens, h_q,
    topk_main, num_blocks_main, block_size_main,
    topk_extra, num_blocks_extra, block_size_extra,
    s_q,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kv_block_main, stride_kv_block_extra,
    stride_idx_main_t, stride_idx_main_k,
    stride_idx_extra_t, stride_idx_extra_k,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    HAS_TOPK_LENGTH_MAIN: tl.constexpr,
    HAS_TOPK_LENGTH_EXTRA: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    OPTIMIZED fused gather+dequant+attention kernel for MODEL1 with dual scope.

    This version uses a helper function (_process_kv_block_and_update_acc) to
    eliminate the ~200 lines of duplicated code between MAIN and EXTRA scope
    processing loops.

    The kernel processes:
    1. MAIN scope: topk_main tokens from KV_Cache_Main
    2. EXTRA scope: topk_extra tokens from KV_Cache_Extra

    Both scopes contribute to the same online softmax accumulator.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    D_ROPE: tl.constexpr = 64
    TILE_SIZE: tl.constexpr = 64
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

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

    # Load Q tiles (shared by both scopes)
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

    # ========================================================================
    # Process MAIN scope
    # ========================================================================
    for n_start in range(0, topk_main, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk_main

        idx_ptrs = Indices_Main + pid_t * stride_idx_main_t + offs_n * stride_idx_main_k
        indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

        is_invalid = indices == -1
        if HAS_TOPK_LENGTH_MAIN:
            topk_len = tl.load(TopkLength_Main + batch_idx)
            is_invalid = is_invalid | (offs_n >= topk_len)

        valid = mask_n & ~is_invalid
        indices_clamped = tl.maximum(indices, 0)

        block_idx = indices_clamped // block_size_main
        offset_in_block = indices_clamped % block_size_main

        block_idx_64 = block_idx.to(tl.int64)
        offset_in_block_64 = offset_in_block.to(tl.int64)

        kv_block_base = KV_Cache_Main + block_idx_64 * stride_kv_block_main
        nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
        scale_base_offset = block_size_main * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE

        valid_2d = valid[:, None]

        # Use helper function for KV processing
        acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = \
            _process_kv_block_and_update_acc(
                kv_block_base, nope_rope_offset, scale_base_offset,
                valid, valid_2d,
                q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7,
                acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7,
                m_i, l_i,
                offs_tile, sm_scale,
                TILE_SIZE, D_NOPE, LOG2E, BLOCK_H, BLOCK_N,
            )

    # ========================================================================
    # Process EXTRA scope
    # ========================================================================
    for n_start in range(0, topk_extra, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk_extra

        idx_ptrs = Indices_Extra + pid_t * stride_idx_extra_t + offs_n * stride_idx_extra_k
        indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

        is_invalid = indices == -1
        if HAS_TOPK_LENGTH_EXTRA:
            topk_len = tl.load(TopkLength_Extra + batch_idx)
            is_invalid = is_invalid | (offs_n >= topk_len)

        valid = mask_n & ~is_invalid
        indices_clamped = tl.maximum(indices, 0)

        block_idx = indices_clamped // block_size_extra
        offset_in_block = indices_clamped % block_size_extra

        block_idx_64 = block_idx.to(tl.int64)
        offset_in_block_64 = offset_in_block.to(tl.int64)

        kv_block_base = KV_Cache_Extra + block_idx_64 * stride_kv_block_extra
        nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
        scale_base_offset = block_size_extra * BYTES_PER_TOKEN_DATA + offset_in_block_64 * BYTES_PER_TOKEN_SCALE

        valid_2d = valid[:, None]

        # Use helper function for KV processing
        acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = \
            _process_kv_block_and_update_acc(
                kv_block_base, nope_rope_offset, scale_base_offset,
                valid, valid_2d,
                q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7,
                acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7,
                m_i, l_i,
                offs_tile, sm_scale,
                TILE_SIZE, D_NOPE, LOG2E, BLOCK_H, BLOCK_N,
            )

    # ========================================================================
    # Finalize: compute LSE and output
    # ========================================================================
    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    is_lonely_q = (l_i == 0.0)

    # Compute output scale
    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    # Apply output scaling and handle lonely queries
    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse = tl.where(is_lonely_q, float("+inf"), lse)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64

    tl.store(o_base + offs_h[:, None] * stride_o_h + offs_tile[None, :] * stride_o_d,
             acc_0.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_1.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (2*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_2.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (3*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_3.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (4*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_4.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (5*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_5.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (6*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_6.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(o_base + offs_h[:, None] * stride_o_h + (7*TILE_SIZE + offs_tile[None, :]) * stride_o_d,
             acc_7.to(tl.bfloat16), mask=mask_h[:, None])

    lse_ptrs = LSE + pid_t * stride_lse_t + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_h)


def fused_gather_attn_decode_model1_dual_scope(
    q: torch.Tensor,
    kv_cache_main: torch.Tensor,
    indices_main: torch.Tensor,
    block_size_main: int,
    kv_cache_extra: torch.Tensor,
    indices_extra: torch.Tensor,
    block_size_extra: int,
    sm_scale: float,
    topk_length_main: Optional[torch.Tensor] = None,
    topk_length_extra: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gather+dequant+attention for MODEL1 with dual scope (main + extra).

    This kernel processes both main and extra KV scopes in a single kernel,
    treating the key sequence as [main_topk, extra_topk] concatenated.

    Args:
        q: Query tensor [total_tokens, h_q, d_qk]
        kv_cache_main: Quantized main KV cache
        indices_main: Main KV indices [total_tokens, topk_main]
        block_size_main: Block size for main KV cache
        kv_cache_extra: Quantized extra KV cache
        indices_extra: Extra KV indices [total_tokens, topk_extra]
        block_size_extra: Block size for extra KV cache
        sm_scale: Softmax scale
        topk_length_main: Optional per-batch topk length for main [b]
        topk_length_extra: Optional per-batch topk length for extra [b]
        attn_sink: Optional attention sink values [h_q]
        s_q: Sequence length per batch

    Returns:
        output: Attention output [total_tokens, h_q, d_v]
        lse: Log-sum-exp values [total_tokens, h_q]
    """
    total_tokens, h_q, d_qk = q.shape
    topk_main = indices_main.shape[1]
    topk_extra = indices_extra.shape[1]
    d_v = MODEL1_D_V
    device = q.device

    # Prepare main KV cache
    kv_uint8_main = kv_cache_main.view(torch.uint8)
    num_blocks_main = kv_cache_main.shape[0]
    stride_kv_block_main = kv_uint8_main.stride(0)
    kv_flat_main = kv_uint8_main.reshape(num_blocks_main, -1)

    # Prepare extra KV cache
    kv_uint8_extra = kv_cache_extra.view(torch.uint8)
    num_blocks_extra = kv_cache_extra.shape[0]
    stride_kv_block_extra = kv_uint8_extra.stride(0)
    kv_flat_extra = kv_uint8_extra.reshape(num_blocks_extra, -1)

    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    # Ensure indices are contiguous
    if not indices_main.is_contiguous():
        indices_main = indices_main.contiguous()
    if not indices_extra.is_contiguous():
        indices_extra = indices_extra.contiguous()

    # Dummy tensors for optional parameters
    topk_length_main_tensor = topk_length_main if topk_length_main is not None else lse[:1, 0]
    topk_length_extra_tensor = topk_length_extra if topk_length_extra is not None else lse[:1, 0]
    attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

    # Use lambda grid for autotune (BLOCK_H is determined by autotune)
    grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))

    _fused_gather_attn_model1_dual_scope_kernel[grid](
        q,
        kv_flat_main, indices_main, topk_length_main_tensor,
        kv_flat_extra, indices_extra, topk_length_extra_tensor,
        attn_sink_tensor,
        output, lse,
        sm_scale, total_tokens, h_q,
        topk_main, num_blocks_main, block_size_main,
        topk_extra, num_blocks_extra, block_size_extra,
        s_q,
        q.stride(0), q.stride(1), q.stride(2),
        stride_kv_block_main, stride_kv_block_extra,
        indices_main.stride(0), indices_main.stride(1),
        indices_extra.stride(0), indices_extra.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_TOPK_LENGTH_MAIN=topk_length_main is not None,
        HAS_TOPK_LENGTH_EXTRA=topk_length_extra is not None,
        HAS_ATTN_SINK=attn_sink is not None,
    )

    return output, lse
