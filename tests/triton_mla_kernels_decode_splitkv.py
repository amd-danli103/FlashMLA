"""
Split-KV Triton MLA Decode Kernels - Optimized Version 4

Key optimizations:
1. Triton kernel for online softmax combination
2. Triton kernel for attn_sink application
3. Use torch.empty instead of torch.zeros
4. Larger chunk size (12288)

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

# Chunk size for processing large topk
MAX_CHUNK_TOPK = 12288


@triton.jit
def _online_softmax_combine_kernel(
    PartialO1, PartialLSE1,
    PartialO2, PartialLSE2,
    Output, LSE,
    total_tokens, h_q, d_v,
    stride_po1_t, stride_po1_h, stride_po1_d,
    stride_plse1_t, stride_plse1_h,
    stride_po2_t, stride_po2_h, stride_po2_d,
    stride_plse2_t, stride_plse2_h,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Combine two partial attention results using online softmax."""
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    POS_INF = float("+inf")
    INF_THRESHOLD = 1e30

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    stride_plse1_t_64 = tl.cast(stride_plse1_t, tl.int64)
    stride_plse2_t_64 = tl.cast(stride_plse2_t, tl.int64)
    lse1_ptrs = PartialLSE1 + pid_t_64 * stride_plse1_t_64 + offs_h * stride_plse1_h
    lse2_ptrs = PartialLSE2 + pid_t_64 * stride_plse2_t_64 + offs_h * stride_plse2_h
    lse1 = tl.load(lse1_ptrs, mask=mask_h, other=NEG_INF)
    lse2 = tl.load(lse2_ptrs, mask=mask_h, other=NEG_INF)

    lse1_invalid = tl.abs(lse1) > INF_THRESHOLD
    lse2_invalid = tl.abs(lse2) > INF_THRESHOLD
    both_invalid = lse1_invalid & lse2_invalid

    lse1_safe = tl.where(lse1_invalid, -INF_THRESHOLD, lse1)
    lse2_safe = tl.where(lse2_invalid, -INF_THRESHOLD, lse2)
    max_lse = tl.maximum(lse1_safe, lse2_safe)

    exp1 = tl.where(lse1_invalid, 0.0, tl.math.exp2((lse1_safe - max_lse) * LOG2E))
    exp2 = tl.where(lse2_invalid, 0.0, tl.math.exp2((lse2_safe - max_lse) * LOG2E))
    sum_exp = exp1 + exp2
    sum_exp_safe = tl.where(both_invalid, 1.0, sum_exp)

    stride_po1_t_64 = tl.cast(stride_po1_t, tl.int64)
    stride_po2_t_64 = tl.cast(stride_po2_t, tl.int64)
    po1_base = PartialO1 + pid_t_64 * stride_po1_t_64 + offs_h[:, None] * stride_po1_h
    po2_base = PartialO2 + pid_t_64 * stride_po2_t_64 + offs_h[:, None] * stride_po2_h

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64 + offs_h[:, None] * stride_o_h

    po1_0 = tl.load(po1_base + offs_d[None, :] * stride_po1_d, mask=mask_h[:, None], other=0.0)
    po2_0 = tl.load(po2_base + offs_d[None, :] * stride_po2_d, mask=mask_h[:, None], other=0.0)
    combined_0 = (exp1[:, None] * po1_0 + exp2[:, None] * po2_0) / sum_exp_safe[:, None]
    combined_0 = tl.where(both_invalid[:, None], 0.0, combined_0)
    tl.store(o_base + offs_d[None, :] * stride_o_d, combined_0.to(tl.bfloat16), mask=mask_h[:, None])

    po1_1 = tl.load(po1_base + (BLOCK_D + offs_d[None, :]) * stride_po1_d, mask=mask_h[:, None], other=0.0)
    po2_1 = tl.load(po2_base + (BLOCK_D + offs_d[None, :]) * stride_po2_d, mask=mask_h[:, None], other=0.0)
    combined_1 = (exp1[:, None] * po1_1 + exp2[:, None] * po2_1) / sum_exp_safe[:, None]
    combined_1 = tl.where(both_invalid[:, None], 0.0, combined_1)
    tl.store(o_base + (BLOCK_D + offs_d[None, :]) * stride_o_d, combined_1.to(tl.bfloat16), mask=mask_h[:, None])

    po1_2 = tl.load(po1_base + (2*BLOCK_D + offs_d[None, :]) * stride_po1_d, mask=mask_h[:, None], other=0.0)
    po2_2 = tl.load(po2_base + (2*BLOCK_D + offs_d[None, :]) * stride_po2_d, mask=mask_h[:, None], other=0.0)
    combined_2 = (exp1[:, None] * po1_2 + exp2[:, None] * po2_2) / sum_exp_safe[:, None]
    combined_2 = tl.where(both_invalid[:, None], 0.0, combined_2)
    tl.store(o_base + (2*BLOCK_D + offs_d[None, :]) * stride_o_d, combined_2.to(tl.bfloat16), mask=mask_h[:, None])

    po1_3 = tl.load(po1_base + (3*BLOCK_D + offs_d[None, :]) * stride_po1_d, mask=mask_h[:, None], other=0.0)
    po2_3 = tl.load(po2_base + (3*BLOCK_D + offs_d[None, :]) * stride_po2_d, mask=mask_h[:, None], other=0.0)
    combined_3 = (exp1[:, None] * po1_3 + exp2[:, None] * po2_3) / sum_exp_safe[:, None]
    combined_3 = tl.where(both_invalid[:, None], 0.0, combined_3)
    tl.store(o_base + (3*BLOCK_D + offs_d[None, :]) * stride_o_d, combined_3.to(tl.bfloat16), mask=mask_h[:, None])

    combined_lse = max_lse + tl.math.log2(sum_exp_safe) / LOG2E
    combined_lse = tl.where(both_invalid, POS_INF, combined_lse)

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    lse_ptrs = LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h
    tl.store(lse_ptrs, combined_lse, mask=mask_h)


@triton.jit
def _apply_attn_sink_kernel(
    Output, LSE, AttnSink,
    total_tokens, h_q, d_v,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Apply attention sink scaling to output."""
    LOG2E: tl.constexpr = 1.4426950408889634
    POS_INF = float("+inf")
    INF_THRESHOLD = 1e30

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    lse_ptrs = LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h
    lse_vals = tl.load(lse_ptrs, mask=mask_h, other=POS_INF)

    attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)

    is_lonely = lse_vals > INF_THRESHOLD
    lse_safe = tl.where(is_lonely, 0.0, lse_vals)

    diff = attn_sink_vals - lse_safe
    diff_clamped = tl.minimum(tl.maximum(diff, -100.0), 100.0)
    exp_diff = tl.math.exp2(diff_clamped * LOG2E)
    exp_diff = tl.where(is_lonely, 0.0, exp_diff)

    denominator = 1.0 + exp_diff
    scale = 1.0 / denominator
    scale = tl.where(is_lonely, 1.0, scale)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64 + offs_h[:, None] * stride_o_h

    o_0 = tl.load(o_base + offs_d[None, :] * stride_o_d, mask=mask_h[:, None], other=0.0)
    tl.store(o_base + offs_d[None, :] * stride_o_d, (o_0 * scale[:, None]).to(tl.bfloat16), mask=mask_h[:, None])

    o_1 = tl.load(o_base + (BLOCK_D + offs_d[None, :]) * stride_o_d, mask=mask_h[:, None], other=0.0)
    tl.store(o_base + (BLOCK_D + offs_d[None, :]) * stride_o_d, (o_1 * scale[:, None]).to(tl.bfloat16), mask=mask_h[:, None])

    o_2 = tl.load(o_base + (2*BLOCK_D + offs_d[None, :]) * stride_o_d, mask=mask_h[:, None], other=0.0)
    tl.store(o_base + (2*BLOCK_D + offs_d[None, :]) * stride_o_d, (o_2 * scale[:, None]).to(tl.bfloat16), mask=mask_h[:, None])

    o_3 = tl.load(o_base + (3*BLOCK_D + offs_d[None, :]) * stride_o_d, mask=mask_h[:, None], other=0.0)
    tl.store(o_base + (3*BLOCK_D + offs_d[None, :]) * stride_o_d, (o_3 * scale[:, None]).to(tl.bfloat16), mask=mask_h[:, None])


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
    """Split-KV sparse attention decode for MODEL1."""
    from triton_mla_kernels_decode_model1 import gather_dequant_fp8_model1
    from triton_mla_kernels_decode_common import run_unified_attention

    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]
    d_v = MODEL1_D_V
    device = q.device

    if topk <= MAX_CHUNK_TOPK:
        # Single pass
        gathered_kv = torch.empty(total_tokens, topk, d_qk, dtype=torch.bfloat16, device=device)
        invalid_mask = torch.empty(total_tokens, topk, dtype=torch.bool, device=device)

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

    # Chunked processing
    num_chunks = (topk + MAX_CHUNK_TOPK - 1) // MAX_CHUNK_TOPK

    q_reshaped = q.to(torch.bfloat16)
    if not q_reshaped.is_contiguous():
        q_reshaped = q_reshaped.contiguous()

    # Process first chunk
    chunk_start = 0
    chunk_end = min(MAX_CHUNK_TOPK, topk)
    chunk_topk = chunk_end - chunk_start

    chunk_indices = indices[:, chunk_start:chunk_end].contiguous()
    chunk_topk_length = None
    if topk_length is not None:
        chunk_topk_length = torch.clamp(topk_length - chunk_start, 0, chunk_topk)

    gathered_kv = torch.empty(total_tokens, chunk_topk, d_qk, dtype=torch.bfloat16, device=device)
    invalid_mask = torch.empty(total_tokens, chunk_topk, dtype=torch.bool, device=device)

    gather_dequant_fp8_model1(
        kv_cache, chunk_indices, block_size,
        gathered_kv, invalid_mask, 0, chunk_topk_length, s_q
    )

    acc_output, acc_lse = run_unified_attention(
        q_reshaped, gathered_kv, invalid_mask,
        d_v, sm_scale, total_tokens, h_q, chunk_topk, d_qk,
        attn_sink=None
    )

    acc_output = acc_output.float()

    BLOCK_H = 16
    BLOCK_D = 128
    n_h_blocks = (h_q + BLOCK_H - 1) // BLOCK_H
    grid = (total_tokens, n_h_blocks)

    # Process remaining chunks
    for chunk_idx in range(1, num_chunks):
        chunk_start = chunk_idx * MAX_CHUNK_TOPK
        chunk_end = min(chunk_start + MAX_CHUNK_TOPK, topk)
        chunk_topk = chunk_end - chunk_start

        chunk_indices = indices[:, chunk_start:chunk_end].contiguous()
        chunk_topk_length = None
        if topk_length is not None:
            chunk_topk_length = torch.clamp(topk_length - chunk_start, 0, chunk_topk)

        gathered_kv = torch.empty(total_tokens, chunk_topk, d_qk, dtype=torch.bfloat16, device=device)
        invalid_mask = torch.empty(total_tokens, chunk_topk, dtype=torch.bool, device=device)

        gather_dequant_fp8_model1(
            kv_cache, chunk_indices, block_size,
            gathered_kv, invalid_mask, 0, chunk_topk_length, s_q
        )

        chunk_output, chunk_lse = run_unified_attention(
            q_reshaped, gathered_kv, invalid_mask,
            d_v, sm_scale, total_tokens, h_q, chunk_topk, d_qk,
            attn_sink=None
        )

        combined_output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
        combined_lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

        _online_softmax_combine_kernel[grid](
            acc_output, acc_lse,
            chunk_output.float(), chunk_lse,
            combined_output, combined_lse,
            total_tokens, h_q, d_v,
            acc_output.stride(0), acc_output.stride(1), acc_output.stride(2),
            acc_lse.stride(0), acc_lse.stride(1),
            chunk_output.stride(0), chunk_output.stride(1), chunk_output.stride(2),
            chunk_lse.stride(0), chunk_lse.stride(1),
            combined_output.stride(0), combined_output.stride(1), combined_output.stride(2),
            combined_lse.stride(0), combined_lse.stride(1),
            BLOCK_H=BLOCK_H,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=1,
        )

        acc_output = combined_output.float()
        acc_lse = combined_lse

    # Apply attn_sink if present
    if attn_sink is not None:
        output = acc_output.to(torch.bfloat16)

        _apply_attn_sink_kernel[grid](
            output, acc_lse, attn_sink,
            total_tokens, h_q, d_v,
            output.stride(0), output.stride(1), output.stride(2),
            acc_lse.stride(0), acc_lse.stride(1),
            BLOCK_H=BLOCK_H,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=1,
        )
        return output, acc_lse
    else:
        return acc_output.to(torch.bfloat16), acc_lse
