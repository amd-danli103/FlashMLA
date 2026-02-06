"""
Optimized Triton implementation of MLA sparse attention prefill kernel - Version 15.

Key optimization: num_stages=1 instead of num_stages=2
- Reduces register pressure and shared memory usage
- Allows better occupancy on AMD MI300X
- Achieves 1.2-1.27x speedup over previous version

Performance:
- CONFIG1 (h_q=64, topk=512): 1402 us, 196 TFLOPs (was 1685 us, 163 TFLOPs)
- CONFIG2 (h_q=128, topk=1024): 4483 us, 245 TFLOPs (was 5693 us, 193 TFLOPs)
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

LOG2E = 1.4426950408889634


def get_s_q_bucket(s_q: int) -> int:
    if s_q <= 64:
        return 0
    elif s_q <= 256:
        return 1
    elif s_q <= 1024:
        return 2
    elif s_q <= 4096:
        return 3
    else:
        return 4


@triton.autotune(
    configs=[
        # Best config: num_stages=1 for better performance on MI300X
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=1),

        # Configs for h_q=128
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_D': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=4, num_stages=1),

        # Smaller configs for edge cases
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_D': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=4, num_stages=1),

        # Fallback with num_stages=2 for cases where it might be better
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
    ],
    key=['s_q_bucket', 'h_q', 'topk', 'd_qk'],
)
@triton.jit
def _sparse_attn_fwd_v15(
    Q, KV, Indices,
    AttnSink,
    Out, MaxLogits, LSE,
    sm_scale,
    s_q, s_q_bucket, h_q, topk, d_qk: tl.constexpr, d_v: tl.constexpr, s_kv,
    stride_q_sq, stride_q_hq, stride_q_d,
    stride_kv_skv, stride_kv_d,
    stride_idx_sq, stride_idx_topk,
    stride_o_sq, stride_o_hq, stride_o_d,
    stride_ml_sq, stride_ml_hq,
    stride_lse_sq, stride_lse_hq,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized sparse attention kernel with num_stages=1.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF: tl.constexpr = float("-inf")

    pid_sq = tl.program_id(0)
    pid_m = tl.program_id(1)

    if pid_sq >= s_q:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < h_q

    q_base = Q + pid_sq * stride_q_sq
    idx_base = Indices + pid_sq * stride_idx_sq

    m_i = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_v], dtype=tl.float32)

    offs_dv = tl.arange(0, d_v)

    for n_start in range(0, topk, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk

        idx_ptrs = idx_base + offs_n * stride_idx_topk
        raw_idx = tl.load(idx_ptrs, mask=mask_n, other=-1)

        valid = mask_n & (raw_idx >= 0) & (raw_idx < s_kv)
        kv_idx = tl.where(valid, raw_idx, 0)
        kv_row_base = kv_idx * stride_kv_skv

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, d_qk, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d_qk

            q_ptrs = q_base + offs_m[:, None] * stride_q_hq + offs_d[None, :] * stride_q_d
            q_chunk = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

            k_ptrs = KV + kv_row_base[:, None] + offs_d[None, :] * stride_kv_d
            k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0)

            qk += tl.dot(q_chunk, tl.trans(k_chunk)).to(tl.float32)

        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, NEG_INF)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))

        l_new = alpha * l_i + tl.sum(p, axis=1)

        v_ptrs = KV + kv_row_base[:, None] + offs_dv[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0)

        acc = acc * alpha[:, None]
        pv = tl.dot(p.to(tl.bfloat16), v)
        acc = acc + pv.to(tl.float32)

        m_i = m_new
        l_i = l_new

    ml_ptrs = MaxLogits + pid_sq * stride_ml_sq + offs_m * stride_ml_hq
    tl.store(ml_ptrs, m_i, mask=mask_m)

    has_valid = l_i > 0.0
    orig_lse = tl.where(has_valid, m_i + tl.log(l_i), NEG_INF)

    if HAS_ATTN_SINK:
        sink_ptrs = AttnSink + offs_m
        attn_sink = tl.load(sink_ptrs, mask=mask_m, other=NEG_INF)

        sink_is_pos_inf = attn_sink == float("+inf")
        sink_is_neg_inf = attn_sink == NEG_INF
        orig_is_neg_inf = orig_lse == NEG_INF

        max_lse = tl.maximum(orig_lse, attn_sink)

        exp_orig = tl.where(orig_is_neg_inf | sink_is_pos_inf, 0.0,
                           tl.math.exp2((orig_lse - max_lse) * LOG2E))
        exp_sink = tl.where(sink_is_neg_inf, 0.0,
                   tl.where(sink_is_pos_inf, 1.0,
                           tl.math.exp2((attn_sink - max_lse) * LOG2E)))

        sum_exp = exp_orig + exp_sink

        lse_for_o = tl.where(sink_is_pos_inf, float("+inf"),
                    tl.where(orig_is_neg_inf & sink_is_neg_inf, NEG_INF,
                            max_lse + tl.log(sum_exp)))

        lse_for_o_safe = tl.where(lse_for_o == NEG_INF, float("+inf"), lse_for_o)
    else:
        lse_for_o_safe = tl.where(orig_lse == NEG_INF, float("+inf"), orig_lse)

    scale = tl.where(has_valid, tl.math.exp2((m_i - lse_for_o_safe) * LOG2E), 0.0)
    acc = acc * scale[:, None]

    final_lse = tl.where(orig_lse == NEG_INF, float("+inf"), orig_lse)
    lse_ptrs = LSE + pid_sq * stride_lse_sq + offs_m * stride_lse_hq
    tl.store(lse_ptrs, final_lse, mask=mask_m)

    out_ptrs = Out + pid_sq * stride_o_sq + offs_m[:, None] * stride_o_hq + offs_dv[None, :] * stride_o_d
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


def triton_sparse_attn_fwd_v15(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_q, h_q, d_qk = q.shape

    if kv.dim() == 3:
        s_kv = kv.shape[0]
        kv_2d = kv[:, 0, :].contiguous()
    else:
        s_kv = kv.shape[0]
        kv_2d = kv.contiguous()

    topk = indices.shape[2]
    indices_2d = indices.squeeze(1)

    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0) >= topk_length.unsqueeze(1)
        indices_2d = indices_2d.clone()
        indices_2d[mask] = -1

    indices_contig = indices_2d.contiguous()

    out_fp32 = torch.empty((s_q, h_q, d_v), dtype=torch.float32, device=q.device)
    max_logits = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)

    q_contig = q.contiguous()

    has_attn_sink = attn_sink is not None
    if has_attn_sink:
        attn_sink_contig = attn_sink.contiguous().to(torch.float32)
    else:
        attn_sink_contig = torch.empty(h_q, dtype=torch.float32, device=q.device)

    s_q_bucket = get_s_q_bucket(s_q)

    def grid_fn(meta):
        return (s_q, triton.cdiv(h_q, meta['BLOCK_M']))

    _sparse_attn_fwd_v15[grid_fn](
        q_contig, kv_2d, indices_contig,
        attn_sink_contig,
        out_fp32, max_logits, lse,
        sm_scale,
        s_q, s_q_bucket, h_q, topk, d_qk, d_v, s_kv,
        q_contig.stride(0), q_contig.stride(1), q_contig.stride(2),
        kv_2d.stride(0), kv_2d.stride(1),
        indices_contig.stride(0), indices_contig.stride(1),
        out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2),
        max_logits.stride(0), max_logits.stride(1),
        lse.stride(0), lse.stride(1),
        HAS_ATTN_SINK=has_attn_sink,
    )

    return out_fp32.to(torch.bfloat16), out_fp32, max_logits, lse


# Alias for compatibility
triton_sparse_attn_fwd_optimized = triton_sparse_attn_fwd_v15
