"""
Optimized Triton implementation of MLA sparse attention prefill kernel.

Key optimizations:
1. Pre-gather KV using PyTorch's optimized index_select
2. Process multiple heads per block using tl.dot for tensor core utilization
3. Online softmax to avoid materializing full attention matrix
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _multihead_sparse_attention_kernel(
    Q, GatheredKV, InvalidMask,
    Out, MaxLogits, LSE,
    sm_scale,
    s_q, h_q, topk, d_qk, d_v,
    stride_q_sq, stride_q_hq, stride_q_d,
    stride_kv_sq, stride_kv_topk, stride_kv_d,
    stride_mask_sq, stride_mask_topk,
    stride_o_sq, stride_o_hq, stride_o_d,
    stride_ml_sq, stride_ml_hq,
    stride_lse_sq, stride_lse_hq,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Multi-head sparse attention kernel.
    Processes BLOCK_H heads per block using tl.dot for tensor core utilization.

    Grid: (s_q, cdiv(h_q, BLOCK_H))
    """
    pid_sq = tl.program_id(0)
    pid_h_block = tl.program_id(1)

    if pid_sq >= s_q:
        return

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    offs_d = tl.arange(0, BLOCK_D)
    mask_d_qk = offs_d < d_qk
    mask_d_v = offs_d < d_v

    q_ptrs = Q + pid_sq * stride_q_sq + offs_h[:, None] * stride_q_hq + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d_qk[None, :], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    for n_start in range(0, topk, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk

        mask_ptrs = InvalidMask + pid_sq * stride_mask_sq + offs_n * stride_mask_topk
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True)
        valid = ~invalid & mask_n

        k_ptrs = GatheredKV + pid_sq * stride_kv_sq + offs_n[:, None] * stride_kv_topk + offs_d[None, :] * stride_kv_d
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d_qk[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * sm_scale
        scores = tl.where(valid[None, :], scores, float("-inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.where(m_i == float("-inf"), 1.0, tl.exp(m_i - m_new))
        p_raw = tl.exp(scores - m_new[:, None])
        p = tl.where(scores == float("-inf"), 0.0, p_raw)

        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d_v[None, :], other=0.0).to(tl.float32)
        acc = acc + tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    has_valid = l_i > 0
    orig_lse = tl.where(has_valid, m_i + tl.log(l_i), float("-inf"))
    final_lse = tl.where(orig_lse == float("-inf"), float("+inf"), orig_lse)

    lse_for_o = tl.where(orig_lse == float("-inf"), float("+inf"), orig_lse)
    scale = tl.where(has_valid, tl.exp(m_i - lse_for_o), 0.0)
    acc = acc * scale[:, None]

    ml_ptrs = MaxLogits + pid_sq * stride_ml_sq + offs_h * stride_ml_hq
    tl.store(ml_ptrs, m_i, mask=mask_h)

    lse_ptrs = LSE + pid_sq * stride_lse_sq + offs_h * stride_lse_hq
    tl.store(lse_ptrs, final_lse, mask=mask_h)

    out_ptrs = Out + pid_sq * stride_o_sq + offs_h[:, None] * stride_o_hq + offs_d[None, :] * stride_o_d
    tl.store(out_ptrs, acc, mask=mask_h[:, None] & mask_d_v[None, :])


def triton_sparse_attn_fwd_optimized(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized sparse attention using pre-gathered KV and multi-head Triton kernel.
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    topk = indices.shape[2]

    indices_2d = indices.squeeze(1).contiguous()

    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0) >= topk_length.unsqueeze(1)
        indices_2d = indices_2d.clone()
        indices_2d[mask] = -1

    invalid_mask = (indices_2d < 0) | (indices_2d >= s_kv)
    safe_indices = indices_2d.masked_fill(invalid_mask, 0)

    gathered_kv = kv.index_select(0, safe_indices.reshape(-1)).reshape(s_q, topk, d_qk).contiguous()

    out_fp32 = torch.empty((s_q, h_q, d_v), dtype=torch.float32, device=q.device)
    max_logits = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)

    q_contig = q.contiguous()
    invalid_mask_contig = invalid_mask.contiguous()

    if d_qk <= 512 and d_v <= 512:
        BLOCK_D = 512
    elif d_qk <= 1024 and d_v <= 1024:
        BLOCK_D = 1024
    else:
        BLOCK_D = 2048

    BLOCK_H = 64
    BLOCK_N = 32

    grid = (s_q, triton.cdiv(h_q, BLOCK_H))

    _multihead_sparse_attention_kernel[grid](
        q_contig, gathered_kv, invalid_mask_contig,
        out_fp32, max_logits, lse,
        sm_scale,
        s_q, h_q, topk, d_qk, d_v,
        q_contig.stride(0), q_contig.stride(1), q_contig.stride(2),
        gathered_kv.stride(0), gathered_kv.stride(1), gathered_kv.stride(2),
        invalid_mask_contig.stride(0), invalid_mask_contig.stride(1),
        out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2),
        max_logits.stride(0), max_logits.stride(1),
        lse.stride(0), lse.stride(1),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    if attn_sink is not None:
        orig_lse = lse.clone()
        orig_lse[orig_lse == float("+inf")] = float("-inf")

        lse_for_o = torch.logsumexp(
            torch.stack([orig_lse, attn_sink.view(1, h_q).expand(s_q, h_q)], dim=0),
            dim=0
        )

        lse_for_o_safe = torch.where(lse_for_o == float("-inf"), float("+inf"), lse_for_o)
        scale = torch.where(orig_lse == float("-inf"), 0.0, torch.exp(orig_lse - lse_for_o_safe))
        out_fp32 = out_fp32 * scale.unsqueeze(-1)

    return out_fp32.to(torch.bfloat16), out_fp32, max_logits, lse
