"""
Optimized Triton implementation of MLA sparse attention prefill kernel - Version 7.

Key optimizations:
1. Use tl.dot with proper BF16 inputs for tensor core acceleration
2. Tile sizes optimized for AMD CDNA3 (16x16 tensor core tiles)
3. Minimize register pressure by processing in smaller chunks
4. Keep Q in registers, stream KV through
5. In-kernel vectorized KV gathering - eliminates expensive pre-gathering
6. **NEW: Fused attention sink computation** - eliminates separate post-processing pass

Matmul precision:
- Q @ K^T and P @ V are computed in BF16 for tensor core acceleration
- Accumulation and intermediate results are kept in FP32 for numerical stability
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

LOG2E = 1.4426950408889634


@triton.autotune(
    configs=[
        # Configs optimized for tensor cores (tile sizes multiple of 16)
        # For h_q=64
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),

        # For h_q=128
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['h_q', 'topk', 'd_qk'],
)
@triton.jit
def _sparse_attn_fwd_fused(
    Q, KV, Indices, InvalidMask,
    AttnSink,  # Attention sink logits [h_q] or None
    Out, MaxLogits, LSE,
    sm_scale,
    s_q, h_q, topk, d_qk: tl.constexpr, d_v: tl.constexpr, s_kv,
    stride_q_sq, stride_q_hq, stride_q_d,
    stride_kv_skv, stride_kv_d,
    stride_idx_sq, stride_idx_topk,
    stride_mask_sq, stride_mask_topk,
    stride_o_sq, stride_o_hq, stride_o_d,
    stride_ml_sq, stride_ml_hq,
    stride_lse_sq, stride_lse_hq,
    HAS_ATTN_SINK: tl.constexpr,  # Whether attention sink is provided
    BLOCK_M: tl.constexpr,  # Block size for heads
    BLOCK_N: tl.constexpr,  # Block size for KV
):
    """
    Sparse attention with in-kernel gathering and fused attention sink.

    Grid: (s_q, cdiv(h_q, BLOCK_M))
    """
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_sq = tl.program_id(0)
    pid_m = tl.program_id(1)

    if pid_sq >= s_q:
        return

    # Head indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < h_q

    # Dimension indices
    offs_d = tl.arange(0, d_qk)
    offs_dv = tl.arange(0, d_v)

    # Load Q: [BLOCK_M, d_qk]
    q_ptrs = Q + pid_sq * stride_q_sq + offs_m[:, None] * stride_q_hq + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_v], dtype=tl.float32)

    # Iterate over KV blocks
    for n_start in range(0, topk, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk

        # Load indices
        idx_ptrs = Indices + pid_sq * stride_idx_sq + offs_n * stride_idx_topk
        kv_idx = tl.load(idx_ptrs, mask=mask_n, other=0)

        # Load invalid mask
        inv_ptrs = InvalidMask + pid_sq * stride_mask_sq + offs_n * stride_mask_topk
        invalid = tl.load(inv_ptrs, mask=mask_n, other=True)

        # Valid mask
        valid = ~invalid & mask_n & (kv_idx >= 0) & (kv_idx < s_kv)

        # Load K: [BLOCK_N, d_qk]
        k_ptrs = KV + kv_idx[:, None] * stride_kv_skv + offs_d[None, :] * stride_kv_d
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0)

        # Load V: [BLOCK_N, d_v]
        v_ptrs = KV + kv_idx[:, None] * stride_kv_skv + offs_dv[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0)

        # Compute Q @ K^T: [BLOCK_M, d_qk] @ [d_qk, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32) * sm_scale

        # Mask invalid positions
        qk = tl.where(valid[None, :], qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale factor
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)
        alpha = tl.where(m_i == float("-inf"), 1.0, alpha)

        # Softmax numerator
        p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
        p = tl.where(qk == float("-inf"), 0.0, p)

        # Update running sum
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Rescale accumulator
        acc = acc * alpha[:, None]

        # Compute P @ V: [BLOCK_M, BLOCK_N] @ [BLOCK_N, d_v] -> [BLOCK_M, d_v]
        pv = tl.dot(p.to(tl.bfloat16), v)
        acc = acc + pv.to(tl.float32)

        # Update state
        m_i = m_new
        l_i = l_new

    # Store max_logits
    ml_ptrs = MaxLogits + pid_sq * stride_ml_sq + offs_m * stride_ml_hq
    tl.store(ml_ptrs, m_i, mask=mask_m)

    # Compute orig_lse
    has_valid = l_i > 0
    orig_lse = tl.where(has_valid, m_i + tl.log(l_i), float("-inf"))

    # Compute the LSE to use for output scaling
    if HAS_ATTN_SINK:
        # Load attention sink for this head block
        sink_ptrs = AttnSink + offs_m
        attn_sink = tl.load(sink_ptrs, mask=mask_m, other=float("-inf"))

        # Handle special cases for +inf and -inf
        sink_is_pos_inf = attn_sink == float("+inf")
        sink_is_neg_inf = attn_sink == float("-inf")
        orig_is_neg_inf = orig_lse == float("-inf")

        # Compute logsumexp(orig_lse, attn_sink)
        max_lse = tl.maximum(orig_lse, attn_sink)

        # Safe exp computation
        exp_orig = tl.where(orig_is_neg_inf | sink_is_pos_inf, 0.0,
                           tl.math.exp2((orig_lse - max_lse) * LOG2E))
        exp_sink = tl.where(sink_is_neg_inf | sink_is_pos_inf, 0.0,
                           tl.math.exp2((attn_sink - max_lse) * LOG2E))
        exp_sink = tl.where(sink_is_pos_inf, 1.0, exp_sink)

        sum_exp = exp_orig + exp_sink

        # Compute lse_for_o
        lse_for_o = tl.where(sink_is_pos_inf, float("+inf"),
                    tl.where(orig_is_neg_inf & sink_is_neg_inf, float("-inf"),
                            max_lse + tl.log(sum_exp)))

        # Convert -inf to +inf for safe division
        lse_for_o_safe = tl.where(lse_for_o == float("-inf"), float("+inf"), lse_for_o)
    else:
        # No attention sink - use orig_lse
        lse_for_o_safe = tl.where(orig_lse == float("-inf"), float("+inf"), orig_lse)

    # Final rescale: acc contains sum(exp(qk - m_i) * V)
    # We need: sum(exp(qk - lse_for_o) * V) = acc * exp(m_i - lse_for_o)
    scale = tl.where(has_valid, tl.math.exp2((m_i - lse_for_o_safe) * LOG2E), 0.0)
    acc = acc * scale[:, None]

    # Store LSE (original LSE, not the combined one)
    final_lse = tl.where(orig_lse == float("-inf"), float("+inf"), orig_lse)
    lse_ptrs = LSE + pid_sq * stride_lse_sq + offs_m * stride_lse_hq
    tl.store(lse_ptrs, final_lse, mask=mask_m)

    # Store output
    out_ptrs = Out + pid_sq * stride_o_sq + offs_m[:, None] * stride_o_hq + offs_dv[None, :] * stride_o_d
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


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
    Optimized sparse attention with tensor core utilization and fused attention sink.
    """
    s_q, h_q, d_qk = q.shape

    if kv.dim() == 3:
        s_kv = kv.shape[0]
        kv_2d = kv[:, 0, :].contiguous()
    else:
        s_kv = kv.shape[0]
        kv_2d = kv.contiguous()

    topk = indices.shape[2]
    indices_2d = indices.squeeze(1).contiguous()

    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0) >= topk_length.unsqueeze(1)
        indices_2d = indices_2d.clone()
        indices_2d[mask] = -1

    invalid_mask = (indices_2d < 0) | (indices_2d >= s_kv)

    out_fp32 = torch.empty((s_q, h_q, d_v), dtype=torch.float32, device=q.device)
    max_logits = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)

    q_contig = q.contiguous()
    indices_contig = indices_2d.contiguous()
    invalid_mask_contig = invalid_mask.contiguous()

    # Prepare attention sink
    has_attn_sink = attn_sink is not None
    if has_attn_sink:
        attn_sink_contig = attn_sink.contiguous().to(torch.float32)
    else:
        attn_sink_contig = torch.empty(h_q, dtype=torch.float32, device=q.device)

    def grid_fn(meta):
        return (s_q, triton.cdiv(h_q, meta['BLOCK_M']))

    _sparse_attn_fwd_fused[grid_fn](
        q_contig, kv_2d, indices_contig, invalid_mask_contig,
        attn_sink_contig,
        out_fp32, max_logits, lse,
        sm_scale,
        s_q, h_q, topk, d_qk, d_v, s_kv,
        q_contig.stride(0), q_contig.stride(1), q_contig.stride(2),
        kv_2d.stride(0), kv_2d.stride(1),
        indices_contig.stride(0), indices_contig.stride(1),
        invalid_mask_contig.stride(0), invalid_mask_contig.stride(1),
        out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2),
        max_logits.stride(0), max_logits.stride(1),
        lse.stride(0), lse.stride(1),
        HAS_ATTN_SINK=has_attn_sink,
    )

    return out_fp32.to(torch.bfloat16), out_fp32, max_logits, lse
