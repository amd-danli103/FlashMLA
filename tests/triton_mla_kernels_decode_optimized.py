"""
Optimized Triton implementation of MLA sparse attention decode kernel.

Key Optimizations:
1. Fused attn_sink and lonely_q masking into the kernel
2. Reduced post-processing overhead
3. Fixed optimal configuration: BLOCK_H=16, BLOCK_N=64, num_warps=4, num_stages=1
4. Use bfloat16 for matrix multiplications to leverage tensor cores

Supports both d_qk=512 and d_qk=576 cases.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

# Constants
LOG2E = 1.4426950408889634

# Optimal block sizes - found through extensive benchmarking
BLOCK_H_OPT = 16
BLOCK_N_OPT = 64
BLOCK_D_OPT = 128


@triton.jit
def _sparse_decode_attn_kernel_fused(
    Q,              # [total_tokens, h_q, d_qk]
    KV,             # [total_tokens, topk, d_qk]
    InvalidMask,    # [total_tokens, topk]
    AttnSink,       # [h_q] or None
    Output,         # [total_tokens, h_q, d_v]
    LSE,            # [total_tokens, h_q]
    sm_scale,
    total_tokens,
    h_q,
    topk,
    d_qk,
    d_v,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kv_t, stride_kv_k, stride_kv_d,
    stride_mask_t, stride_mask_k,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized Triton kernel for sparse attention decode with fused post-processing.
    Fuses attn_sink scaling and lonely_q masking into the kernel.
    Uses bfloat16 for matrix multiplications to leverage tensor cores.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF: tl.constexpr = float("-inf")
    POS_INF: tl.constexpr = float("inf")

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= total_tokens:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    # Initialize online softmax state (keep in float32 for numerical stability)
    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Accumulators for output - 4 chunks for d_v=512 (keep in float32 for accumulation)
    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    q_base = Q + pid_t * stride_q_t
    kv_base = KV + pid_t * stride_kv_t
    mask_base = InvalidMask + pid_t * stride_mask_t

    # Main loop over topk
    for n_start in range(0, topk, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk

        # Load invalid mask
        mask_ptrs = mask_base + offs_n * stride_mask_k
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True)
        valid = mask_n & ~invalid

        # Compute Q @ K^T in chunks using bfloat16 for tensor core acceleration
        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        # Process d_qk in chunks of BLOCK_D
        for d_start in range(0, d_qk, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d_qk

            # Load Q chunk in bfloat16
            q_ptrs = q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
            q_chunk = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

            # Load K chunk in bfloat16
            k_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_d[None, :] * stride_kv_d
            k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

            # Accumulate dot product (tensor core accelerated with bf16)
            qk += tl.dot(q_chunk, tl.trans(k_chunk)).to(tl.float32)

        # Scale and mask
        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, NEG_INF)

        # Online softmax update using exp2 (faster than exp)
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))

        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Convert p to bfloat16 for P @ V multiplication (tensor core accelerated)
        p_bf16 = p.to(tl.bfloat16)

        # V chunk 0
        offs_v = tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        # V chunk 1
        offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        # V chunk 2
        offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        # V chunk 3
        offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        m_i = m_new
        l_i = l_new

    # Compute LSE: lse = m + log(l)
    # Using log2 for consistency with exp2
    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E

    # Check for lonely_q (all invalid tokens -> l_i == 0)
    is_lonely_q = (l_i == 0.0)

    # Compute output scale factor
    # If has attn_sink: scale = 1 / (1 + exp(attn_sink - lse)) = 1 / (l + exp(attn_sink - m))
    # If lonely_q: output = 0, lse = +inf
    if HAS_ATTN_SINK:
        # Load attn_sink values
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        # scale = 1 / (l + exp(attn_sink - m))
        # But we need to be careful with the math here
        # lse = m + log(l)
        # We want: output *= 1 / (1 + exp(attn_sink - lse))
        #        = 1 / (1 + exp(attn_sink - m - log(l)))
        #        = 1 / (1 + exp(attn_sink - m) / l)
        #        = l / (l + exp(attn_sink - m))
        # So final_scale = 1 / (l + exp(attn_sink - m))
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        # Avoid division by zero
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        # Just normalize by l_i
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    # Apply scale to accumulators
    acc_0 = acc_0 * output_scale[:, None]
    acc_1 = acc_1 * output_scale[:, None]
    acc_2 = acc_2 * output_scale[:, None]
    acc_3 = acc_3 * output_scale[:, None]

    # Zero out if lonely_q
    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0)
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1)
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2)
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3)

    # Set lse to +inf for lonely_q
    lse = tl.where(is_lonely_q, POS_INF, lse)

    # Store LSE
    lse_ptrs = LSE + pid_t * stride_lse_t + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_h)

    # Store output
    o_base = Output + pid_t * stride_o_t

    offs_v = tl.arange(0, BLOCK_D)
    o_ptrs = o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d
    tl.store(o_ptrs, acc_0.to(tl.bfloat16), mask=mask_h[:, None])

    offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
    o_ptrs = o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d
    tl.store(o_ptrs, acc_1.to(tl.bfloat16), mask=mask_h[:, None] & (offs_v[None, :] < d_v))

    offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    o_ptrs = o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d
    tl.store(o_ptrs, acc_2.to(tl.bfloat16), mask=mask_h[:, None] & (offs_v[None, :] < d_v))

    offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
    o_ptrs = o_base + offs_h[:, None] * stride_o_h + offs_v[None, :] * stride_o_d
    tl.store(o_ptrs, acc_3.to(tl.bfloat16), mask=mask_h[:, None] & (offs_v[None, :] < d_v))


def _run_triton_attention_fused(q_reshaped, gathered_kv, invalid_mask_reshaped, d_v, sm_scale,
                                 total_tokens, h_q, total_topk, d_qk, attn_sink=None):
    """Run the optimized Triton kernel with fused post-processing."""
    output = torch.empty((total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q_reshaped.device)
    lse = torch.empty((total_tokens, h_q), dtype=torch.float32, device=q_reshaped.device)

    # Fixed optimal configuration
    BLOCK_H = BLOCK_H_OPT  # 16
    BLOCK_N = BLOCK_N_OPT  # 64
    BLOCK_D = BLOCK_D_OPT  # 128

    grid = (total_tokens, triton.cdiv(h_q, BLOCK_H))

    HAS_ATTN_SINK = attn_sink is not None

    # Create a dummy tensor if attn_sink is None
    if attn_sink is None:
        attn_sink_tensor = torch.empty(1, device=q_reshaped.device, dtype=torch.float32)
    else:
        attn_sink_tensor = attn_sink

    _sparse_decode_attn_kernel_fused[grid](
        q_reshaped, gathered_kv, invalid_mask_reshaped,
        attn_sink_tensor,
        output, lse,
        sm_scale,
        total_tokens, h_q, total_topk, d_qk, d_v,
        q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2),
        gathered_kv.stride(0), gathered_kv.stride(1), gathered_kv.stride(2),
        invalid_mask_reshaped.stride(0), invalid_mask_reshaped.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_ATTN_SINK=HAS_ATTN_SINK,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )

    return output, lse


def _run_pytorch_attention(q_reshaped, gathered_kv, invalid_mask_reshaped, d_v, sm_scale, total_tokens, h_q, total_topk):
    """Fallback to PyTorch for very large topk that may cause memory issues in Triton."""
    attn_weight = q_reshaped @ gathered_kv.transpose(-1, -2)
    attn_weight *= sm_scale
    attn_weight[invalid_mask_reshaped.unsqueeze(1).broadcast_to(total_tokens, h_q, total_topk)] = float("-inf")

    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = (attn_weight @ gathered_kv[..., :d_v]).to(torch.bfloat16)

    return output, lse


def triton_sparse_attn_decode(
    q: torch.Tensor,
    kv_scope,
    extra_kv_scope,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized sparse attention decode using Triton with fused post-processing.
    Supports both d_qk=512 and d_qk=576.

    Key optimization: attn_sink scaling and lonely_q masking are fused into the kernel,
    reducing post-processing overhead. Uses bfloat16 for matrix multiplications.
    """
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape

    def process_kv_scope(scope) -> Tuple[torch.Tensor, torch.Tensor]:
        assert scope.indices_in_kvcache is not None
        topk = scope.indices_in_kvcache.size(-1)
        indices_in_kv_cache_fixed = torch.clamp_min(scope.indices_in_kvcache, 0)
        gathered_kv = scope.blocked_k.view(-1, d_qk).index_select(
            0, indices_in_kv_cache_fixed.view(-1)
        ).view(b, s_q, topk, d_qk)
        invalid_mask = scope.indices_in_kvcache == -1
        if scope.topk_length is not None:
            invalid_mask = invalid_mask | (
                torch.arange(0, topk, device=q.device).view(1, 1, topk).broadcast_to(b, s_q, topk) 
                >= scope.topk_length.view(b, 1, 1)
            )
        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_kv_scope(kv_scope)

    if extra_kv_scope is not None:
        gathered_kv1, invalid_mask1 = process_kv_scope(extra_kv_scope)
        gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)
        invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)

    total_topk = gathered_kv.shape[2]
    total_tokens = b * s_q

    # Keep data in bfloat16 for tensor core acceleration
    gathered_kv = gathered_kv.reshape(total_tokens, total_topk, d_qk).to(torch.bfloat16)
    # Handle NaN values
    gathered_kv = torch.where(gathered_kv != gathered_kv, torch.zeros_like(gathered_kv), gathered_kv)

    q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk)
    invalid_mask_reshaped = invalid_mask.reshape(total_tokens, total_topk)

    if not q_reshaped.is_contiguous():
        q_reshaped = q_reshaped.contiguous()
    if not gathered_kv.is_contiguous():
        gathered_kv = gathered_kv.contiguous()
    if not invalid_mask_reshaped.is_contiguous():
        invalid_mask_reshaped = invalid_mask_reshaped.contiguous()

    # Use Triton for topk <= 8192, fallback to PyTorch for larger topk
    USE_TRITON = total_topk <= 8192

    if USE_TRITON:
        # Pass attn_sink to the fused kernel
        output, lse = _run_triton_attention_fused(
            q_reshaped, gathered_kv, invalid_mask_reshaped, 
            d_v, sm_scale, total_tokens, h_q, total_topk, d_qk,
            attn_sink=attn_sink
        )
    else:
        # For PyTorch path, convert to float for computation
        q_reshaped_f32 = q_reshaped.float()
        gathered_kv_f32 = gathered_kv.float()
        output, lse = _run_pytorch_attention(
            q_reshaped_f32, gathered_kv_f32, invalid_mask_reshaped,
            d_v, sm_scale, total_tokens, h_q, total_topk
        )
        # For PyTorch path, still need to do post-processing
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

    # For Triton path, post-processing is already done in the kernel
    output = output.view(b, s_q, h_q, d_v)
    lse = lse.view(b, s_q, h_q)

    return output, lse.transpose(1, 2)
