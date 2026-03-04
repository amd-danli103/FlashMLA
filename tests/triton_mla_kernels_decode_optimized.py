"""
Optimized Triton MLA Decode Kernels - Unified Interface

This module provides optimized sparse attention decode that reduces Python
overhead for small workloads.

Optimizations applied:
1. Inlined attention kernel call to reduce function call overhead
2. Pre-computed strides to reduce tensor metadata operations
3. Avoided redundant tensor operations
4. Unified implementation for both MODEL1 and V3.2
5. Fused gather+dequant+attention for MODEL1 without extra scope

Supports:
- MODEL1 (d_qk=512)
- V3.2 (d_qk=576)

Note: This implementation assumes KV cache is always FP8 quantized.
"""

import torch
import triton
from typing import Optional, Tuple

from triton_mla_kernels_decode_common import (
    compute_token_ranges,
    _unified_sparse_decode_kernel,
    _unified_sparse_decode_kernel_fixed,
)

from triton_mla_kernels_decode_model1 import (
    fused_gather_dequant_fp8_model1,
    gather_dequant_fp8_model1,
    MODEL1_D_QK,
)

from triton_mla_kernels_decode_v32 import (
    fused_gather_dequant_fp8_v32,
    gather_dequant_fp8_v32,
    V32_D_QK,
)

from triton_mla_kernels_decode_fused import (
    fused_gather_attn_decode_model1,
)


def triton_sparse_attn_decode(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified optimized sparse attention decode that dispatches based on d_qk."""
    d_qk = q.shape[-1]

    if d_qk == MODEL1_D_QK:
        return _triton_sparse_attn_decode_optimized(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink,
            d_qk=MODEL1_D_QK,
            fused_gather_fn=fused_gather_dequant_fp8_model1,
            gather_fn=gather_dequant_fp8_model1,
            fused_attn_fn=fused_gather_attn_decode_model1,
        )
    elif d_qk == V32_D_QK:
        return _triton_sparse_attn_decode_optimized(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink,
            d_qk=V32_D_QK,
            fused_gather_fn=fused_gather_dequant_fp8_v32,
            gather_fn=gather_dequant_fp8_v32,
            fused_attn_fn=None,  # V3.2 uses separate gather + attention
        )
    else:
        raise ValueError(f"Unsupported d_qk: {d_qk}. Expected {MODEL1_D_QK} or {V32_D_QK}")


def _triton_sparse_attn_decode_optimized(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int, attn_sink: Optional[torch.Tensor],
    d_qk: int, fused_gather_fn, gather_fn, fused_attn_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized sparse attention decode - unified implementation.

    Assumes KV cache is always FP8 quantized (blocked_k_quantized is not None).
    """
    b, s_q, h_q, _ = q.shape
    total_tokens = b * s_q
    device = q.device

    topk_main = kv_scope.indices_in_kvcache.size(-1)
    topk_extra = extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0
    total_topk = topk_main + topk_extra

    # Check if chunking needed (rare for small workloads)
    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk)
    if len(token_ranges) > 1:
        if d_qk == MODEL1_D_QK:
            from triton_mla_kernels_decode_model1 import triton_sparse_attn_decode_model1
            return triton_sparse_attn_decode_model1(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)
        else:
            from triton_mla_kernels_decode_v32 import triton_sparse_attn_decode_v32
            return triton_sparse_attn_decode_v32(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)

    # Get quantized KV cache (always FP8 quantized)
    kv_quantized_main = kv_scope.blocked_k_quantized
    block_size_main = kv_scope.blocked_k.shape[1]

    # Use fused kernel when: no extra scope and fused_attn_fn exists
    if extra_kv_scope is None and fused_attn_fn is not None:
        q_reshaped = q.reshape(total_tokens, h_q, d_qk)
        if not q_reshaped.is_contiguous():
            q_reshaped = q_reshaped.contiguous()

        indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)
        if not indices_main.is_contiguous():
            indices_main = indices_main.contiguous()

        output, lse = fused_attn_fn(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            sm_scale,
            topk_length=kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
        )
        return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)

    # Separate gather + attention path
    gathered_kv = torch.empty(total_tokens, total_topk, d_qk, dtype=torch.bfloat16, device=device)
    invalid_mask = torch.empty(total_tokens, total_topk, dtype=torch.bool, device=device)
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)

    if extra_kv_scope is not None:
        # Fused gather for both main and extra scope
        block_size_extra = extra_kv_scope.blocked_k.shape[1]
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)
        fused_gather_fn(
            kv_quantized_main, indices_main, block_size_main, kv_scope.topk_length,
            extra_kv_scope.blocked_k_quantized, indices_extra, block_size_extra, extra_kv_scope.topk_length,
            gathered_kv, invalid_mask, s_q)
    else:
        # Single gather for main scope only
        gather_fn(kv_quantized_main, indices_main, block_size_main,
                  gathered_kv, invalid_mask, 0, kv_scope.topk_length, s_q)

    # Prepare Q tensor
    if q.dtype == torch.bfloat16 and q.is_contiguous():
        q_reshaped = q.view(total_tokens, h_q, d_qk)
    else:
        q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk)
        if not q_reshaped.is_contiguous():
            q_reshaped = q_reshaped.contiguous()

    stride_q_t, stride_q_h, stride_q_d = q_reshaped.stride()
    stride_kv_t, stride_kv_k, stride_kv_d = gathered_kv.stride()
    stride_mask_t, stride_mask_k = invalid_mask.stride()
    stride_o_t, stride_o_h, stride_o_d = output.stride()
    stride_lse_t, stride_lse_h = lse.stride()

    HAS_ATTN_SINK = attn_sink is not None
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1]

    if total_tokens * h_q < 1024 and total_topk <= 1024:
        grid = (total_tokens, triton.cdiv(h_q, 16))
        _unified_sparse_decode_kernel_fixed[grid](
            q_reshaped, gathered_kv, invalid_mask, attn_sink_tensor,
            output, lse,
            sm_scale, total_tokens, h_q, total_topk, d_qk, d_v,
            stride_q_t, stride_q_h, stride_q_d,
            stride_kv_t, stride_kv_k, stride_kv_d,
            stride_mask_t, stride_mask_k,
            stride_o_t, stride_o_h, stride_o_d,
            stride_lse_t, stride_lse_h,
            HAS_ATTN_SINK=HAS_ATTN_SINK,
            BLOCK_H=16, BLOCK_N=64, BLOCK_D=128,
            num_warps=4, num_stages=1,
        )
    else:
        grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))
        _unified_sparse_decode_kernel[grid](
            q_reshaped, gathered_kv, invalid_mask, attn_sink_tensor,
            output, lse,
            sm_scale, total_tokens, h_q, total_topk, d_qk, d_v,
            stride_q_t, stride_q_h, stride_q_d,
            stride_kv_t, stride_kv_k, stride_kv_d,
            stride_mask_t, stride_mask_k,
            stride_o_t, stride_o_h, stride_o_d,
            stride_lse_t, stride_lse_h,
            HAS_ATTN_SINK=HAS_ATTN_SINK,
        )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)
