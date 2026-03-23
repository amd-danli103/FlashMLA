"""
Optimized Triton MLA Decode Kernels - Unified Interface

This module provides optimized sparse attention decode that reduces Python
overhead for small workloads.

Optimizations applied:
1. Inlined attention kernel call to reduce function call overhead
2. Pre-computed strides to reduce tensor metadata operations
3. Avoided redundant tensor operations
4. Unified implementation for both MODEL1 and V3.2
5. Fused gather+dequant+attention for MODEL1 (single and dual scope)
   - Single scope cases: use fused for small topk, 2-phase for large topk
   - Dual scope cases: use fused dual-scope kernel or separate gather + attention
6. Split-K optimization for small batch sizes to increase GPU parallelism
7. Extended Split-K for h_q=64 + large topk + medium batch sizes (tokens<=128)

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
    fused_gather_attn_decode_model1_dual_scope,
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
            fused_attn_dual_fn=fused_gather_attn_decode_model1_dual_scope,
        )
    elif d_qk == V32_D_QK:
        return _triton_sparse_attn_decode_optimized(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink,
            d_qk=V32_D_QK,
            fused_gather_fn=fused_gather_dequant_fp8_v32,
            gather_fn=gather_dequant_fp8_v32,
            fused_attn_fn=None,
            fused_attn_dual_fn=None,
        )
    else:
        raise ValueError(f"Unsupported d_qk: {d_qk}. Expected {MODEL1_D_QK} or {V32_D_QK}")


def _should_use_fused_single_scope(topk: int, h_q: int, total_tokens: int) -> bool:
    """Determine whether to use fused kernel for single-scope cases."""
    if topk >= 8192:
        return False
    return True


def _should_use_fused_dual_scope(total_tokens: int, h_q: int, total_topk: int) -> bool:
    """Determine whether to use fused kernel for dual-scope cases.

    For small batch sizes, we want to use the splitk version inside
    fused_gather_attn_decode_model1_dual_scope for better parallelism.

    Extended to also cover h_q=64 + large topk + medium batch sizes (tokens<=128),
    which benefit from fused+splitk (~13% improvement for bs=64).
    """
    # Always use fused for very small batches - the fused function will
    # internally decide whether to use splitk
    if total_tokens <= 4:
        return True
    # For h_q=64 with small topk, fused is efficient up to total_tokens=256
    if h_q <= 64 and total_topk <= 800:
        return total_tokens <= 256
    # NEW: For h_q=64 with large topk (>=1024), fused+splitk is efficient
    # only up to total_tokens=128 (tested: ~13% improvement for CONFIG3 bs=64)
    if h_q <= 64 and total_topk >= 1024:
        return total_tokens <= 128
    # For other cases, fall back to 2-phase
    return False


def _triton_sparse_attn_decode_optimized(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int, attn_sink: Optional[torch.Tensor],
    d_qk: int, fused_gather_fn, gather_fn, fused_attn_fn, fused_attn_dual_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized sparse attention decode - unified implementation."""
    b, s_q, h_q, _ = q.shape
    total_tokens = b * s_q
    device = q.device

    topk_main = kv_scope.indices_in_kvcache.shape[-1]
    kv_quantized_main = kv_scope.blocked_k_quantized
    block_size_main = kv_scope.blocked_k.shape[1]

    # Single scope case
    if extra_kv_scope is None:
        if fused_attn_fn is not None and _should_use_fused_single_scope(topk_main, h_q, total_tokens):
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
        else:
            if d_qk == MODEL1_D_QK:
                from triton_mla_kernels_decode_model1 import triton_sparse_attn_decode_model1
                return triton_sparse_attn_decode_model1(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)
            else:
                from triton_mla_kernels_decode_v32 import triton_sparse_attn_decode_v32
                return triton_sparse_attn_decode_v32(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)

    # Dual scope case
    topk_extra = extra_kv_scope.indices_in_kvcache.shape[-1]
    total_topk = topk_main + topk_extra

    # Check if chunking needed
    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk)
    if len(token_ranges) > 1:
        if d_qk == MODEL1_D_QK:
            from triton_mla_kernels_decode_model1 import triton_sparse_attn_decode_model1
            return triton_sparse_attn_decode_model1(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)
        else:
            from triton_mla_kernels_decode_v32 import triton_sparse_attn_decode_v32
            return triton_sparse_attn_decode_v32(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)

    # Use fused dual-scope kernel for MODEL1 dual scope cases when beneficial
    if fused_attn_dual_fn is not None and _should_use_fused_dual_scope(total_tokens, h_q, total_topk):
        q_reshaped = q.reshape(total_tokens, h_q, d_qk)
        if not q_reshaped.is_contiguous():
            q_reshaped = q_reshaped.contiguous()

        indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)
        if not indices_main.is_contiguous():
            indices_main = indices_main.contiguous()

        block_size_extra = extra_kv_scope.blocked_k.shape[1]
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)
        if not indices_extra.is_contiguous():
            indices_extra = indices_extra.contiguous()

        output, lse = fused_attn_dual_fn(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            extra_kv_scope.blocked_k_quantized,
            indices_extra,
            block_size_extra,
            sm_scale,
            topk_length_main=kv_scope.topk_length,
            topk_length_extra=extra_kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
        )
        return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)

    # Fallback: Separate gather + attention path
    gathered_kv = torch.empty(total_tokens, total_topk, d_qk, dtype=torch.bfloat16, device=device)
    invalid_mask = torch.empty(total_tokens, total_topk, dtype=torch.bool, device=device)
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)

    block_size_extra = extra_kv_scope.blocked_k.shape[1]
    indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)
    fused_gather_fn(
        kv_quantized_main, indices_main, block_size_main, kv_scope.topk_length,
        extra_kv_scope.blocked_k_quantized, indices_extra, block_size_extra, extra_kv_scope.topk_length,
        gathered_kv, invalid_mask, s_q)

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
