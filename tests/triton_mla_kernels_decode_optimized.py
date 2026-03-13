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
   - All single scope cases use fused kernel path
   - Dual scope cases use fused dual-scope kernel or separate gather + attention
6. Split-KV optimization for MODEL1 (optional, controlled by USE_SPLITKV env var)

Supports:
- MODEL1 (d_qk=512)
- V3.2 (d_qk=576)

Note: This implementation assumes KV cache is always FP8 quantized.
"""

import os
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

from triton_mla_kernels_decode_splitkv import (
    splitkv_sparse_attn_decode_model1,
)

# Threshold for using fused kernel
# Fused kernel is efficient when total_tokens is small (reduces kernel launch overhead)
# For larger total_tokens, separate gather + attention is more efficient due to better parallelism
FUSED_KERNEL_TOTAL_TOKENS_THRESHOLD = 64  # Only use fused kernel for small batches

# Environment variable to enable Split-KV optimization
# Set USE_SPLITKV=1 to enable Split-KV for MODEL1 single-scope cases
# This is an alternative optimization path that may be useful for certain workloads
USE_SPLITKV = os.environ.get("USE_SPLITKV", "0") == "1"


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
            fused_attn_fn=None,  # V3.2 uses separate gather + attention
            fused_attn_dual_fn=None,  # V3.2 uses separate gather + attention
        )
    else:
        raise ValueError(f"Unsupported d_qk: {d_qk}. Expected {MODEL1_D_QK} or {V32_D_QK}")


def _triton_sparse_attn_decode_optimized(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int, attn_sink: Optional[torch.Tensor],
    d_qk: int, fused_gather_fn, gather_fn, fused_attn_fn, fused_attn_dual_fn,
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

    # Get quantized KV cache (always FP8 quantized)
    kv_quantized_main = kv_scope.blocked_k_quantized
    block_size_main = kv_scope.blocked_k.shape[1]

    # Use fused kernel for all single scope cases (no extra scope)
    # Note: AMD buffer_ops is disabled in fused kernel to avoid int32 overflow with large KV cache
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

    # Use Split-KV for MODEL1 single scope when enabled via environment variable
    # This is an alternative optimization path (currently fused kernel is preferred)
    if USE_SPLITKV and d_qk == MODEL1_D_QK and extra_kv_scope is None:
        q_reshaped = q.reshape(total_tokens, h_q, d_qk)
        if not q_reshaped.is_contiguous():
            q_reshaped = q_reshaped.contiguous()

        indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)
        if not indices_main.is_contiguous():
            indices_main = indices_main.contiguous()

        output, lse = splitkv_sparse_attn_decode_model1(
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

    # Check if chunking needed (for non-fused paths with large buffer requirements)
    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk)
    if len(token_ranges) > 1:
        if d_qk == MODEL1_D_QK:
            from triton_mla_kernels_decode_model1 import triton_sparse_attn_decode_model1
            return triton_sparse_attn_decode_model1(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)
        else:
            from triton_mla_kernels_decode_v32 import triton_sparse_attn_decode_v32
            return triton_sparse_attn_decode_v32(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)

    # Use fused dual-scope kernel when total tokens is small
    if extra_kv_scope is not None and fused_attn_dual_fn is not None and total_tokens <= FUSED_KERNEL_TOTAL_TOKENS_THRESHOLD:
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

    # Fallback: Separate gather + attention path (more efficient for large topk/KV cache)
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
