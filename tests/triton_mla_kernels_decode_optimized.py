"""
Optimized Triton MLA Decode Kernels - Unified Interface

This module provides a unified interface for sparse attention decode that
automatically dispatches to the appropriate implementation based on d_qk:
- MODEL1 (d_qk=512): Uses triton_mla_kernels_decode_model1
- V3.2 (d_qk=576): Uses triton_mla_kernels_decode_v32

For direct access to model-specific implementations, import from:
- triton_mla_kernels_decode_model1: MODEL1-specific kernels
- triton_mla_kernels_decode_v32: V3.2-specific kernels
- triton_mla_kernels_decode_common: Shared utilities and attention kernels
"""

import torch
from typing import Optional, Tuple

# Import MODEL1 implementation
from triton_mla_kernels_decode_model1 import (
    triton_sparse_attn_decode_model1,
    gather_dequant_fp8_model1,
    fused_gather_dequant_fp8_model1,
    MODEL1_D_QK,
)

# Import V32 implementation
from triton_mla_kernels_decode_v32 import (
    triton_sparse_attn_decode_v32,
    gather_dequant_fp8_v32,
    fused_gather_dequant_fp8_v32,
    V32_D_QK,
)

# Import common utilities (for backward compatibility)
from triton_mla_kernels_decode_common import (
    _get_workload_size_category,
    run_unified_attention,
    run_chunked_attention_triton,
    slice_kv_scope_for_tokens,
    compute_token_ranges,
    SlicedKVScope,
)


def triton_sparse_attn_decode(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified sparse attention decode that dispatches based on d_qk.

    Args:
        q: Query tensor of shape (b, s_q, h_q, d_qk)
        kv_scope: Main KV scope with blocked_k, blocked_k_quantized, indices_in_kvcache, topk_length
        extra_kv_scope: Optional extra KV scope (for MODEL1 with extra tokens)
        sm_scale: Softmax scale factor
        d_v: Value dimension (default 512)
        attn_sink: Optional attention sink tensor

    Returns:
        Tuple of (output, lse) tensors
    """
    d_qk = q.shape[-1]

    if d_qk == MODEL1_D_QK:
        return triton_sparse_attn_decode_model1(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
        )
    elif d_qk == V32_D_QK:
        return triton_sparse_attn_decode_v32(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
        )
    else:
        raise ValueError(f"Unsupported d_qk: {d_qk}. Expected {MODEL1_D_QK} (MODEL1) or {V32_D_QK} (V3.2)")


# Backward compatibility aliases
_SlicedKVScope = SlicedKVScope
_slice_kv_scope_for_tokens = slice_kv_scope_for_tokens
_compute_token_ranges = compute_token_ranges
_run_unified_attention = run_unified_attention
_run_chunked_attention_triton = run_chunked_attention_triton
