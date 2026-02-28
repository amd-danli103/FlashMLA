"""
Triton MLA Decode Kernels for V3.2 (d_qk=576).

This module contains V3.2-specific gather+dequant kernels and the main
sparse attention decode entry point for V3.2.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

from triton_mla_kernels_decode_common import (
    _get_workload_size_category,
    run_unified_attention,
    run_chunked_attention_triton,
    slice_kv_scope_for_tokens,
    compute_token_ranges,
)


# Constants for V32 layout
V32_D_QK = 576
V32_D_NOPE = 512
V32_D_ROPE = 64
V32_TILE_SIZE = 128
V32_NUM_TILES = 4
V32_BYTES_PER_TOKEN = 656


# ============================================================================
# V32 Gather+Dequant Kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_TK': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_TK': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_TK': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_TK': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_TK': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TK': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_TK': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TK': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TK': 256}, num_warps=16, num_stages=1),
    ],
    key=['total_tokens', 'topk', 'workload_size_cat'],
)
@triton.jit
def _gather_dequant_v32_kernel(
    KV_Cache,
    Indices,
    TopkLength,
    OutputKV,
    OutputMask,
    total_tokens,
    topk,
    num_blocks,
    block_size,
    workload_size_cat,
    k_offset,
    s_q,
    stride_kv_block,
    stride_kv_token,
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_k, stride_out_d,
    stride_mask_t, stride_mask_k,
    BLOCK_TK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
):
    """Unified gather + dequant + mask kernel for V32 layout."""
    pid = tl.program_id(0)
    num_tk = total_tokens * topk

    offs_tk = pid * BLOCK_TK + tl.arange(0, BLOCK_TK)
    mask_tk = offs_tk < num_tk

    t_idx = offs_tk // topk
    k_idx = offs_tk % topk

    idx_ptrs = Indices + t_idx * stride_idx_t + k_idx * stride_idx_k
    indices = tl.load(idx_ptrs, mask=mask_tk, other=-1)

    is_invalid = indices == -1

    if HAS_TOPK_LENGTH:
        batch_idx = t_idx // s_q
        topk_len = tl.load(TopkLength + batch_idx, mask=mask_tk, other=topk)
        is_invalid = is_invalid | (k_idx >= topk_len)

    mask_out_ptrs = OutputMask + t_idx * stride_mask_t + (k_idx + k_offset) * stride_mask_k
    tl.store(mask_out_ptrs, is_invalid, mask=mask_tk)

    valid_mask = mask_tk & ~is_invalid
    indices_clamped = tl.maximum(indices, 0)

    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size

    block_idx_64 = block_idx.to(tl.int64)
    offset_in_block_64 = offset_in_block.to(tl.int64)

    kv_base_ptrs = KV_Cache + block_idx_64 * stride_kv_block + offset_in_block_64 * stride_kv_token

    t_idx_64 = t_idx.to(tl.int64)
    k_idx_64 = k_idx.to(tl.int64)
    stride_out_t_64 = tl.cast(stride_out_t, tl.int64)
    stride_out_k_64 = tl.cast(stride_out_k, tl.int64)
    out_base_ptrs = OutputKV + t_idx_64 * stride_out_t_64 + (k_idx_64 + k_offset) * stride_out_k_64

    for tile_idx in range(NUM_TILES):
        tile_start = tile_idx * TILE_SIZE

        scale_byte_offset = D_NOPE + tile_idx * 4
        scale_b0_ptrs = kv_base_ptrs + scale_byte_offset
        scale_b1_ptrs = kv_base_ptrs + scale_byte_offset + 1
        scale_b2_ptrs = kv_base_ptrs + scale_byte_offset + 2
        scale_b3_ptrs = kv_base_ptrs + scale_byte_offset + 3

        scale_b0 = tl.load(scale_b0_ptrs, mask=valid_mask, other=0)
        scale_b1 = tl.load(scale_b1_ptrs, mask=valid_mask, other=0)
        scale_b2 = tl.load(scale_b2_ptrs, mask=valid_mask, other=0)
        scale_b3 = tl.load(scale_b3_ptrs, mask=valid_mask, other=0)

        scale_uint32 = (scale_b0.to(tl.uint32) |
                       (scale_b1.to(tl.uint32) << 8) |
                       (scale_b2.to(tl.uint32) << 16) |
                       (scale_b3.to(tl.uint32) << 24))
        scale_f32 = scale_uint32.to(tl.float32, bitcast=True)

        for chunk in range(2):
            chunk_start = tile_start + chunk * 64
            offs_d = tl.arange(0, 64)

            nope_ptrs = kv_base_ptrs[:, None] + chunk_start + offs_d[None, :]
            nope_uint8 = tl.load(nope_ptrs, mask=valid_mask[:, None], other=0)

            nope_fp8 = nope_uint8.to(tl.float8e4nv, bitcast=True)
            nope_f32 = nope_fp8.to(tl.float32)

            dequant = nope_f32 * scale_f32[:, None]
            dequant = tl.where(dequant != dequant, 0.0, dequant)
            dequant = tl.maximum(tl.minimum(dequant, 65504.0), -65504.0)
            dequant = tl.where(is_invalid[:, None], 0.0, dequant)

            out_ptrs = out_base_ptrs[:, None] + (chunk_start + offs_d[None, :]) * stride_out_d
            tl.store(out_ptrs, dequant.to(tl.bfloat16), mask=mask_tk[:, None])

    rope_byte_offset = D_NOPE + NUM_TILES * 4
    offs_rope = tl.arange(0, D_ROPE)

    rope_lo_ptrs = kv_base_ptrs[:, None] + rope_byte_offset + offs_rope[None, :] * 2
    rope_hi_ptrs = kv_base_ptrs[:, None] + rope_byte_offset + offs_rope[None, :] * 2 + 1

    rope_lo = tl.load(rope_lo_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)
    rope_hi = tl.load(rope_hi_ptrs, mask=valid_mask[:, None], other=0).to(tl.uint16)

    rope_uint16 = rope_lo | (rope_hi << 8)
    rope_bf16 = rope_uint16.to(tl.bfloat16, bitcast=True)
    rope_bf16 = tl.where(rope_bf16 != rope_bf16, 0.0, rope_bf16)
    rope_bf16 = tl.maximum(tl.minimum(rope_bf16, 65504.0), -65504.0)
    rope_bf16 = tl.where(is_invalid[:, None], 0.0, rope_bf16)

    out_ptrs = out_base_ptrs[:, None] + (D_NOPE + offs_rope[None, :]) * stride_out_d
    tl.store(out_ptrs, rope_bf16.to(tl.bfloat16), mask=mask_tk[:, None])


# ============================================================================
# V32 Wrapper Functions
# ============================================================================

def gather_dequant_fp8_v32(
    kv_cache_quantized: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    output_kv: torch.Tensor,
    output_mask: torch.Tensor,
    k_offset: int = 0,
    topk_length: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> bool:
    """Unified V32 gather+dequant with optional topk_length mask."""
    total_tokens, topk = indices.shape

    num_blocks = kv_cache_quantized.shape[0]

    kv_uint8 = kv_cache_quantized.view(torch.uint8)

    stride_kv_block = kv_uint8.stride(0)
    stride_kv_token = kv_uint8.stride(1)
    workload_size_cat = _get_workload_size_category(total_tokens, topk)

    grid = lambda meta: (triton.cdiv(total_tokens * topk, meta['BLOCK_TK']),)

    topk_length_tensor = topk_length if topk_length is not None else output_mask[:1, 0]
    has_topk_length = topk_length is not None

    _gather_dequant_v32_kernel[grid](
        kv_uint8,
        indices,
        topk_length_tensor,
        output_kv,
        output_mask,
        total_tokens,
        topk,
        num_blocks,
        block_size,
        workload_size_cat,
        k_offset,
        s_q,
        stride_kv_block, stride_kv_token,
        indices.stride(0), indices.stride(1),
        output_kv.stride(0), output_kv.stride(1), output_kv.stride(2),
        output_mask.stride(0), output_mask.stride(1),
        D_NOPE=V32_D_NOPE,
        D_ROPE=V32_D_ROPE,
        TILE_SIZE=V32_TILE_SIZE,
        NUM_TILES=V32_NUM_TILES,
        HAS_TOPK_LENGTH=has_topk_length,
    )
    return True


def fused_gather_dequant_fp8_v32(
    kv_cache_main, indices_main, block_size_main, topk_length_main,
    kv_cache_extra, indices_extra, block_size_extra, topk_length_extra,
    output_kv, output_mask, s_q=1,
):
    """Fused V32 gather - optimized wrapper for both scopes."""
    total_tokens, topk_main = indices_main.shape
    topk_extra = indices_extra.shape[1]

    kv_uint8_main = kv_cache_main.view(torch.uint8)
    stride_kv_block_main = kv_uint8_main.stride(0)
    stride_kv_token_main = kv_uint8_main.stride(1)
    num_blocks_main = kv_cache_main.shape[0]

    kv_uint8_extra = kv_cache_extra.view(torch.uint8)
    stride_kv_block_extra = kv_uint8_extra.stride(0)
    stride_kv_token_extra = kv_uint8_extra.stride(1)
    num_blocks_extra = kv_cache_extra.shape[0]

    topk_length_main_tensor = topk_length_main if topk_length_main is not None else output_mask[:1, 0]
    topk_length_extra_tensor = topk_length_extra if topk_length_extra is not None else output_mask[:1, 0]

    workload_main = _get_workload_size_category(total_tokens, topk_main)
    workload_extra = _get_workload_size_category(total_tokens, topk_extra)

    grid_main = lambda meta: (triton.cdiv(total_tokens * topk_main, meta['BLOCK_TK']),)
    _gather_dequant_v32_kernel[grid_main](
        kv_uint8_main, indices_main, topk_length_main_tensor,
        output_kv, output_mask,
        total_tokens, topk_main, num_blocks_main, block_size_main,
        workload_main, 0, s_q,
        stride_kv_block_main, stride_kv_token_main,
        indices_main.stride(0), indices_main.stride(1),
        output_kv.stride(0), output_kv.stride(1), output_kv.stride(2),
        output_mask.stride(0), output_mask.stride(1),
        D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
        TILE_SIZE=V32_TILE_SIZE, NUM_TILES=V32_NUM_TILES,
        HAS_TOPK_LENGTH=topk_length_main is not None,
    )

    grid_extra = lambda meta: (triton.cdiv(total_tokens * topk_extra, meta['BLOCK_TK']),)
    _gather_dequant_v32_kernel[grid_extra](
        kv_uint8_extra, indices_extra, topk_length_extra_tensor,
        output_kv, output_mask,
        total_tokens, topk_extra, num_blocks_extra, block_size_extra,
        workload_extra, topk_main, s_q,
        stride_kv_block_extra, stride_kv_token_extra,
        indices_extra.stride(0), indices_extra.stride(1),
        output_kv.stride(0), output_kv.stride(1), output_kv.stride(2),
        output_mask.stride(0), output_mask.stride(1),
        D_NOPE=V32_D_NOPE, D_ROPE=V32_D_ROPE,
        TILE_SIZE=V32_TILE_SIZE, NUM_TILES=V32_NUM_TILES,
        HAS_TOPK_LENGTH=topk_length_extra is not None,
    )
    return True


# ============================================================================
# V32 Main Entry Point
# ============================================================================

def triton_sparse_attn_decode_v32(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention decode for V3.2 (d_qk=576)."""
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == V32_D_QK, f"Expected d_qk={V32_D_QK} for V3.2, got {d_qk}"
    total_tokens = b * s_q

    topk_main = kv_scope.indices_in_kvcache.size(-1)
    topk_extra = extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0
    total_topk = topk_main + topk_extra

    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk)

    if len(token_ranges) == 1:
        return _triton_sparse_attn_decode_v32_impl(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
        )

    outputs = []
    lses = []

    for start_t, end_t in token_ranges:
        chunk_tokens = end_t - start_t

        q_chunk = q.reshape(total_tokens, h_q, d_qk)[start_t:end_t]
        q_input = q_chunk.reshape(chunk_tokens, 1, h_q, d_qk)
        chunk_kv_scope = slice_kv_scope_for_tokens(kv_scope, start_t, end_t, s_q)
        chunk_extra_kv_scope = slice_kv_scope_for_tokens(extra_kv_scope, start_t, end_t, s_q)

        chunk_out, chunk_lse = _triton_sparse_attn_decode_v32_impl(
            q_input, chunk_kv_scope, chunk_extra_kv_scope, sm_scale, d_v, attn_sink
        )

        outputs.append(chunk_out.reshape(chunk_tokens, h_q, d_v))
        lses.append(chunk_lse.reshape(chunk_tokens, h_q))

    output = torch.cat(outputs, dim=0).reshape(b, s_q, h_q, d_v)
    lse = torch.cat(lses, dim=0).reshape(b, s_q, h_q).transpose(1, 2)

    return output, lse


def _triton_sparse_attn_decode_v32_impl(
    q: torch.Tensor, kv_scope, extra_kv_scope, sm_scale: float,
    d_v: int = 512, attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal implementation of sparse attention decode for V3.2."""
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q

    topk_main = kv_scope.indices_in_kvcache.size(-1)
    topk_extra = extra_kv_scope.indices_in_kvcache.size(-1) if extra_kv_scope is not None else 0
    total_topk = topk_main + topk_extra

    gathered_kv = torch.empty(total_tokens, total_topk, d_qk, dtype=torch.bfloat16, device=q.device)
    invalid_mask = torch.empty(total_tokens, total_topk, dtype=torch.bool, device=q.device)

    block_size_main = kv_scope.blocked_k.shape[1]
    indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)

    use_fused = (kv_scope.blocked_k_quantized is not None and
                 extra_kv_scope is not None and
                 extra_kv_scope.blocked_k_quantized is not None)

    if use_fused:
        block_size_extra = extra_kv_scope.blocked_k.shape[1]
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)
        fused_gather_dequant_fp8_v32(
            kv_scope.blocked_k_quantized, indices_main, block_size_main, kv_scope.topk_length,
            extra_kv_scope.blocked_k_quantized, indices_extra, block_size_extra, extra_kv_scope.topk_length,
            gathered_kv, invalid_mask, s_q)
    else:
        if kv_scope.blocked_k_quantized is not None:
            gather_dequant_fp8_v32(
                kv_scope.blocked_k_quantized, indices_main, block_size_main,
                gathered_kv, invalid_mask, 0,
                kv_scope.topk_length, s_q)
        else:
            scope_invalid_mask = indices_main == -1
            if kv_scope.topk_length is not None:
                topk_length_mask = (
                    torch.arange(0, topk_main, device=q.device).view(1, 1, topk_main).broadcast_to(b, s_q, topk_main)
                    >= kv_scope.topk_length.view(b, 1, 1)
                ).reshape(total_tokens, topk_main)
                scope_invalid_mask = scope_invalid_mask | topk_length_mask
            invalid_mask[:, :topk_main] = scope_invalid_mask
            indices_clamped = torch.clamp(indices_main, min=0)
            kv_gathered = kv_scope.blocked_k.view(-1, d_qk).index_select(0, indices_clamped.view(-1)).view(total_tokens, topk_main, d_qk).to(torch.bfloat16)
            kv_gathered[scope_invalid_mask] = 0
            gathered_kv[:, :topk_main, :] = kv_gathered

        if extra_kv_scope is not None:
            block_size_extra = extra_kv_scope.blocked_k.shape[1]
            indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)

            if extra_kv_scope.blocked_k_quantized is not None:
                gather_dequant_fp8_v32(
                    extra_kv_scope.blocked_k_quantized, indices_extra, block_size_extra,
                    gathered_kv, invalid_mask, topk_main,
                    extra_kv_scope.topk_length, s_q)
            else:
                scope_invalid_mask = indices_extra == -1
                if extra_kv_scope.topk_length is not None:
                    topk_length_mask = (
                        torch.arange(0, topk_extra, device=q.device).view(1, 1, topk_extra).broadcast_to(b, s_q, topk_extra)
                        >= extra_kv_scope.topk_length.view(b, 1, 1)
                    ).reshape(total_tokens, topk_extra)
                    scope_invalid_mask = scope_invalid_mask | topk_length_mask
                invalid_mask[:, topk_main:] = scope_invalid_mask
                indices_clamped = torch.clamp(indices_extra, min=0)
                kv_gathered = extra_kv_scope.blocked_k.view(-1, d_qk).index_select(0, indices_clamped.view(-1)).view(total_tokens, topk_extra, d_qk).to(torch.bfloat16)
                kv_gathered[scope_invalid_mask] = 0
                gathered_kv[:, topk_main:, :] = kv_gathered

    q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk)

    if not q_reshaped.is_contiguous():
        q_reshaped = q_reshaped.contiguous()

    if total_topk <= 65536:
        output, lse = run_unified_attention(
            q_reshaped, gathered_kv, invalid_mask,
            d_v, sm_scale, total_tokens, h_q, total_topk, d_qk,
            attn_sink=attn_sink
        )
    else:
        output, lse = run_chunked_attention_triton(
            q_reshaped, gathered_kv, invalid_mask,
            d_v, sm_scale, total_tokens, h_q, total_topk, d_qk,
            attn_sink=attn_sink, chunk_size=32768
        )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)
