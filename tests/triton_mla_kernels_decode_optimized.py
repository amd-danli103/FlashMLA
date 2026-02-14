import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

# Constants
LOG2E = 1.4426950408889634

# Optimal block sizes
BLOCK_H_OPT = 16
BLOCK_N_OPT = 64
BLOCK_D_OPT = 128


def gather_dequant_fp8_v32(
    kv_cache_quantized: torch.Tensor,  # [num_blocks, block_size, 1, bytes_per_token]
    indices: torch.Tensor,              # [total_tokens, topk] - flattened indices
    invalid_mask: torch.Tensor,         # [total_tokens, topk]
    block_size: int,
) -> torch.Tensor:
    """
    Gather and dequantize FP8 KV cache to BF16 for V32 layout (d_qk=576).
    """
    d_qk = 576
    d_nope = 512
    d_rope = 64
    tile_size = 128
    num_tiles = 4
    bytes_per_token = 656

    total_tokens, topk = indices.shape
    device = kv_cache_quantized.device

    # Reshape quantized cache to [num_blocks * block_size, bytes_per_token]
    num_blocks = kv_cache_quantized.shape[0]
    kv_flat = kv_cache_quantized.reshape(num_blocks * block_size, bytes_per_token)

    # Clamp invalid indices to 0
    indices_clamped = torch.clamp(indices, min=0)

    # Gather raw bytes: [total_tokens, topk, bytes_per_token]
    gathered_bytes = kv_flat[indices_clamped.reshape(-1)].reshape(total_tokens, topk, bytes_per_token)

    # Extract FP8 nope part
    nope_bytes = gathered_bytes[..., :d_nope].contiguous()
    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)

    # Extract scales (4 float32 values)
    scale_bytes = gathered_bytes[..., d_nope:d_nope + num_tiles * 4].contiguous()
    scales = scale_bytes.view(torch.float32)

    # Extract BF16 rope part
    rope_bytes = gathered_bytes[..., d_nope + num_tiles * 4:].contiguous()
    rope_bf16 = rope_bytes.view(torch.bfloat16)

    # Dequantize NOPE: fp8 * scale -> bf16
    output = torch.empty(total_tokens, topk, d_qk, dtype=torch.bfloat16, device=device)

    for tile_idx in range(num_tiles):
        tile_start = tile_idx * tile_size
        tile_end = tile_start + tile_size
        cur_nope = nope_fp8[..., tile_start:tile_end].to(torch.float32)
        cur_scale = scales[..., tile_idx:tile_idx+1]
        output[..., tile_start:tile_end] = (cur_nope * cur_scale).to(torch.bfloat16)

    # Copy ROPE part
    output[..., d_nope:] = rope_bf16

    # Zero out invalid positions
    output[invalid_mask] = 0

    return output


def gather_dequant_fp8_model1(
    kv_cache_quantized: torch.Tensor,  # [num_blocks, block_size, 1, bytes_per_token]
    indices: torch.Tensor,              # [total_tokens, topk] - flattened indices
    invalid_mask: torch.Tensor,         # [total_tokens, topk]
    block_size: int,
) -> torch.Tensor:
    """
    Gather and dequantize FP8 KV cache to BF16 for MODEL1 layout (d_qk=512).
    Optimized vectorized implementation.
    """
    d_qk = 512
    d_nope = 448
    d_rope = 64
    tile_size = 64
    num_tiles = 7
    bytes_per_token_data = d_nope + d_rope * 2  # 576
    bytes_per_token_scale = 8

    total_tokens, topk = indices.shape
    device = kv_cache_quantized.device
    num_blocks = kv_cache_quantized.shape[0]

    # Clamp invalid indices to 0
    indices_clamped = torch.clamp(indices, min=0)

    # Compute block index and offset within block
    block_idx = indices_clamped // block_size  # [total_tokens, topk]
    offset_in_block = indices_clamped % block_size  # [total_tokens, topk]

    # Reshape quantized cache to access per-block data as bytes
    # Shape: [num_blocks, total_bytes_per_block]
    kv_cache_bytes = kv_cache_quantized.view(torch.uint8).reshape(num_blocks, -1)
    total_bytes_per_block = kv_cache_bytes.shape[1]

    # Calculate offsets for data and scales
    # Data offset for token i in block: i * 576
    # Scale offset for token i in block: block_size * 576 + i * 8
    scale_base_offset = block_size * bytes_per_token_data

    # Flatten indices for efficient gathering
    flat_block_idx = block_idx.reshape(-1)  # [total_tokens * topk]
    flat_offset = offset_in_block.reshape(-1)  # [total_tokens * topk]
    n_elements = flat_block_idx.shape[0]

    # Create output tensor
    output = torch.empty(total_tokens, topk, d_qk, dtype=torch.bfloat16, device=device)

    # Gather nope data (448 bytes) - vectorized
    # Create index tensor for all bytes of nope
    nope_byte_indices = flat_offset[:, None] * bytes_per_token_data + torch.arange(d_nope, device=device)[None, :]  # [n_elements, d_nope]
    gathered_nope_bytes = kv_cache_bytes[flat_block_idx[:, None].expand(-1, d_nope), nope_byte_indices]  # [n_elements, d_nope]
    gathered_nope = gathered_nope_bytes.view(total_tokens, topk, d_nope).view(torch.float8_e4m3fn)

    # Gather rope data (128 bytes = 64 bf16) - vectorized
    rope_byte_indices = flat_offset[:, None] * bytes_per_token_data + d_nope + torch.arange(d_rope * 2, device=device)[None, :]
    gathered_rope_bytes = kv_cache_bytes[flat_block_idx[:, None].expand(-1, d_rope * 2), rope_byte_indices]
    gathered_rope = gathered_rope_bytes.view(total_tokens, topk, d_rope * 2).contiguous().view(torch.bfloat16)

    # Gather scale data (7 bytes) - vectorized
    scale_byte_indices = scale_base_offset + flat_offset[:, None] * bytes_per_token_scale + torch.arange(num_tiles, device=device)[None, :]
    gathered_scale_bytes = kv_cache_bytes[flat_block_idx[:, None].expand(-1, num_tiles), scale_byte_indices]
    gathered_scales = gathered_scale_bytes.view(total_tokens, topk, num_tiles).view(torch.float8_e8m0fnu)

    # Dequantize NOPE: fp8 * scale -> bf16
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * tile_size
        tile_end = min(tile_start + tile_size, d_nope)

        cur_nope = gathered_nope[..., tile_start:tile_end].to(torch.bfloat16)
        cur_scale = gathered_scales[..., tile_idx:tile_idx+1].to(torch.bfloat16)
        output[..., tile_start:tile_end] = cur_nope * cur_scale

    # Copy ROPE part
    output[..., d_nope:] = gathered_rope

    # Zero out invalid positions
    output[invalid_mask] = 0

    return output


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

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    q_base = Q + pid_t * stride_q_t
    kv_base = KV + pid_t * stride_kv_t
    mask_base = InvalidMask + pid_t * stride_mask_t

    for n_start in range(0, topk, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk

        mask_ptrs = mask_base + offs_n * stride_mask_k
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True)
        valid = mask_n & ~invalid

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, d_qk, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d_qk

            q_ptrs = q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
            q_chunk = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

            k_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_d[None, :] * stride_kv_d
            k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.bfloat16)

            qk += tl.dot(q_chunk, tl.trans(k_chunk)).to(tl.float32)

        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, NEG_INF)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))

        l_new = alpha * l_i + tl.sum(p, axis=1)

        p_bf16 = p.to(tl.bfloat16)

        offs_v = tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v).to(tl.float32)

        m_i = m_new
        l_i = l_new

    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E

    is_lonely_q = (l_i == 0.0)

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    acc_0 = acc_0 * output_scale[:, None]
    acc_1 = acc_1 * output_scale[:, None]
    acc_2 = acc_2 * output_scale[:, None]
    acc_3 = acc_3 * output_scale[:, None]

    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0)
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1)
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2)
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3)

    lse = tl.where(is_lonely_q, POS_INF, lse)

    lse_ptrs = LSE + pid_t * stride_lse_t + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_h)

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
    """Run the optimized Triton kernel with gathered BF16 KV data."""
    output = torch.empty((total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q_reshaped.device)
    lse = torch.empty((total_tokens, h_q), dtype=torch.float32, device=q_reshaped.device)

    BLOCK_H = BLOCK_H_OPT
    BLOCK_N = BLOCK_N_OPT
    BLOCK_D = BLOCK_D_OPT

    grid = (total_tokens, triton.cdiv(h_q, BLOCK_H))

    HAS_ATTN_SINK = attn_sink is not None

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
    """Fallback to PyTorch for very large topk."""
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
    Sparse attention decode using Triton.

    Takes FP8 quantized KV cache as input and performs gather + dequant
    for both V32 (d_qk=576) and MODEL1 (d_qk=512) layouts.
    """
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape

    def process_kv_scope(scope) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process KV scope with FP8 gather + dequant."""
        assert scope.indices_in_kvcache is not None
        topk = scope.indices_in_kvcache.size(-1)

        # Get block size
        block_size = scope.blocked_k.shape[1]

        # Build invalid mask
        invalid_mask = scope.indices_in_kvcache == -1
        if scope.topk_length is not None:
            invalid_mask = invalid_mask | (
                torch.arange(0, topk, device=q.device).view(1, 1, topk).broadcast_to(b, s_q, topk)
                >= scope.topk_length.view(b, 1, 1)
            )

        # Reshape for processing
        total_tokens = b * s_q
        indices_reshaped = scope.indices_in_kvcache.reshape(total_tokens, topk)
        invalid_mask_reshaped = invalid_mask.reshape(total_tokens, topk)

        # Use FP8 gather + dequant
        if scope.blocked_k_quantized is not None:
            if d_qk == 576:
                # V32 layout
                gathered_kv = gather_dequant_fp8_v32(
                    scope.blocked_k_quantized,
                    indices_reshaped,
                    invalid_mask_reshaped,
                    block_size
                )
            elif d_qk == 512:
                # MODEL1 layout
                gathered_kv = gather_dequant_fp8_model1(
                    scope.blocked_k_quantized,
                    indices_reshaped,
                    invalid_mask_reshaped,
                    block_size
                )
            else:
                raise ValueError(f"Unsupported d_qk: {d_qk}")
        else:
            # Fallback if no quantized data (should not happen in normal use)
            indices_clamped = torch.clamp(indices_reshaped, min=0)
            gathered_kv = scope.blocked_k.view(-1, d_qk).index_select(
                0, indices_clamped.view(-1)
            ).view(total_tokens, topk, d_qk).to(torch.bfloat16)
            gathered_kv[invalid_mask_reshaped] = 0

        return gathered_kv, invalid_mask_reshaped

    gathered_kv, invalid_mask_reshaped = process_kv_scope(kv_scope)

    if extra_kv_scope is not None:
        gathered_kv_extra, invalid_mask_extra = process_kv_scope(extra_kv_scope)
        gathered_kv = torch.cat([gathered_kv, gathered_kv_extra], dim=1)
        invalid_mask_reshaped = torch.cat([invalid_mask_reshaped, invalid_mask_extra], dim=1)

    total_topk = gathered_kv.shape[1]
    total_tokens = b * s_q

    # Ensure BF16 and handle NaN
    gathered_kv = gathered_kv.to(torch.bfloat16)
    gathered_kv = torch.where(gathered_kv != gathered_kv, torch.zeros_like(gathered_kv), gathered_kv)

    q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk)

    if not q_reshaped.is_contiguous():
        q_reshaped = q_reshaped.contiguous()
    if not gathered_kv.is_contiguous():
        gathered_kv = gathered_kv.contiguous()
    if not invalid_mask_reshaped.is_contiguous():
        invalid_mask_reshaped = invalid_mask_reshaped.contiguous()

    USE_TRITON = total_topk <= 8192

    if USE_TRITON:
        output, lse = _run_triton_attention_fused(
            q_reshaped, gathered_kv, invalid_mask_reshaped,
            d_v, sm_scale, total_tokens, h_q, total_topk, d_qk,
            attn_sink=attn_sink
        )
    else:
        q_reshaped_f32 = q_reshaped.float()
        gathered_kv_f32 = gathered_kv.float()
        output, lse = _run_pytorch_attention(
            q_reshaped_f32, gathered_kv_f32, invalid_mask_reshaped,
            d_v, sm_scale, total_tokens, h_q, total_topk
        )
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

    output = output.view(b, s_q, h_q, d_v)
    lse = lse.view(b, s_q, h_q)

    return output, lse.transpose(1, 2)
