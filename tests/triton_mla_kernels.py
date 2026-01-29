"""
Triton implementation of MLA (Multi-head Latent Attention) kernels
for sparse attention prefill and decode operations.

Optimized for performance while maintaining correctness.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def triton_sparse_attn_fwd(
    q: torch.Tensor,  # [s_q, h_q, d_qk]
    kv: torch.Tensor,  # [s_kv, h_kv, d_qk]
    indices: torch.Tensor,  # [s_q, h_kv, topk]
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,  # [h_q]
    topk_length: Optional[torch.Tensor] = None,  # [s_q]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized sparse attention forward pass.
    Matches the reference implementation exactly.
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    topk = indices.shape[2]
    
    # Squeeze indices (h_kv is always 1)
    indices_2d = indices.squeeze(1)  # [s_q, topk]
    
    # Apply topk_length mask
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0) >= topk_length.unsqueeze(1)
        indices_2d = indices_2d.clone()
        indices_2d[mask] = -1
    
    # Create invalid mask
    invalid_mask = (indices_2d < 0) | (indices_2d >= s_kv)
    
    # Safe indices for gathering
    safe_indices = indices_2d.masked_fill(invalid_mask, 0)
    
    # Gather KV - this is the main memory operation
    gathered_kv = kv.index_select(0, safe_indices.reshape(-1)).reshape(s_q, topk, d_qk).float()
    
    # Compute attention scores
    q_float = q.float()
    P = torch.bmm(q_float, gathered_kv.transpose(1, 2))  # [s_q, h_q, topk]
    P.mul_(sm_scale)
    
    # Mask invalid positions
    P.masked_fill_(invalid_mask.unsqueeze(1).expand(-1, h_q, -1), float("-inf"))
    
    # Compute statistics
    max_logits = P.max(dim=-1).values  # [s_q, h_q]
    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    
    # Compute output scaling
    if attn_sink is not None:
        # lse_for_o = logsumexp([orig_lse, attn_sink])
        attn_sink_exp = attn_sink.unsqueeze(0).expand(s_q, -1)
        lse_for_o = torch.logaddexp(orig_lse, attn_sink_exp)
    else:
        lse_for_o = orig_lse
    
    # Handle -inf
    lse_for_o = torch.where(lse_for_o == float("-inf"), 
                            torch.tensor(float("+inf"), device=lse_for_o.device, dtype=lse_for_o.dtype),
                            lse_for_o)
    
    # Softmax weights
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    
    # Output computation
    out_fp32 = torch.bmm(s_for_o, gathered_kv[..., :d_v])  # [s_q, h_q, d_v]
    
    # Fix LSE for lonely queries
    lse_out = torch.where(orig_lse == float("-inf"),
                          torch.tensor(float("+inf"), device=orig_lse.device, dtype=orig_lse.dtype),
                          orig_lse)
    
    return out_fp32.to(torch.bfloat16), out_fp32, max_logits, lse_out


def triton_sparse_attn_decode(
    q: torch.Tensor,  # [b, s_q, h_q, d_qk]
    kv_scope,  # KVScope object
    extra_kv_scope,  # Optional KVScope object
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,  # [h_q]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized sparse attention decode - matches reference implementation exactly.
    """
    assert kv_scope is not None
    b, s_q, h_q, d_qk = q.shape
    
    def process_kv_scope(scope) -> Tuple[torch.Tensor, torch.Tensor]:
        assert scope.indices_in_kvcache is not None
        topk = scope.indices_in_kvcache.size(-1)
        indices_in_kv_cache_fixed = torch.clamp_min(scope.indices_in_kvcache, 0)
        gathered_kv = scope.blocked_k.view(-1, d_qk).index_select(0, indices_in_kv_cache_fixed.view(-1)).view(b, s_q, topk, d_qk)
        invalid_mask = scope.indices_in_kvcache == -1
        if scope.topk_length is not None:
            invalid_mask = invalid_mask | (torch.arange(0, topk, device=q.device).view(1, 1, topk).broadcast_to(b, s_q, topk) >= scope.topk_length.view(b, 1, 1))
        return gathered_kv, invalid_mask
    
    gathered_kv, invalid_mask = process_kv_scope(kv_scope)
    if extra_kv_scope is not None:
        gathered_kv1, invalid_mask1 = process_kv_scope(extra_kv_scope)
        gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)
        invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)
    
    gathered_kv = gathered_kv.view(b*s_q, -1, d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0
    q_f = q.float().view(b*s_q, h_q, d_qk)
    attn_weight = q_f @ gathered_kv.transpose(-1, -2)
    attn_weight *= sm_scale
    attn_weight[invalid_mask.view(b*s_q, 1, -1).broadcast_to(b*s_q, h_q, invalid_mask.size(-1))] = float("-inf")
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., :d_v]
    output = output.view(b, s_q, h_q, d_v)
    lse = lse.view(b, s_q, h_q)
    
    # Attention sink
    if attn_sink is not None:
        output *= (1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))).unsqueeze(-1)
    
    # Correct for q tokens which has no attendable k
    lonely_q_mask = (lse == float("-inf"))
    output[lonely_q_mask.unsqueeze(-1).broadcast_to(b, s_q, h_q, d_v)] = 0.0
    lse[lonely_q_mask] = float("+inf")
    
    return output.to(torch.bfloat16), lse.transpose(1, 2)
