"""
Rotary Position Embeddings (RoPE)
Paper: https://arxiv.org/abs/2104.09864

Key properties:
- No extra learnable parameters
- Encodes relative positions naturally
- Extrapolates better to longer sequences than learned embeddings
- Used by LLaMA, DeepSeek, Mistral, Qwen
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for RoPE.
    Returns (cos, sin) each of shape [seq_len, head_dim]
    """
    # Inverse frequencies: theta_i = 1 / (theta^(2i/d)) for i in 0..d//2
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))

    # Positions: [seq_len]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Outer product: [seq_len, head_dim//2]
    freqs = torch.outer(t, inv_freq)

    # Duplicate for full head_dim: [seq_len, head_dim]
    emb = torch.cat([freqs, freqs], dim=-1)

    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last half of the head_dim: [-x2, x1]"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, kv_heads, seq_len, head_dim]
        cos: [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        sin: [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        position_ids: [batch, seq_len] — for non-contiguous positions (e.g., KV cache)
    """
    if position_ids is not None:
        # Gather the specific positions
        cos = cos[position_ids]  # [batch, seq_len, head_dim]
        sin = sin[position_ids]
        cos = cos.unsqueeze(1)   # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
    else:
        seq_len = q.shape[2]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RoPE(nn.Module):
    """RoPE module that caches cos/sin and handles dynamic resizing."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        cos, sin = build_rope_cache(max_seq_len, head_dim, theta)
        # Register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_rope(q, k, self.cos_cache, self.sin_cache, position_ids)

    def extend(self, new_seq_len: int):
        """Extend the cache if we encounter a longer sequence."""
        if new_seq_len > self.max_seq_len:
            cos, sin = build_rope_cache(
                new_seq_len, self.head_dim, self.theta,
                device=self.cos_cache.device, dtype=self.cos_cache.dtype
            )
            self.cos_cache = cos
            self.sin_cache = sin
            self.max_seq_len = new_seq_len
