"""
Grouped Query Attention (GQA) with KV Cache support.
Paper: https://arxiv.org/abs/2305.13245

GQA is the key efficiency trick in LLaMA 2/3, Mistral, DeepSeek:
- Standard MHA:  Q, K, V all have `num_heads` heads  → expensive KV cache
- MQA:           Q has `num_heads`, K/V have 1 head   → too much quality loss
- GQA (sweet spot): Q has `num_heads`, K/V have `num_kv_heads` heads
                    where num_kv_heads divides num_heads evenly

Memory savings: (num_heads / num_kv_heads)x smaller KV cache
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import RoPE


class KVCache:
    """
    Pre-allocated KV cache for fast autoregressive inference.
    Avoids recomputing keys/values for already-processed tokens.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.current_len = 0

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new k/v and return full accumulated k/v."""
        seq_len = k.shape[2]
        start = self.current_len
        end = start + seq_len
        self.k_cache[:, :, start:end, :] = k
        self.v_cache[:, :, start:end, :] = v
        self.current_len = end
        return self.k_cache[:, :, :end, :], self.v_cache[:, :, :end, :]

    def reset(self):
        self.current_len = 0


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with:
    - RoPE positional embeddings
    - KV cache support for inference
    - Flash Attention (if available, falls back to standard)
    - Sliding window attention (optional)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_query_groups = config.num_query_groups  # num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window

        # Projections — note KV projections are smaller
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta)
        self.attn_dropout = config.attention_dropout

        # Check Flash Attention availability once
        self._flash_available = self._check_flash()

    def _check_flash(self) -> bool:
        try:
            from flash_attn import flash_attn_func  # noqa: F401
            return True
        except ImportError:
            return False

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, H]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # [B, T, num_heads * head_dim]
        k = self.k_proj(x)  # [B, T, num_kv_heads * head_dim]
        v = self.v_proj(x)

        # Reshape to [B, heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, position_ids)

        # Update KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Expand K, V for GQA: repeat to match num_heads
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_query_groups, dim=1)  # [B, num_heads, S, head_dim]
            v = v.repeat_interleave(self.num_query_groups, dim=1)

        # Attention
        if self.config.use_flash_attention and self._flash_available and not self.training:
            attn_out = self._flash_attention(q, k, v)
        else:
            attn_out = self._standard_attention(q, k, v, attention_mask, T)

        # Merge heads and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        out = self.o_proj(attn_out)

        return out, kv_cache if use_cache else None

    def _flash_attention(self, q, k, v) -> torch.Tensor:
        """Flash Attention path (requires flash_attn package)."""
        from flash_attn import flash_attn_func
        # flash_attn expects [B, T, heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            causal=True,
            window_size=(self.sliding_window, self.sliding_window) if self.sliding_window else (-1, -1),
        )
        return out.transpose(1, 2)

    def _standard_attention(self, q, k, v, mask, T) -> torch.Tensor:
        """Standard scaled dot-product attention with causal mask."""
        # Use PyTorch's built-in SDPA (uses Flash Attention kernel if available via torch.backends)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,          # We use is_causal=True instead
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        return attn_out
