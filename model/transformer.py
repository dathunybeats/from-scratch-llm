"""
ScratchLLM — Full Transformer Model.

Architecture summary:
  Embedding → N x (RMSNorm → GQA → residual → RMSNorm → FFN → residual) → RMSNorm → LM Head

Design choices:
  - Pre-normalization (norm before sub-layer, not after) — more stable training
  - RMSNorm instead of LayerNorm — faster, same quality
  - No bias in Linear layers — cleaner, marginal quality gain
  - Tied input/output embeddings (optional) — saves parameters
  - Multi-Token Prediction auxiliary head (optional, DeepSeek V3)
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import ModelConfig
from .attention import GroupedQueryAttention, KVCache
from .feedforward import SwiGLUFFN, MoEFFN


# -----------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Paper: https://arxiv.org/abs/1910.07467

    Faster than LayerNorm: no mean subtraction (just RMS scaling).
    Used by LLaMA, DeepSeek, Mistral, Qwen, etc.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


# -----------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block.
    Pre-norm architecture: Norm → Sub-layer → Residual
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)

        # FFN — use MoE every `moe_layer_freq` layers (if enabled)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        use_moe_this_layer = (
            config.use_moe and
            layer_idx % config.moe_layer_freq == 0 and
            layer_idx > 0  # Always start with dense layer
        )
        self.ffn = MoEFFN(config) if use_moe_this_layer else SwiGLUFFN(config)
        self.is_moe = use_moe_this_layer

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[torch.Tensor]]:
        # Attention with pre-norm and residual
        residual = x
        x = self.attn_norm(x)
        x, kv_cache = self.attn(x, attention_mask, position_ids, kv_cache, use_cache)
        x = residual + x

        # FFN with pre-norm and residual
        residual = x
        x = self.ffn_norm(x)
        aux_loss = None
        if self.is_moe:
            x, aux_loss = self.ffn(x)
        else:
            x = self.ffn(x)
        x = residual + x

        return x, kv_cache, aux_loss


# -----------------------------------------------------------------------
# Multi-Token Prediction Head (DeepSeek V3 inspired)
# -----------------------------------------------------------------------

class MultiTokenPredictionHead(nn.Module):
    """
    Predict the next N tokens simultaneously as an auxiliary training objective.
    This improves sample efficiency — the model learns to plan ahead.

    Paper: https://arxiv.org/abs/2404.19737
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_tokens = config.mtp_num_tokens
        self.heads = nn.ModuleList([
            nn.Sequential(
                RMSNorm(config.hidden_size, config.rms_norm_eps),
                nn.Linear(config.hidden_size, config.vocab_size, bias=False),
            )
            for _ in range(config.mtp_num_tokens - 1)  # First token predicted by main head
        ])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Returns logits for tokens 2..N (token 1 handled by main LM head)."""
        return [head(hidden_states) for head in self.heads]


# -----------------------------------------------------------------------
# Main Model
# -----------------------------------------------------------------------

class ScratchLLM(nn.Module):
    """
    ScratchLLM: A modern, efficient LLM built from first principles.

    Combines the best ideas from:
    - LLaMA: RMSNorm, SwiGLU, RoPE, GQA
    - DeepSeek: MoE layers, MTP auxiliary training
    - Mistral: GQA, sliding window attention (optional)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, idx) for idx in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Optionally tie embeddings (saves params, used by many models)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Multi-Token Prediction (optional)
        self.mtp_head = MultiTokenPredictionHead(config) if config.use_mtp else None

        # Initialize weights
        self.apply(self._init_weights)

        # Scale residual projections (GPT-2 trick for depth stability)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.num_layers))

        print(f"ScratchLLM initialized: {self.num_parameters():,} parameters")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed_tokens.weight.numel()
            if not self.config.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,                   # [B, T]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,     # [B, T] for training
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape

        # Embed tokens
        x = self.embed_tokens(input_ids)  # [B, T, H]

        # Auto-generate position_ids if not provided
        if position_ids is None:
            offset = kv_caches[0].current_len if (kv_caches and kv_caches[0]) else 0
            position_ids = torch.arange(offset, offset + T, device=input_ids.device).unsqueeze(0)

        # Forward through all layers
        total_aux_loss = torch.tensor(0.0, device=x.device)
        new_caches = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            cache = kv_caches[idx] if kv_caches else None

            # Gradient checkpointing: recompute activations on backward instead of storing them.
            # Saves ~60% of activation memory at the cost of ~30% extra compute.
            # Only applies during training (use_cache=False); inference always takes the normal path.
            if self.config.use_gradient_checkpointing and self.training and not use_cache:
                x, new_cache, aux_loss = grad_checkpoint(
                    layer, x, attention_mask, position_ids, None, False,
                    use_reentrant=False,
                )
            else:
                x, new_cache, aux_loss = layer(x, attention_mask, position_ids, cache, use_cache)

            if use_cache:
                new_caches.append(new_cache)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        # Final norm
        x = self.norm(x)

        # LM head logits
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # MTP auxiliary logits
        mtp_logits = self.mtp_head(x) if self.mtp_head is not None else None

        # Loss computation
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Label smoothing: only during training (eval should report true NLL for fair comparison)
            ls = getattr(self.config, 'label_smoothing', 0.0) if self.training else 0.0

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=ls,
            )

            # Z-loss: penalizes large softmax normalizers (from Gemini/PaLM).
            # Prevents logit scale explosion that causes training instability over long runs.
            # Formula: z_loss = weight * E[log(sum(exp(logits)))^2]
            z_loss_weight = getattr(self.config, 'z_loss_weight', 0.0)
            if self.training and z_loss_weight > 0:
                log_z = torch.logsumexp(shift_logits.view(-1, self.config.vocab_size).float(), dim=-1)
                z_loss = z_loss_weight * log_z.pow(2).mean()
                loss = loss + z_loss

            # Add MTP auxiliary loss
            if mtp_logits is not None:
                for i, extra_logits in enumerate(mtp_logits):
                    offset = i + 2  # predict token at t+2, t+3, ...
                    if T > offset:
                        mtp_shift_logits = extra_logits[:, :-offset, :].contiguous()
                        mtp_shift_labels = labels[:, offset:].contiguous()
                        mtp_loss = F.cross_entropy(
                            mtp_shift_logits.view(-1, self.config.vocab_size),
                            mtp_shift_labels.view(-1),
                            ignore_index=-100,
                        )
                        loss = loss + 0.1 * mtp_loss  # small auxiliary weight

            # Add MoE load balancing loss
            loss = loss + total_aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "kv_caches": new_caches,
        }

    def create_kv_caches(
        self,
        batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> List[KVCache]:
        """Allocate KV caches for all layers (for inference)."""
        if max_seq_len is None:
            max_seq_len = self.config.max_seq_len
        if device is None:
            device = next(self.parameters()).device
        return [
            KVCache(batch_size, max_seq_len, self.config.num_kv_heads, self.config.head_dim, device, dtype)
            for _ in range(self.config.num_layers)
        ]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        stream: bool = False,
    ):
        """
        Autoregressive generation with KV cache.
        Supports temperature, top-p (nucleus), top-k, and repetition penalty.
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        kv_caches = self.create_kv_caches(B, device=device, dtype=next(self.parameters()).dtype)

        # Prefill: process the full prompt
        out = self.forward(input_ids, kv_caches=kv_caches, use_cache=True)
        kv_caches = out["kv_caches"]
        logits = out["logits"][:, -1, :]  # [B, vocab]

        generated = []
        for _ in range(max_new_tokens):
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                prev_tokens = torch.tensor(generated, device=device).unsqueeze(0)
                logits = _apply_repetition_penalty(logits, prev_tokens, repetition_penalty)

            # Top-k filtering
            if top_k > 0:
                logits = _top_k_filter(logits, top_k)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                logits = _top_p_filter(logits, top_p)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            token_id = next_token[0, 0].item()
            generated.append(token_id)

            if stream:
                yield token_id

            if eos_token_id is not None and token_id == eos_token_id:
                break

            # Next step: single new token
            out = self.forward(next_token, kv_caches=kv_caches, use_cache=True)
            kv_caches = out["kv_caches"]
            logits = out["logits"][:, -1, :]

        if not stream:
            yield generated


# -----------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------

def _apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    score = torch.gather(logits, 1, prev_tokens)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, prev_tokens, score)
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = float("-inf")
    return logits


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above threshold (shift right to keep first above)
    sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_idx_to_remove] = float("-inf")
    logits.scatter_(1, sorted_idx, sorted_logits)
    return logits
