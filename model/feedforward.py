"""
Feed-Forward Networks.

SwiGLU FFN — the standard in modern LLMs (LLaMA, DeepSeek, PaLM, Mistral).
Paper: https://arxiv.org/abs/2002.05202

SwiGLU formula:
  FFN(x) = (SiLU(xW_gate) ⊙ xW_up) W_down

Where ⊙ is element-wise multiply (gating).
This replaces the classic two-layer MLP and consistently outperforms GELU FFN.

Also includes DeepSeek-style Mixture of Experts (MoE) with:
- Top-k expert routing per token
- Auxiliary load balancing loss to prevent expert collapse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Intermediate size is typically ~2.75x hidden_size (not 4x like classic FFN)
    to keep parameter counts comparable.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Note: no bias — matches LLaMA/DeepSeek convention (faster, marginal quality difference)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up, then project down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertFFN(nn.Module):
    """Single expert in a MoE layer — same as SwiGLU but standalone."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """
    Mixture of Experts FFN (DeepSeek-inspired).

    Architecture:
    - Router: linear → softmax → top-k selection
    - N experts, each a SwiGLU FFN
    - Only top-k experts are activated per token
    - Load balancing auxiliary loss prevents all tokens routing to same expert

    Effective parameters = N * expert_params
    Active parameters per token = k * expert_params
    Speedup ≈ N/k (compute is for k experts, not N)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        # Router: maps hidden → expert logits
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Expert pool
        self.experts = nn.ModuleList([
            ExpertFFN(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])

        # Shared expert (DeepSeek V3 uses this for stability — always active)
        self.shared_expert = ExpertFFN(config.hidden_size, config.intermediate_size // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, T, hidden_size]
            aux_loss: scalar load balancing loss
        """
        B, T, H = x.shape
        flat_x = x.view(-1, H)  # [B*T, H]
        num_tokens = flat_x.shape[0]

        # Route
        router_logits = self.router(flat_x)              # [B*T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [B*T, num_experts]
        top_k_probs, top_k_ids = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize routing weights
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # [B*T, top_k]

        # Dispatch tokens to experts
        output = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            # Find which tokens route to expert i
            token_mask = (top_k_ids == i).any(dim=-1)  # [B*T]
            if not token_mask.any():
                continue
            expert_input = flat_x[token_mask]
            expert_output = expert(expert_input)

            # Gather the routing weight for this expert
            expert_weights = torch.zeros(num_tokens, device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                mask_k = (top_k_ids[:, k] == i)
                expert_weights[mask_k] = top_k_probs[:, k][mask_k]

            output[token_mask] += expert_output * expert_weights[token_mask].unsqueeze(-1)

        # Add shared expert (always active, DeepSeek V3 trick)
        output = output + self.shared_expert(flat_x)

        # Load balancing loss (auxiliary)
        # Encourages uniform expert usage: minimize variance of expert load
        # Loss = num_experts * sum(f_i * P_i) where f_i = fraction of tokens to expert i
        expert_fractions = (top_k_ids == torch.arange(self.num_experts, device=x.device).unsqueeze(0).unsqueeze(0)).float().mean(0).mean(0)
        expert_probs_mean = router_probs.mean(0)
        aux_loss = self.aux_loss_coeff * self.num_experts * (expert_fractions * expert_probs_mean).sum()

        return output.view(B, T, H), aux_loss
