"""
ScratchLLM Model Configuration
Inspired by DeepSeek, LLaMA, and Mistral architectures.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # -------------------------------------------------------------------
    # Core dimensions
    # -------------------------------------------------------------------
    vocab_size: int = 32000          # Tokenizer vocabulary size
    hidden_size: int = 2048          # Embedding / hidden dimension
    intermediate_size: int = 5632    # FFN intermediate (≈ 2.75x hidden, SwiGLU)
    num_layers: int = 24             # Transformer blocks
    num_heads: int = 16              # Query heads
    num_kv_heads: int = 4            # Key/Value heads (GQA: num_heads // num_kv_heads groups)
    head_dim: int = 128              # Dimension per head (hidden_size // num_heads)
    max_seq_len: int = 4096          # Maximum context length
    rope_theta: float = 500000.0     # RoPE base frequency (high = better long context)

    # -------------------------------------------------------------------
    # Normalization & Init
    # -------------------------------------------------------------------
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02
    tie_word_embeddings: bool = False  # Tie input/output embeddings

    # -------------------------------------------------------------------
    # Mixture of Experts (optional)
    # -------------------------------------------------------------------
    use_moe: bool = False
    num_experts: int = 8             # Total experts
    num_experts_per_token: int = 2   # Top-k routing
    moe_layer_freq: int = 2          # Every N layers uses MoE (rest use dense FFN)
    moe_aux_loss_coeff: float = 0.01 # Load balancing loss coefficient

    # -------------------------------------------------------------------
    # Attention options
    # -------------------------------------------------------------------
    attention_dropout: float = 0.0
    use_flash_attention: bool = True  # Falls back gracefully if unavailable
    sliding_window: Optional[int] = None  # None = full attention (like Mistral SWA)

    # -------------------------------------------------------------------
    # Multi-Token Prediction (DeepSeek V3 inspired)
    # -------------------------------------------------------------------
    use_mtp: bool = False             # Multi-token prediction auxiliary heads
    mtp_num_tokens: int = 3           # Predict next N tokens simultaneously

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------
    dropout: float = 0.0             # 0 for large models (regularized by data scale)

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
        self.head_dim = self.hidden_size // self.num_heads

    @property
    def num_query_groups(self) -> int:
        return self.num_heads // self.num_kv_heads


# -----------------------------------------------------------------------
# Preset configurations
# -----------------------------------------------------------------------

def get_nano_config() -> ModelConfig:
    """~125M parameters. Trainable on any GPU with 8GB VRAM."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1408,   # 512 * 2.75
        num_layers=12,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=2048,
        rope_theta=10000.0,
    )


def get_small_config() -> ModelConfig:
    """~1B parameters. Trainable on a single RTX 3090/4090."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        max_seq_len=4096,
        rope_theta=500000.0,
    )


def get_medium_config() -> ModelConfig:
    """~3B parameters. Trainable on A100/H100."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=3072,
        intermediate_size=8192,
        num_layers=32,
        num_heads=24,
        num_kv_heads=8,
        max_seq_len=8192,
        rope_theta=500000.0,
    )


def get_medium_moe_config() -> ModelConfig:
    """~3B active / 8B total parameters with MoE. Same compute as 3B dense."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        max_seq_len=4096,
        rope_theta=500000.0,
        use_moe=True,
        num_experts=8,
        num_experts_per_token=2,
        moe_layer_freq=2,
    )


CONFIGS = {
    "nano": get_nano_config,
    "small": get_small_config,
    "medium": get_medium_config,
    "medium-moe": get_medium_moe_config,
}
