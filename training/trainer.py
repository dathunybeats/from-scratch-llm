"""
ScratchLLM Trainer.

Features:
- Mixed precision (bf16/fp16) training
- Gradient accumulation (simulate large batch sizes on small GPU)
- Gradient clipping (critical for stability)
- Cosine LR schedule with warmup
- Checkpoint saving/resuming
- Evaluation loop with perplexity
- WandB logging (optional)
- Multi-GPU support via PyTorch DDP (optional)
"""
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from model.transformer import ScratchLLM
from model.config import ModelConfig


# -----------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Output
    out_dir: str = "checkpoints"
    run_name: str = "scratchllm"

    # Dataset
    train_data: str = "data/train.bin"         # Path to training data
    val_data: Optional[str] = "data/val.bin"   # Optional validation data
    data_format: str = "bin"                   # "bin" for pre-training, "jsonl" for SFT

    # Training hyperparameters
    max_steps: int = 100_000
    batch_size: int = 16                       # Micro batch size
    gradient_accumulation_steps: int = 4       # Effective batch = batch_size * grad_accum
    seq_len: int = 2048

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.1                  # min_lr = learning_rate * min_lr_ratio

    # Schedule
    warmup_steps: int = 2000
    lr_decay_steps: Optional[int] = None       # Defaults to max_steps

    # Precision
    dtype: str = "bf16"                        # "bf16", "fp16", or "fp32"

    # Evaluation & logging
    eval_interval: int = 500
    eval_steps: int = 50                       # Steps of evaluation data to average over
    log_interval: int = 10
    save_interval: int = 1000

    # Resuming
    resume_from: Optional[str] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "scratchllm"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# -----------------------------------------------------------------------
# LR Schedule
# -----------------------------------------------------------------------

def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    """
    Linear warmup + cosine decay schedule.
    - Steps 0..warmup_steps: linear from 0 to learning_rate
    - Steps warmup_steps..decay_steps: cosine decay to min_lr
    - After decay_steps: constant min_lr
    """
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    if step > decay_steps:
        return min_lr
    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: ScratchLLM,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Precision
        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[config.dtype]
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = config.dtype in ("bf16", "fp16") and self.device_type == "cuda"
        self.scaler = GradScaler(device=self.device_type, enabled=(config.dtype == "fp16" and self.use_amp))

        # Optimizer — separate weight decay for different param types
        self.optimizer = self._build_optimizer()

        # State
        self.step = 0
        self.best_val_loss = float("inf")

        # Resume if requested
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        # WandB
        self.wandb = None
        if config.use_wandb:
            try:
                import wandb
                wandb.init(project=config.wandb_project, name=config.run_name)
                self.wandb = wandb
            except ImportError:
                print("WandB not installed, skipping logging.")

        os.makedirs(config.out_dir, exist_ok=True)

        print(f"Trainer initialized on {self.device}")
        print(f"Effective batch size: {config.effective_batch_size}")
        print(f"Training for {config.max_steps:,} steps")

    def _build_optimizer(self) -> AdamW:
        """
        AdamW with weight decay applied only to 2D params (weights),
        not to 1D params (biases, norms). This is the standard setup.
        """
        decay_params = []
        no_decay_params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    no_decay_params.append(p)

        groups = [
            {"params": decay_params,    "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            fused=torch.cuda.is_available(),  # Fused AdamW is faster on CUDA (no-op on CPU)
            foreach=not torch.cuda.is_available(),  # foreach is faster on CPU
        )

    def _set_lr(self, step: int):
        decay_steps = self.config.lr_decay_steps or self.config.max_steps
        lr = cosine_lr_with_warmup(
            step,
            self.config.warmup_steps,
            decay_steps,
            self.config.learning_rate,
            self.config.learning_rate * self.config.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train(self):
        """Main training loop."""
        self.model.train()
        loader_iter = iter(self.train_loader)
        t0 = time.time()
        losses = []

        while self.step < self.config.max_steps:
            # Set LR for this step
            lr = self._set_lr(self.step)

            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                input_ids, labels = self._prepare_batch(batch)

                # Forward pass with AMP
                with autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.use_amp):
                    out = self.model(input_ids, labels=labels)
                    loss = out["loss"] / self.config.gradient_accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            # Gradient clipping (critical for stability)
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            losses.append(step_loss)
            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                t1 = time.time()
                dt = (t1 - t0) / self.config.log_interval
                tokens_per_sec = (
                    self.config.effective_batch_size * self.config.seq_len / dt
                )
                avg_loss = sum(losses[-self.config.log_interval:]) / self.config.log_interval
                print(
                    f"step {self.step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | {tokens_per_sec/1000:.1f}K tok/s"
                )
                t0 = t1

                if self.wandb:
                    self.wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                    }, step=self.step)

            # Evaluation
            if self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"  Val loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
                if self.wandb:
                    self.wandb.log({"val/loss": val_loss, "val/ppl": math.exp(val_loss)}, step=self.step)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")

                self.model.train()

            # Save checkpoint
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}.pt")

        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute validation loss."""
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in self.val_loader:
            if count >= self.config.eval_steps:
                break
            input_ids, labels = self._prepare_batch(batch)
            with autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.use_amp):
                out = self.model(input_ids, labels=labels)
            total_loss += out["loss"].item()
            count += 1
        return total_loss / max(count, 1)

    def _prepare_batch(self, batch) -> tuple:
        """Move batch to device."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            return x.to(self.device), y.to(self.device)
        # Dict batch (instruction tuning)
        return (
            batch["input_ids"].to(self.device),
            batch["labels"].to(self.device),
        )

    def save_checkpoint(self, filename: str):
        """Save model, optimizer, and training state."""
        path = os.path.join(self.config.out_dir, filename)
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.model.config,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Resume from a checkpoint."""
        print(f"Resuming from {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from step {self.step}")
