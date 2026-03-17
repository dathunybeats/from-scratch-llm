"""
ScratchLLM — Main Training Script.

Pre-training:
    python train.py --config nano --data data/train.bin --val_data data/val.bin

Fine-tuning (SFT):
    python train.py --config nano --mode sft --data data/instructions.jsonl \\
        --resume checkpoints/best.pt --lr 1e-4 --max_steps 5000

Quick smoke test:
    python train.py --config nano --dummy --max_steps 100
"""
import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Train ScratchLLM")

    # Model
    parser.add_argument("--config", default="nano",
                        choices=["nano", "small", "medium", "medium-moe"],
                        help="Model size preset")
    parser.add_argument("--use_mtp", action="store_true", help="Enable Multi-Token Prediction")
    parser.add_argument("--seq_len", type=int, default=None, help="Override sequence length")

    # Data
    parser.add_argument("--data", type=str, default="data/train.bin")
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/")
    parser.add_argument("--mode", default="pretrain", choices=["pretrain", "sft"],
                        help="pretrain = raw tokens, sft = instruction JSONL")
    parser.add_argument("--dummy", action="store_true", help="Create and use dummy data (for testing)")

    # Training
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    # Output
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume", type=str, default=None)

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Imports (after argparse so --help is fast)
    # ------------------------------------------------------------------
    from model.config import CONFIGS
    from model.transformer import ScratchLLM
    from training.trainer import Trainer, TrainingConfig
    from training.data import (
        TokenizedDataset, ShuffledTokenDataset, InstructionDataset, make_dataloader
    )

    # ------------------------------------------------------------------
    # Setup dummy data if requested
    # ------------------------------------------------------------------
    if args.dummy:
        print("Creating dummy data for smoke testing...")
        from data.prepare import create_dummy_data
        data_dir = os.path.abspath("data")
        os.makedirs(data_dir, exist_ok=True)
        create_dummy_data(data_dir, args.tokenizer, num_tokens=500_000)
        args.data = os.path.join(data_dir, "train.bin")
        args.val_data = os.path.join(data_dir, "val.bin")
        args.max_steps = min(args.max_steps, 200)
        args.eval_interval = 50
        args.save_interval = 100
        args.warmup_steps = 10

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = CONFIGS[args.config]()
    if args.use_mtp:
        model_config.use_mtp = True
    if args.seq_len:
        model_config.seq_len = args.seq_len

    model = ScratchLLM(model_config)
    total_params = model.num_parameters()
    print(f"\nModel: ScratchLLM-{args.config}")
    print(f"Parameters: {total_params/1e6:.1f}M")
    print(f"Non-embedding params: {model.num_parameters(exclude_embeddings=True)/1e6:.1f}M\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    seq_len = args.seq_len or model_config.max_seq_len

    if args.mode == "pretrain":
        train_dataset = ShuffledTokenDataset(args.data, seq_len)
        val_dataset = TokenizedDataset(args.val_data, seq_len) if args.val_data else None
        train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=False)
        val_loader = make_dataloader(val_dataset, args.batch_size, shuffle=False) if val_dataset else None
    else:
        # SFT
        from tokenizer.bpe import BPETokenizer
        tok = BPETokenizer.load(args.tokenizer)
        train_dataset = InstructionDataset(args.data, tok, seq_len)
        val_dataset = InstructionDataset(args.val_data, tok, seq_len) if args.val_data else None
        train_loader = make_dataloader(train_dataset, args.batch_size)
        val_loader = make_dataloader(val_dataset, args.batch_size) if val_dataset else None

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    run_name = args.run_name or f"scratchllm-{args.config}"
    train_config = TrainingConfig(
        out_dir=os.path.join(args.out_dir, run_name),
        run_name=run_name,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seq_len=seq_len,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        dtype=args.dtype,
        resume_from=args.resume,
        use_wandb=args.wandb,
        wandb_project="scratchllm",
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(model, train_config, train_loader, val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
