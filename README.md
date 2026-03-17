# ScratchLLM

A modern, efficient LLM built from first principles — inspired by DeepSeek, LLaMA, and Mistral.

## Architecture

| Component | Design | Source |
|-----------|--------|--------|
| Normalization | RMSNorm (pre-norm) | LLaMA / DeepSeek |
| Positional encoding | RoPE | LLaMA / GPT-NeoX |
| Attention | Grouped Query Attention (GQA) | LLaMA 2 / Mistral |
| Activation | SwiGLU FFN | PaLM / LLaMA |
| Expert routing | MoE with load balancing (optional) | DeepSeek V2/V3 |
| Auxiliary training | Multi-Token Prediction (optional) | DeepSeek V3 |
| Inference | KV Cache + top-p/top-k/temperature | Standard |
| Flash Attention | PyTorch SDPA + optional flash-attn | Tri Dao |

## Model Sizes

| Config | Params | VRAM (inference) | VRAM (training) |
|--------|--------|-----------------|-----------------|
| nano | ~125M | 0.5 GB | 2 GB |
| small | ~1B | 2 GB | 8 GB |
| medium | ~3B | 6 GB | 24 GB |
| medium-moe | ~3B active / 8B total | 8 GB | 24 GB |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data (smoke test with dummy data)
```bash
python data/prepare.py --dummy --output data/ --tokenizer tokenizer/
```

### 3. Train the tokenizer on your corpus
```bash
python data/prepare.py --train_tokenizer --input your_corpus.txt \
    --vocab_size 32000 --save_tokenizer tokenizer/
```

### 4. Tokenize your training data
```bash
python data/prepare.py --input your_corpus.txt --output data/train.bin --tokenizer tokenizer/
```

### 5. Train
```bash
# Smoke test (nano model, dummy data)
python train.py --config nano --dummy --max_steps 200

# Real pre-training
python train.py --config nano --data data/train.bin --val_data data/val.bin

# Larger model
python train.py --config small --data data/train.bin --batch_size 8 --grad_accum 8

# With WandB logging
python train.py --config nano --data data/train.bin --wandb
```

### 6. Fine-tune on instructions (SFT)
```bash
python train.py --config nano --mode sft \
    --data data/instructions.jsonl \
    --resume checkpoints/scratchllm-nano/best.pt \
    --lr 1e-4 --max_steps 5000
```

### 7. Chat
```bash
python chat.py --checkpoint checkpoints/scratchllm-nano/best.pt
```

## Instruction Data Format (JSONL)

ChatML format:
```json
{"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4."}
]}
```

Alpaca format:
```json
{"instruction": "Translate to French.", "input": "Hello world", "output": "Bonjour le monde"}
```

## Training Tips

- **bf16** is recommended for A100/H100. Use **fp16** for older GPUs.
- Start with `nano` to validate your data pipeline before scaling.
- Effective batch size = `--batch_size` × `--grad_accum`. Target 256K-1M tokens/batch for stability.
- LR: `3e-4` for pre-training, `1e-4` to `5e-5` for SFT.
- Use `--wandb` to track loss curves and catch divergences early.
- Gradient clipping at 1.0 is critical — don't remove it.

## Project Structure

```
from_scratch_llm/
├── model/
│   ├── config.py       # Model configurations (nano/small/medium/MoE)
│   ├── rope.py         # Rotary Position Embeddings
│   ├── attention.py    # Grouped Query Attention + KV Cache
│   ├── feedforward.py  # SwiGLU FFN + MoE
│   └── transformer.py  # Full model (ScratchLLM)
├── tokenizer/
│   └── bpe.py          # BPE tokenizer (train from scratch or use tiktoken)
├── training/
│   ├── data.py         # Pre-training + SFT datasets
│   └── trainer.py      # Training loop with AMP, grad clipping, LR schedule
├── data/
│   └── prepare.py      # Data preparation utilities
├── train.py            # Main training script
├── chat.py             # Interactive chat
└── requirements.txt
```
