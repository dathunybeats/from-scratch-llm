"""
ScratchLLM — Quick text generation for testing checkpoints during training.

Usage:
    python generate.py --checkpoint checkpoints/scratchllm-nano/best.pt
    python generate.py --checkpoint checkpoints/scratchllm-nano/best.pt --prompt "Once upon a time"
    python generate.py --checkpoint checkpoints/scratchllm-nano/step_5000.pt --max_tokens 200
"""
import argparse
import torch
from model.transformer import ScratchLLM
from tokenizer.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ScratchLLM(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()
    print(f"Model: {model.num_parameters()/1e6:.1f}M params | step {ckpt.get('step', '?')} | val_loss {ckpt.get('best_val_loss', float('nan')):.4f}\n")

    tok = BPETokenizer.load(args.tokenizer)

    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    print(args.prompt, end="", flush=True)

    ids = tok.encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for token_id in model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=True,
        ):
            print(tok.decode([token_id]), end="", flush=True)

    print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
