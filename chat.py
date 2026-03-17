"""
ScratchLLM — Interactive Chat Interface.

Usage:
    # Basic chat
    python chat.py --checkpoint checkpoints/scratchllm-nano/best.pt

    # With custom system prompt
    python chat.py --checkpoint checkpoints/best.pt \\
        --system "You are an expert Python programmer. Give concise, correct answers."

    # Adjust generation parameters
    python chat.py --checkpoint checkpoints/best.pt \\
        --temperature 0.7 --top_p 0.9 --max_tokens 512

    # Stream output token-by-token
    python chat.py --checkpoint checkpoints/best.pt --stream
"""
import argparse
import os
import sys


DEFAULT_SYSTEM = (
    "You are ScratchLLM, a helpful, harmless, and honest AI assistant. "
    "You give clear, accurate, and thoughtful answers."
)


def load_model_and_tokenizer(checkpoint_path: str, device: str = "auto"):
    import torch
    from model.transformer import ScratchLLM
    from tokenizer.bpe import BPETokenizer

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model_config = ckpt["config"]
    model = ScratchLLM(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = model.to(torch.bfloat16 if device == "cuda" else torch.float32)

    # Load tokenizer (from same dir as checkpoint, or default)
    tok_dir = os.path.join(os.path.dirname(checkpoint_path), "..", "tokenizer")
    if not os.path.exists(tok_dir):
        tok_dir = "tokenizer"
    tokenizer = BPETokenizer.load(tok_dir)

    print(f"Model loaded on {device} | {model.num_parameters()/1e6:.1f}M params")
    return model, tokenizer, device


def chat_loop(
    model,
    tokenizer,
    device: str,
    system_prompt: str = DEFAULT_SYSTEM,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 512,
    repetition_penalty: float = 1.1,
    stream: bool = True,
):
    import torch

    print("\n" + "="*60)
    print("ScratchLLM Chat")
    print("Commands: /clear (reset), /system (change system prompt), /quit")
    print("="*60 + "\n")

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("[Conversation cleared]\n")
            continue
        if user_input.lower().startswith("/system "):
            system_prompt = user_input[8:].strip()
            messages = [{"role": "system", "content": system_prompt}]
            print(f"[System prompt updated]\n")
            continue
        if user_input.lower() == "/help":
            print("Commands: /clear, /system <prompt>, /quit\n")
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Encode
        input_ids = tokenizer.encode_chat(messages, add_generation_prompt=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generate
        print("Assistant: ", end="", flush=True)
        generated_text = ""

        try:
            if stream:
                gen = model.generate(
                    input_tensor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=tokenizer.im_end_id,
                    stream=True,
                )
                for token_id in gen:
                    token_text = tokenizer.decode([token_id])
                    print(token_text, end="", flush=True)
                    generated_text += token_text
                print()
            else:
                gen = model.generate(
                    input_tensor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=tokenizer.im_end_id,
                    stream=False,
                )
                for token_ids in gen:
                    generated_text = tokenizer.decode(token_ids)
                print(generated_text)

        except torch.cuda.OutOfMemoryError:
            print("\n[OOM error — try reducing context length or max_tokens]")
            messages.pop()  # Remove the user message that caused OOM
            continue

        # Add assistant response to history
        messages.append({"role": "assistant", "content": generated_text.strip()})
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no_stream", action="store_false", dest="stream")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.device)
    chat_loop(
        model, tokenizer, device,
        system_prompt=args.system,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
