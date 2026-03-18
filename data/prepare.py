"""
Data preparation script.
Converts raw text into memory-mapped binary token files for efficient training.

Usage:
    # RECOMMENDED: one-shot setup (samples text, trains tokenizer, tokenizes data)
    python data/prepare.py --setup
    python data/prepare.py --setup --max_tokens 500000000  # cap at 500M tokens (~10GB)

    # Manual steps (if you want control over each stage):
    python data/prepare.py --sample_text --num_sample_docs 50000   # 1. sample text for tokenizer
    python data/prepare.py --train_tokenizer --input data/tok_sample.txt --vocab_size 32000
    python data/prepare.py --fineweb --max_tokens 500000000 --overwrite

    # From a text file
    python data/prepare.py --input corpus.txt --output data/train.bin --tokenizer tokenizer/

    # Smoke test
    python data/prepare.py --dummy
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -----------------------------------------------------------------------
# Tokenizer training
# -----------------------------------------------------------------------

def train_tokenizer(text_file: str, vocab_size: int, save_dir: str):
    from tokenizer.bpe import BPETokenizer
    print(f"Training BPE tokenizer on {text_file} (vocab_size={vocab_size})")
    with open(text_file, encoding="utf-8") as f:
        text = f.read()
    tok = BPETokenizer()
    tok.train(text, vocab_size=vocab_size, verbose=True)
    tok.save(save_dir)
    print(f"Tokenizer saved to {save_dir}")
    return tok


# -----------------------------------------------------------------------
# Text file tokenization
# -----------------------------------------------------------------------

def tokenize_file(
    text_file: str,
    output_file: str,
    tokenizer_dir: str,
    chunk_size: int = 1_000_000,
):
    """Tokenize a large text file into a binary token file."""
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer.load(tokenizer_dir)
    print(f"Tokenizing {text_file} → {output_file}")

    all_ids = []
    total_tokens = 0

    with open(text_file, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            ids = tok.encode(chunk)
            all_ids.extend(ids)
            total_tokens += len(ids)
            print(f"  Processed {total_tokens:,} tokens...", end="\r")

    print(f"\nTotal tokens: {total_tokens:,}")
    arr = np.array(all_ids, dtype=np.uint16)
    with open(output_file, "wb") as f:
        f.write(arr.tobytes())
    print(f"Saved to {output_file} ({arr.nbytes / 1e9:.2f} GB)")


# -----------------------------------------------------------------------
# HuggingFace dataset tokenization
# -----------------------------------------------------------------------

def tokenize_hf_dataset(
    dataset_name: str,
    output_dir: str,
    tokenizer_dir: str,
    split: str = "train",
    text_column: str = "text",
    num_proc: int = 8,
):
    """Tokenize a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer.load(tokenizer_dir)

    print(f"Loading {dataset_name} ({split})")
    ds = load_dataset(dataset_name, split=split, num_proc=num_proc)

    def tokenize(batch):
        return {"ids": [tok.encode(text, add_eos=True) for text in batch[text_column]]}

    ds = ds.map(tokenize, batched=True, num_proc=num_proc, remove_columns=ds.column_names)

    # Flatten and save
    all_ids = []
    for sample in ds:
        all_ids.extend(sample["ids"])

    arr = np.array(all_ids, dtype=np.uint16)
    output_file = os.path.join(output_dir, f"{split}.bin")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(arr.tobytes())
    print(f"Saved {len(arr):,} tokens to {output_file}")


# -----------------------------------------------------------------------
# FineWeb-Edu: sample text for tokenizer training
# -----------------------------------------------------------------------

def sample_fineweb_text(
    output_file: str,
    num_docs: int = 50_000,
    subset: str = "sample-10BT",
):
    """
    Stream the first N documents from FineWeb-Edu and save as plain text.
    Used to train the BPE tokenizer on representative text before tokenizing everything.

    50K documents ≈ 500MB of text — enough for a good 32K vocab BPE tokenizer.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Sampling {num_docs:,} docs from FineWeb-Edu ({subset}) → {output_file}")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(ds):
            if i >= num_docs:
                break
            f.write(doc["text"] + "\n\n")
            if (i + 1) % 5000 == 0:
                print(f"  Sampled {i+1:,}/{num_docs:,} docs...", end="\r")

    size_mb = os.path.getsize(output_file) / 1e6
    print(f"\nSample saved: {output_file} ({size_mb:.1f} MB)")


# -----------------------------------------------------------------------
# FineWeb-Edu: full tokenization
# -----------------------------------------------------------------------

def download_fineweb_edu(
    output_dir: str,
    tokenizer_dir: str,
    subset: str = "sample-10BT",
    val_fraction: float = 0.005,
    shard_size: int = 100_000_000,
    max_tokens: int = 0,
    overwrite: bool = False,
):
    """
    Download and tokenize FineWeb-Edu from HuggingFace.

    FineWeb-Edu is the best freely available pre-training dataset:
    - Filtered Common Crawl with educational quality scoring
    - 1.3T tokens total; "sample-10BT" = 10B tokens (good starting point)
    - Outperforms The Pile, C4, and OpenWebText on downstream benchmarks

    Subsets:
        "sample-10BT"   — 10B tokens,  ~21 GB download  (RTX 3090 / 4090 scale)
        "sample-100BT"  — 100B tokens, ~210 GB download  (A100 scale)
        "default"       — 1.3T tokens, full dataset      (datacenter scale)
    """
    import time

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    from tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.load(tokenizer_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.bin")
    val_path   = os.path.join(output_dir, "val.bin")

    # Wipe existing files if overwrite requested
    if overwrite:
        for p in (train_path, val_path):
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed existing {p}")

    def _load_ds(skip: int = 0):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=subset,
            split="train",
            streaming=True,
        )
        if skip > 0:
            ds = ds.skip(skip)
        return ds

    train_ids = []
    val_ids   = []
    total_docs = 0
    total_tokens = 0
    max_retries = 20

    print(f"Streaming FineWeb-Edu ({subset}) from HuggingFace...")
    print("Tokenizing (streaming — no full dataset download required)...")

    ds = _load_ds()
    retries = 0

    while True:
        try:
            for doc in ds:
                text = doc["text"]
                # BOS marks document start (helps model learn clean document boundaries),
                # EOS marks document end (helps model learn when to stop generating).
                ids = tok.encode(text, add_bos=True, add_eos=True)
                total_tokens += len(ids)
                total_docs   += 1
                retries = 0  # reset on success

                if total_docs % round(1 / val_fraction) == 0:
                    val_ids.extend(ids)
                else:
                    train_ids.extend(ids)

                if total_docs % 5000 == 0:
                    print(f"  {total_docs:,} docs | {total_tokens/1e9:.2f}B tokens", end="\r")

                if max_tokens > 0 and total_tokens >= max_tokens:
                    print(f"\nReached {max_tokens/1e9:.2f}B token limit, stopping.")
                    break

                if len(train_ids) >= shard_size:
                    _flush(train_ids, train_path)
                    train_ids = []
                if len(val_ids) >= shard_size // 10:
                    _flush(val_ids, val_path)
                    val_ids = []

            break  # completed without error

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"\nFailed after {max_retries} retries. Saving progress and exiting.")
                break
            wait = min(2 ** retries, 60)
            print(f"\nNetwork error (retry {retries}/{max_retries}, waiting {wait}s): {e}")
            # Flush what we have before reconnecting
            if train_ids:
                _flush(train_ids, train_path)
                train_ids = []
            if val_ids:
                _flush(val_ids, val_path)
                val_ids = []
            time.sleep(wait)
            print(f"  Reconnecting from doc {total_docs:,}...")
            ds = _load_ds(skip=total_docs)

    # Final flush
    if train_ids:
        _flush(train_ids, train_path)
    if val_ids:
        _flush(val_ids, val_path)

    # Write sentinel file so other processes know download is fully complete
    done_path = os.path.join(output_dir, ".download_done")
    with open(done_path, "w") as f:
        f.write(f"{total_tokens}\n")

    # Write metadata.json — used by train.py to validate vocab_size before training
    _write_metadata(output_dir, tokenizer_dir, tok.vocab_size, total_tokens, subset)

    print(f"\nDone! {total_docs:,} documents | {total_tokens/1e9:.2f}B tokens")
    print(f"  train -> {train_path}")
    print(f"  val   -> {val_path}")
    print(f"\nTip: Train with:")
    print(f"  python train.py --config nano --data data/train.bin --val_data data/val.bin --compile --grad_checkpointing --num_workers 2")


def _flush(ids: list, path: str):
    """Append a list of token IDs to a binary file."""
    arr = np.array(ids, dtype=np.uint16)
    with open(path, "ab") as f:
        f.write(arr.tobytes())


def _write_metadata(output_dir: str, tokenizer_dir: str, vocab_size: int, total_tokens: int, subset: str = ""):
    """
    Write metadata.json alongside the .bin files.
    train.py reads this to validate that the model vocab_size matches the tokenizer.
    This prevents the #1 silent failure: tokenizer/model vocab mismatch.
    """
    meta = {
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "tokenizer_dir": os.path.abspath(tokenizer_dir),
        "subset": subset,
    }
    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata written: {path}")


# -----------------------------------------------------------------------
# One-shot setup (RECOMMENDED)
# -----------------------------------------------------------------------

def setup_training(
    output_dir: str = "data",
    tokenizer_dir: str = "tokenizer",
    subset: str = "sample-10BT",
    vocab_size: int = 32000,
    num_sample_docs: int = 50_000,
    max_tokens: int = 0,
    overwrite: bool = False,
):
    """
    One-shot setup: sample text → train tokenizer → tokenize FineWeb-Edu.

    This is the recommended way to set up training from scratch.
    Run: python data/prepare.py --setup

    Steps:
      1. Stream 50K FineWeb-Edu docs → tok_sample.txt  (for tokenizer training)
      2. Train 32K BPE tokenizer on the sample
      3. Stream all of FineWeb-Edu, tokenize with the new tokenizer → train.bin + val.bin

    The metadata.json written at the end is read by train.py to validate
    that the model vocab_size matches what the data was tokenized with.
    """
    tok_path = os.path.join(tokenizer_dir, "tokenizer.json")
    sample_file = os.path.join(output_dir, "tok_sample.txt")

    print("=" * 60)
    print("ScratchLLM — One-shot training setup")
    print("=" * 60)

    # Step 1: Sample text for tokenizer training
    if os.path.exists(tok_path) and not overwrite:
        print(f"\n[Step 1/3] Tokenizer already exists at {tok_path} — skipping.")
        print("  (Use --overwrite to retrain from scratch.)")
    else:
        if os.path.exists(sample_file) and not overwrite:
            size_mb = os.path.getsize(sample_file) / 1e6
            print(f"\n[Step 1a/3] Sample file already exists ({size_mb:.1f} MB) — skipping text sampling.")
        else:
            print(f"\n[Step 1a/3] Sampling {num_sample_docs:,} docs from FineWeb-Edu for tokenizer training...")
            sample_fineweb_text(sample_file, num_docs=num_sample_docs, subset=subset)

        # Step 2: Train tokenizer
        print(f"\n[Step 1b/3] Training {vocab_size:,}-vocab BPE tokenizer...")
        train_tokenizer(sample_file, vocab_size, tokenizer_dir)

    # Step 3: Tokenize full dataset
    print(f"\n[Step 3/3] Tokenizing FineWeb-Edu ({subset})...")
    if max_tokens > 0:
        print(f"  Token cap: {max_tokens/1e9:.1f}B tokens")
    download_fineweb_edu(
        output_dir,
        tokenizer_dir,
        subset=subset,
        max_tokens=max_tokens,
        overwrite=overwrite,
    )

    print("\n" + "=" * 60)
    print("Setup complete! Start training with:")
    print("  python train.py --config nano \\")
    print("    --data data/train.bin --val_data data/val.bin \\")
    print("    --compile --grad_checkpointing --num_workers 2")
    print("=" * 60)


# -----------------------------------------------------------------------
# Dummy data (for smoke testing the pipeline)
# -----------------------------------------------------------------------

def create_dummy_data(output_dir: str, tokenizer_dir: str, num_tokens: int = 1_000_000):
    """Create dummy training data for testing the pipeline."""
    from tokenizer.bpe import BPETokenizer

    output_dir = os.path.abspath(output_dir)
    tokenizer_dir = os.path.abspath(tokenizer_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Simple repeating text for smoke testing
    sample_text = """The quick brown fox jumps over the lazy dog.
    In the beginning was the Word, and the Word was with God.
    To be, or not to be, that is the question.
    It was the best of times, it was the worst of times.
    All happy families are alike; each unhappy family is unhappy in its own way.
    Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.
    """ * 10000

    # Check if tokenizer exists, if not create a tiny one
    if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        print("Training tiny tokenizer on dummy data...")
        tok = BPETokenizer()
        tok.train(sample_text, vocab_size=1000, verbose=False)
        tok.save(tokenizer_dir)
    else:
        tok = BPETokenizer.load(tokenizer_dir)

    ids = tok.encode(sample_text * 3)
    ids = (ids * (num_tokens // len(ids) + 1))[:num_tokens]
    arr = np.array(ids, dtype=np.uint16)

    # 90/10 train/val split
    split = int(0.9 * len(arr))
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.bin")
    val_path   = os.path.join(output_dir, "val.bin")
    arr[:split].tofile(train_path)
    arr[split:].tofile(val_path)

    _write_metadata(output_dir, tokenizer_dir, tok.vocab_size, num_tokens, subset="dummy")

    print(f"Dummy data: {split:,} train tokens, {len(arr)-split:,} val tokens")
    print(f"  train -> {train_path}")
    print(f"  val   -> {val_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- One-shot setup (recommended) ---
    parser.add_argument("--setup", action="store_true",
                        help="One-shot: sample text + train tokenizer + tokenize FineWeb-Edu")
    parser.add_argument("--num_sample_docs", type=int, default=50_000,
                        help="Number of FineWeb docs to sample for tokenizer training (default: 50000)")

    # --- Manual steps ---
    parser.add_argument("--sample_text", action="store_true",
                        help="Only sample text from FineWeb-Edu for tokenizer training")
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--fineweb", action="store_true",
                        help="Download & tokenize FineWeb-Edu (requires tokenizer already trained)")
    parser.add_argument("--dummy", action="store_true",
                        help="Create dummy data for smoke testing the pipeline")

    # --- Paths ---
    parser.add_argument("--input", type=str, help="Input text file")
    parser.add_argument("--output", type=str, default="data/", help="Output directory or file")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/", help="Tokenizer directory")
    parser.add_argument("--save_tokenizer", type=str, default="tokenizer/")
    parser.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name")

    # --- Options ---
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="FineWeb-Edu subset: sample-10BT | sample-100BT | default")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="Stop after this many tokens (0=no limit). Cap at 500M for RTX 3090.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing data/tokenizer files before starting fresh")

    args = parser.parse_args()

    if args.setup:
        setup_training(
            output_dir=args.output,
            tokenizer_dir=args.tokenizer,
            subset=args.subset,
            vocab_size=args.vocab_size,
            num_sample_docs=args.num_sample_docs,
            max_tokens=args.max_tokens,
            overwrite=args.overwrite,
        )
    elif args.sample_text:
        sample_file = os.path.join(args.output, "tok_sample.txt")
        sample_fineweb_text(sample_file, num_docs=args.num_sample_docs, subset=args.subset)
    elif args.dummy:
        create_dummy_data(args.output, args.tokenizer)
    elif args.fineweb:
        download_fineweb_edu(
            args.output, args.tokenizer,
            subset=args.subset,
            max_tokens=args.max_tokens,
            overwrite=args.overwrite,
        )
    elif args.train_tokenizer:
        train_tokenizer(args.input, args.vocab_size, args.save_tokenizer)
    elif args.hf_dataset:
        tokenize_hf_dataset(args.hf_dataset, args.output, args.tokenizer)
    elif args.input:
        tokenize_file(args.input, args.output, args.tokenizer)
    else:
        print("Usage:")
        print("  python data/prepare.py --setup                  # recommended one-shot setup")
        print("  python data/prepare.py --dummy                  # smoke test")
        print("  python data/prepare.py --help                   # all options")
