"""
Data preparation script.
Converts raw text into memory-mapped binary token files for efficient training.

Usage:
    # From a text file
    python data/prepare.py --input corpus.txt --output data/train.bin --tokenizer tokenizer/

    # From HuggingFace datasets
    python data/prepare.py --hf_dataset "openwebtext" --output data/ --tokenizer tokenizer/

    # Train the tokenizer first
    python data/prepare.py --train_tokenizer --input corpus.txt --vocab_size 32000 --save_tokenizer tokenizer/
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    Usage:
        python data/prepare.py --fineweb --output data/ --tokenizer tokenizer/
        python data/prepare.py --fineweb --subset sample-100BT --output data/ --tokenizer tokenizer/
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
                ids = tok.encode(text, add_eos=True)
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

    print(f"\nDone! {total_docs:,} documents | {total_tokens/1e9:.2f}B tokens")
    print(f"  train -> {train_path}")
    print(f"  val   -> {val_path}")
    print(f"\nTip: Train with:")
    print(f"  python train.py --config nano --data {train_path} --val_data {val_path} --dtype bf16 --grad_checkpointing --compile")


def _flush(ids: list, path: str):
    """Append a list of token IDs to a binary file."""
    arr = np.array(ids, dtype=np.uint16)
    with open(path, "ab") as f:
        f.write(arr.tobytes())


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
    os.makedirs(output_dir, exist_ok=True)  # ensure it exists right before writing
    train_path = os.path.join(output_dir, "train.bin")
    val_path   = os.path.join(output_dir, "val.bin")
    arr[:split].tofile(train_path)
    arr[split:].tofile(val_path)
    print(f"Dummy data: {split:,} train tokens, {len(arr)-split:,} val tokens")
    print(f"  train -> {train_path}")
    print(f"  val   -> {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input text file")
    parser.add_argument("--output", type=str, default="data/", help="Output directory or file")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/", help="Tokenizer directory")
    parser.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--save_tokenizer", type=str, default="tokenizer/")
    parser.add_argument("--dummy", action="store_true", help="Create dummy data for testing")
    parser.add_argument("--fineweb", action="store_true",
                        help="Download & tokenize FineWeb-Edu (best free pre-training dataset)")
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="FineWeb-Edu subset: sample-10BT | sample-100BT | default")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="Stop after this many tokens (0=no limit). Use to cap disk usage.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing data files before downloading (fresh start)")
    args = parser.parse_args()

    if args.dummy:
        create_dummy_data(args.output, args.tokenizer)
    elif args.fineweb:
        download_fineweb_edu(args.output, args.tokenizer, subset=args.subset, max_tokens=args.max_tokens, overwrite=args.overwrite)
    elif args.train_tokenizer:
        train_tokenizer(args.input, args.vocab_size, args.save_tokenizer)
    elif args.hf_dataset:
        tokenize_hf_dataset(args.hf_dataset, args.output, args.tokenizer)
    elif args.input:
        tokenize_file(args.input, args.output, args.tokenizer)
    else:
        print("Use --dummy to create test data, --fineweb for real data, or provide --input / --hf_dataset")
