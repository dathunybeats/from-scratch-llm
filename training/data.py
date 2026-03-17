"""
Data loading and preprocessing for ScratchLLM.

Supports:
- Raw text files (pre-training)
- JSONL instruction datasets (fine-tuning)
- HuggingFace datasets (via the datasets library)
- Memory-mapped binary files for large-scale training (like nanoGPT)

For pre-training at scale, the recommended pipeline is:
  text files → tokenize → save as .bin (uint16 numpy) → memory-map during training
  This avoids loading the full dataset into RAM.
"""
import json
import os
import struct
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# -----------------------------------------------------------------------
# Pre-training dataset: Memory-mapped token files
# -----------------------------------------------------------------------

class TokenizedDataset(Dataset):
    """
    Memory-mapped pre-training dataset.
    Expects the data to be pre-tokenized and saved as a binary file of uint16 tokens.
    This is efficient for large datasets (100B+ tokens) since it never loads all data into RAM.

    To prepare data:
        python data/prepare.py --input corpus.txt --output data/train.bin --tokenizer tokenizer/
    """

    def __init__(self, bin_file: str, seq_len: int):
        self.seq_len = seq_len
        # Memory-map the file
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.num_samples = (len(self.data) - 1) // seq_len
        print(f"Dataset: {len(self.data):,} tokens -> {self.num_samples:,} sequences of length {seq_len}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for the label shift
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class ShuffledTokenDataset(IterableDataset):
    """
    Streaming dataset that randomly samples chunks from a large token file.
    Better for training than sequential access (avoids distribution shifts mid-epoch).
    """

    def __init__(self, bin_file: str, seq_len: int, seed: int = 42):
        self.seq_len = seq_len
        self.seed = seed
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        while True:
            start = rng.integers(0, self.num_tokens - self.seq_len - 1)
            chunk = torch.from_numpy(self.data[start:start + self.seq_len + 1].astype(np.int64))
            yield chunk[:-1], chunk[1:]


# -----------------------------------------------------------------------
# Fine-tuning dataset: Instruction/Chat JSONL
# -----------------------------------------------------------------------

class InstructionDataset(Dataset):
    """
    JSONL dataset for supervised fine-tuning (SFT).

    Each line is a JSON object with either:
    - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    - {"instruction": "...", "output": "..."}  (Alpaca format)

    Labels are -100 for prompt tokens (don't compute loss on them),
    and actual token IDs for completion tokens (train the model to generate these).
    This is the standard instruction-tuning approach.
    """

    def __init__(self, jsonl_file: str, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        self._load(jsonl_file)

    def _load(self, path: str):
        print(f"Loading instruction dataset from {path}...")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)
        print(f"Loaded {len(self.samples):,} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Support both ChatML messages and Alpaca format
        if "messages" in sample:
            ids = self.tokenizer.encode_chat(sample["messages"], add_generation_prompt=False)
            # Mask prompt tokens: find the last assistant turn start
            labels = _mask_prompt_tokens(ids, self.tokenizer)
        else:
            # Alpaca format
            prompt = self._alpaca_prompt(sample)
            completion = sample.get("output", sample.get("response", ""))
            prompt_ids = self.tokenizer.encode(prompt)
            completion_ids = self.tokenizer.encode(completion, add_eos=True)
            ids = prompt_ids + completion_ids
            # Mask out prompt from labels
            labels = [-100] * len(prompt_ids) + completion_ids

        # Truncate to max_seq_len
        ids = ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(ids)
        pad_id = self.tokenizer.pad_id
        ids = ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _alpaca_prompt(self, sample: dict) -> str:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _mask_prompt_tokens(ids: List[int], tokenizer) -> List[int]:
    """
    For ChatML formatted sequences:
    mask everything except the assistant's response tokens.
    """
    labels = [-100] * len(ids)
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id

    i = 0
    while i < len(ids):
        if ids[i] == im_start:
            # Find the role
            j = i + 1
            role_tokens = []
            while j < len(ids) and ids[j] != im_end:
                role_tokens.append(ids[j])
                j += 1
            role_text = tokenizer.decode(role_tokens, skip_special=True).strip()

            if role_text.startswith("assistant"):
                # Unmask from after "assistant\n" to im_end
                # Find the actual content start (after the newline)
                content_start = i + 1
                while content_start < len(ids) and tokenizer.decode([ids[content_start]]).strip() in ("assistant", ""):
                    content_start += 1
                for k in range(content_start, min(j + 1, len(ids))):
                    labels[k] = ids[k]
            i = j + 1
        else:
            i += 1

    return labels


# -----------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------

def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,      # 0 = main process (safe on Windows; increase on Linux)
    shuffle: bool = True,
    pin_memory: bool = False,  # Only useful with CUDA
) -> DataLoader:
    import torch
    pin = pin_memory and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=False,
    )
