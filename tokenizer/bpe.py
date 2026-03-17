"""
Byte Pair Encoding (BPE) Tokenizer — built from scratch.
Paper: https://arxiv.org/abs/1508.07909

This is a clean, fast implementation compatible with the tiktoken/GPT-4 style:
- Byte-level BPE (handles any Unicode out of the box)
- Special tokens support (<bos>, <eos>, <pad>, <unk>, <|im_start|>, <|im_end|>)
- Can train on a text corpus or load a pre-trained vocabulary

ChatML format for conversations:
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  Hello!<|im_end|>
  <|im_start|>assistant
  Hi there!<|im_end|>
"""
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple


# -----------------------------------------------------------------------
# Special tokens
# -----------------------------------------------------------------------

SPECIAL_TOKENS = {
    "<|bos|>":       0,   # Beginning of sequence
    "<|eos|>":       1,   # End of sequence
    "<|pad|>":       2,   # Padding
    "<|unk|>":       3,   # Unknown
    "<|im_start|>":  4,   # Chat message start (ChatML)
    "<|im_end|>":    5,   # Chat message end (ChatML)
    "<|sep|>":       6,   # Separator
    "<|mask|>":      7,   # Mask token (for future masked LM)
}


# -----------------------------------------------------------------------
# Byte-level BPE Tokenizer
# -----------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Training steps:
    1. Start with 256 byte tokens (covers all text losslessly)
    2. Repeatedly find the most frequent adjacent pair
    3. Merge that pair into a new token
    4. Repeat until vocabulary is full

    The byte-level approach means we never see UNK tokens for normal text.
    """

    # GPT-2 style regex pattern: splits on whitespace/punctuation boundaries
    # This is the key to good tokenization — don't merge across word boundaries
    _SPLIT_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        re.UNICODE
    )

    def __init__(self):
        # vocab: token_id → bytes
        self.vocab: Dict[int, bytes] = {}
        # merges: (pair of token_ids) → merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # reverse vocab: bytes → token_id
        self._vocab_inv: Dict[bytes, int] = {}
        # special tokens
        self.special_tokens: Dict[str, int] = {}
        self._special_inv: Dict[int, str] = {}

        self._initialized = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int = 32000, verbose: bool = True) -> None:
        """
        Train BPE on a text corpus.

        Args:
            text: Training corpus (can be very large)
            vocab_size: Target vocabulary size (must be >= 256 + len(special_tokens))
            verbose: Print progress
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (byte vocab)"
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert num_merges >= 0, f"vocab_size too small for {len(SPECIAL_TOKENS)} special tokens"

        # Step 1: Initialize with 256 byte tokens
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Step 2: Register special tokens (after byte vocab)
        self.special_tokens = {}
        for token, _ in SPECIAL_TOKENS.items():
            idx = 256 + len(self.special_tokens)
            self.special_tokens[token] = idx
            self._special_inv[idx] = token

        # Step 3: Split text into initial byte sequences
        if verbose:
            print(f"Tokenizer training: target vocab={vocab_size}, merges={num_merges}")

        # Tokenize into words (segments)
        words = self._SPLIT_PATTERN.findall(text)
        # Encode each word as list of byte-token IDs
        word_freqs: Dict[Tuple[int, ...], int] = defaultdict(int)
        for word in words:
            byte_seq = tuple(b for b in word.encode("utf-8"))
            word_freqs[byte_seq] += 1

        # Convert to mutable list of tokens per word
        vocab_sequences: Dict[Tuple[int, ...], int] = dict(word_freqs)

        # Step 4: Iteratively find and apply merges
        self.merges = {}
        next_id = 256 + len(SPECIAL_TOKENS)

        for merge_idx in range(num_merges):
            # Count all adjacent pairs
            pair_counts = self._count_pairs(vocab_sequences)
            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            if pair_counts[best_pair] < 2:
                break

            # Merge
            new_id = next_id + merge_idx
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to all sequences
            vocab_sequences = self._apply_merge(vocab_sequences, best_pair, new_id)

            if verbose and (merge_idx + 1) % 1000 == 0:
                print(f"  Merge {merge_idx+1}/{num_merges} — vocab size: {len(self.vocab)}")

        # Build reverse vocab
        self._vocab_inv = {v: k for k, v in self.vocab.items()}
        self._initialized = True

        if verbose:
            print(f"Training complete. Vocabulary: {len(self.vocab)} tokens.")

    def _count_pairs(
        self, sequences: Dict[Tuple[int, ...], int]
    ) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for seq, freq in sequences.items():
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += freq
        return counts

    def _apply_merge(
        self,
        sequences: Dict[Tuple[int, ...], int],
        pair: Tuple[int, int],
        new_id: int,
    ) -> Dict[Tuple[int, ...], int]:
        new_sequences = {}
        a, b = pair
        for seq, freq in sequences.items():
            new_seq = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i+1] == b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences[tuple(new_seq)] = freq
        return new_sequences

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        self._ensure_initialized()

        ids = []
        if add_bos:
            ids.append(self.bos_id)

        # Handle special tokens: split on them first
        parts = self._split_on_special_tokens(text)
        for part_text, is_special in parts:
            if is_special:
                ids.append(self.special_tokens[part_text])
            else:
                ids.extend(self._encode_chunk(part_text))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def _split_on_special_tokens(self, text: str) -> List[Tuple[str, bool]]:
        """Split text, marking special tokens."""
        if not self.special_tokens:
            return [(text, False)]

        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        parts = re.split(pattern, text)
        return [(p, p in self.special_tokens) for p in parts if p]

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode a plain text chunk (no special tokens) using BPE."""
        words = self._SPLIT_PATTERN.findall(text)
        ids = []
        for word in words:
            # Start with bytes
            tokens = list(word.encode("utf-8"))
            # Apply merges in learned order
            while len(tokens) > 1:
                # Find the highest-priority (earliest learned) merge
                best = None
                best_rank = float("inf")
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    rank = self.merges.get(pair, float("inf"))
                    if rank < best_rank:
                        best_rank = rank
                        best = (i, pair)
                if best is None or best_rank == float("inf"):
                    break
                i, pair = best
                new_id = self.merges[pair]
                tokens = tokens[:i] + [new_id] + tokens[i+2:]
            ids.extend(tokens)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        self._ensure_initialized()
        parts = []
        for tid in ids:
            if tid in self._special_inv:
                if not skip_special:
                    parts.append(self._special_inv[tid].encode("utf-8"))
            elif tid in self.vocab:
                parts.append(self.vocab[tid])
            # else: unknown token, skip
        return b"".join(parts).decode("utf-8", errors="replace")

    def encode_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> List[int]:
        """
        Encode a chat conversation using ChatML format.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            add_generation_prompt: Append <|im_start|>assistant\\n to prompt generation

        Returns:
            Token IDs
        """
        self._ensure_initialized()
        ids = [self.bos_id]
        im_start = self.special_tokens["<|im_start|>"]
        im_end = self.special_tokens["<|im_end|>"]
        nl_id = self.encode("\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            ids.append(im_start)
            ids.extend(self.encode(role))
            ids.extend(nl_id)
            ids.extend(self.encode(content))
            ids.append(im_end)
            ids.extend(nl_id)

        if add_generation_prompt:
            ids.append(im_start)
            ids.extend(self.encode("assistant"))
            ids.extend(nl_id)

        return ids

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": {f"{a},{b}": c for (a, b), c in self.merges.items()},
            "special_tokens": self.special_tokens,
        }
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from directory."""
        with open(os.path.join(path, "tokenizer.json")) as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.merges = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data["merges"].items()
        }
        tok.special_tokens = data["special_tokens"]
        tok._vocab_inv = {v: k for k, v in tok.vocab.items()}
        tok._special_inv = {v: k for k, v in tok.special_tokens.items()}
        tok._initialized = True
        return tok

    # ------------------------------------------------------------------
    # Properties & helpers
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def im_start_id(self) -> int:
        return self.special_tokens["<|im_start|>"]

    @property
    def im_end_id(self) -> int:
        return self.special_tokens["<|im_end|>"]

    def _ensure_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "Tokenizer not initialized. Call .train() or .load() first."
            )

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size})"


# -----------------------------------------------------------------------
# Shortcut: wrap tiktoken (optional, for fast inference with cl100k)
# -----------------------------------------------------------------------

class TiktokenWrapper:
    """
    Optional wrapper around tiktoken (cl100k_base = GPT-4 tokenizer).
    Use this when you want to skip training your own tokenizer.
    """

    def __init__(self, model: str = "cl100k_base"):
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(model)
            self.special_tokens = {
                "<|bos|>": self._enc.encode("<|endoftext|>")[0],
                "<|eos|>": self._enc.encode("<|endoftext|>")[0],
                "<|pad|>": 0,
                "<|im_start|>": self._enc.encode("<|im_start|>", allowed_special="all")[0],
                "<|im_end|>": self._enc.encode("<|im_end|>", allowed_special="all")[0],
            }
        except ImportError:
            raise ImportError("Install tiktoken: pip install tiktoken")

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._enc.encode(text, allowed_special="all")
        if add_bos:
            ids = [self.special_tokens["<|bos|>"]] + ids
        if add_eos:
            ids = ids + [self.special_tokens["<|eos|>"]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self._enc.decode(ids)
