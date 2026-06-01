"""
Utilities for loading the sentence_pairs_large TSV parallel corpus into the
format expected by d2l.Seq2Seq models.
"""

from __future__ import annotations

import io
import os
import re
from collections import Counter
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


_UNICODE_SPACES = r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]"
LINE_REGEX = re.compile(r"^\s*(\d+)\s+(.*?)\s+(\d+)\s+(.*?)\s*$", flags=re.UNICODE)


def normalize_spaces(text: str) -> str:
    text = re.sub(_UNICODE_SPACES, " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_line(raw: str) -> Optional[Tuple[int, str, int, str]]:
    line = normalize_spaces(raw)
    match = LINE_REGEX.match(line)
    if not match:
        return None
    es_id = int(match.group(1))
    es_text = match.group(2).strip()
    en_id = int(match.group(3))
    en_text = match.group(4).strip()
    if not es_text or not en_text:
        return None
    return es_id, es_text, en_id, en_text


def tokenize(text: str) -> List[str]:
    return normalize_spaces(text).lower().split()


class Vocab:
    def __init__(self, tokens: List[List[str]], min_freq: int = 1):
        reserved = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.idx_to_token = list(reserved)
        self.token_to_idx = {tok: idx for idx, tok in enumerate(self.idx_to_token)}
        counter = Counter(token for line in tokens for token in line)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)
        self.pad_id = self.token_to_idx["<pad>"]
        self.bos_id = self.token_to_idx["<bos>"]
        self.eos_id = self.token_to_idx["<eos>"]
        self.unk_id = self.token_to_idx["<unk>"]

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.unk_id)

    def to_tokens(self, ids: List[int]) -> List[str]:
        return [
            self.idx_to_token[i] if 0 <= i < len(self.idx_to_token) else "<unk>"
            for i in ids
        ]


def encode_with_bos_eos(tokens: List[str], vocab: Vocab) -> List[int]:
    return [vocab.bos_id] + [vocab[token] for token in tokens] + [vocab.eos_id]


def pad_or_trim(ids: List[int], num_steps: int, pad_id: int) -> Tuple[List[int], int]:
    valid_len = min(len(ids), num_steps)
    if len(ids) < num_steps:
        ids = ids + [pad_id] * (num_steps - len(ids))
    else:
        ids = ids[:num_steps]
    return ids, valid_len


class SpanishEnglishPairs(Dataset):
    def __init__(
        self,
        lines: List[str],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        num_steps: int,
        drop_invalid: bool = True,
        keep_first_translation_only: bool = False,
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        seen_src = set()
        for raw in lines:
            parsed = parse_line(raw)
            if not parsed:
                if drop_invalid:
                    continue
                raise ValueError(f"Cannot parse line: {raw!r}")
            es_id, es_text, _, en_text = parsed
            if keep_first_translation_only:
                if es_id in seen_src:
                    continue
                seen_src.add(es_id)
            src_tokens = tokenize(es_text)
            tgt_tokens = tokenize(en_text)
            src_ids = encode_with_bos_eos(src_tokens, src_vocab)
            tgt_ids = encode_with_bos_eos(tgt_tokens, tgt_vocab)
            src_ids, src_len = pad_or_trim(src_ids, num_steps, src_vocab.pad_id)
            tgt_ids, _ = pad_or_trim(tgt_ids, num_steps, tgt_vocab.pad_id)
            decoder_inputs = torch.tensor(tgt_ids[:-1], dtype=torch.long)
            decoder_labels = torch.tensor(tgt_ids[1:], dtype=torch.long)
            self.samples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    decoder_inputs,
                    torch.tensor(src_len, dtype=torch.long),
                    decoder_labels,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class TSVSeq2SeqData:
    def __init__(
        self,
        path: str,
        batch_size: int = 64,
        num_steps: int = 50,
        min_freq: int = 2,
        val_frac: float = 0.05,
        test_frac: float = 0.0,
        seed: int = 42,
        keep_first_translation_only: bool = False,
        sample_percent: float = 1.0,
    ):
        self.batch_size = batch_size
        self.num_steps = num_steps
        with io.open(os.path.expanduser(path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        rng = torch.Generator().manual_seed(seed)
        idxs = torch.randperm(len(lines), generator=rng).tolist()
        if not (0 < sample_percent <= 1):
            raise ValueError("sample_percent must be in (0, 1].")
        sample_size = max(1, int(len(idxs) * sample_percent))
        idxs = idxs[:sample_size]
        n_total = len(idxs)
        n_test = int(n_total * test_frac)
        n_val = int(n_total * val_frac)
        n_train = n_total - n_val - n_test
        train_lines = [lines[i] for i in idxs[:n_train]]
        val_lines = [lines[i] for i in idxs[n_train : n_train + n_val]]
        test_lines = [lines[i] for i in idxs[n_train + n_val :]] if n_test else []
        src_tokens, tgt_tokens = [], []
        seen_src = set()
        for raw in train_lines:
            parsed = parse_line(raw)
            if not parsed:
                continue
            es_id, es_text, _, en_text = parsed
            if keep_first_translation_only:
                if es_id in seen_src:
                    continue
                seen_src.add(es_id)
            src_tokens.append(tokenize(es_text))
            tgt_tokens.append(tokenize(en_text))
        self.src_vocab = Vocab(src_tokens, min_freq=min_freq)
        self.tgt_vocab = Vocab(tgt_tokens, min_freq=min_freq)
        self.train_ds = SpanishEnglishPairs(
            train_lines,
            self.src_vocab,
            self.tgt_vocab,
            num_steps,
            keep_first_translation_only=keep_first_translation_only,
        )
        self.val_ds = (
            SpanishEnglishPairs(
                val_lines,
                self.src_vocab,
                self.tgt_vocab,
                num_steps,
                keep_first_translation_only=keep_first_translation_only,
            )
            if val_lines
            else None
        )
        self.test_ds = (
            SpanishEnglishPairs(
                test_lines,
                self.src_vocab,
                self.tgt_vocab,
                num_steps,
                keep_first_translation_only=keep_first_translation_only,
            )
            if test_lines
            else None
        )

    @staticmethod
    def _collate(batch):
        src, tgt_in, src_len, tgt_label = zip(*batch)
        return (
            torch.stack(src),
            torch.stack(tgt_in),
            torch.stack(src_len),
            torch.stack(tgt_label),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=self._collate,
        )

    def build(self, src_sentences: List[str], tgt_sentences: List[str]):
        if len(src_sentences) != len(tgt_sentences):
            raise ValueError("src_sentences and tgt_sentences must be the same length")
        batch_src, batch_tgt, batch_src_len, batch_tgt_len = [], [], [], []
        for src_text, tgt_text in zip(src_sentences, tgt_sentences):
            src_ids, src_len = pad_or_trim(
                encode_with_bos_eos(tokenize(src_text), self.src_vocab),
                self.num_steps,
                self.src_vocab.pad_id,
            )
            tgt_ids, tgt_len = pad_or_trim(
                encode_with_bos_eos(tokenize(tgt_text), self.tgt_vocab),
                self.num_steps,
                self.tgt_vocab.pad_id,
            )
            batch_src.append(torch.tensor(src_ids, dtype=torch.long))
            batch_tgt.append(torch.tensor(tgt_ids[:-1], dtype=torch.long))
            batch_src_len.append(torch.tensor(src_len, dtype=torch.long))
            batch_tgt_len.append(torch.tensor(tgt_len - 1, dtype=torch.long))
        return (
            torch.stack(batch_src),
            torch.stack(batch_tgt),
            torch.stack(batch_src_len),
            torch.stack(batch_tgt_len),
        )
