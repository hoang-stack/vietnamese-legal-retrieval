"""Zalo AI 2021 Legal Text Retrieval, BEIR-formatted copy on Hugging Face (MIT licence).

61,425 articles of Vietnamese legal documents (decrees, circulars, laws), citizens' questions,
and for each question the article(s) that answer it.

The published train and test question sets share 24 questions. They are removed from the
training side here, so that tuning and fine-tuning never see a test question.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn/resolve/main"
FILES = {
    "corpus.parquet": (
        "corpus/test-00000-of-00001.parquet",
        "8cbbb5ee5603970a54e3608b71a50715079750457482948a4ba73f86c8f12fcf",
    ),
    "queries.jsonl": (
        "queries.jsonl",
        "7cf566b867fbd5cd6428058dea1115593c08c4b0ce0d7b0b00d6b40ae24ee509",
    ),
    "qrels_train.jsonl": (
        "qrels/train.jsonl",
        "8f21b5bb07a34bd67c6df1c31435feab44986c8be717854f92291faa67329b1f",
    ),
    "qrels_test.jsonl": (
        "qrels/test.jsonl",
        "ba46b40e07f591409a62c2b6a4e67cb5a0357ef27adb0f89bf8ce68ec723a747",
    ),
}


def download(data_dir: str | Path = "data") -> Path:
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, (remote, digest) in FILES.items():
        path = raw / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        with urllib.request.urlopen(f"{BASE}/{remote}", timeout=300) as resp:
            payload = resp.read()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError(f"checksum mismatch for {remote}")
        path.write_bytes(payload)
    return raw


@dataclass
class Dataset:
    doc_ids: list[str]
    doc_texts: list[str]  # "title\ntext"
    queries: dict[str, str]  # query id -> question
    qrels: dict[str, dict[str, set[str]]]  # split -> query id -> relevant doc ids

    def split_queries(self, split: str) -> tuple[list[str], list[str]]:
        ids = sorted(self.qrels[split])
        return ids, [self.queries[q] for q in ids]


def _qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["score"] > 0:
            out.setdefault(r["query-id"], set()).add(r["corpus-id"])
    return out


def load(data_dir: str | Path = "data", dev_share: float = 0.2, seed: int = 0) -> Dataset:
    """Corpus, queries and three disjoint question sets: train, dev (for tuning) and test."""
    raw = download(data_dir)
    corpus = pd.read_parquet(raw / "corpus.parquet")
    queries = {}
    for line in (raw / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        queries[r["_id"]] = r["text"]
    train, test = _qrels(raw / "qrels_train.jsonl"), _qrels(raw / "qrels_test.jsonl")
    train = {q: d for q, d in train.items() if q not in test}  # drop the 24 shared questions
    ids = sorted(train)
    rng = np.random.default_rng(seed)
    dev_ids = set(rng.choice(ids, int(round(dev_share * len(ids))), replace=False).tolist())
    texts = (corpus["title"].fillna("") + "\n" + corpus["text"].fillna("")).tolist()
    return Dataset(
        doc_ids=corpus["id"].astype(str).tolist(),
        doc_texts=texts,
        queries=queries,
        qrels={
            "train": {q: train[q] for q in ids if q not in dev_ids},
            "dev": {q: train[q] for q in ids if q in dev_ids},
            "test": test,
        },
    )
