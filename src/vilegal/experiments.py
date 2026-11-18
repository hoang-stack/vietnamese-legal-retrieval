"""Evaluate every retrieval system with the same code: BM25 variants (CPU, here) and the
neural systems whose rankings were produced on Kaggle (kaggle/dense_retrieval.py)."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np

from vilegal import data, metrics
from vilegal.bm25 import BM25
from vilegal.text import tokens

log = logging.getLogger("vilegal")
METRICS = ("mrr@10", "ndcg@10", "recall@1", "recall@5", "recall@10", "recall@100")


class Recorder:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.per_query: dict[str, np.ndarray] = {}

    def add(self, stage: str, system: str, split: str, ranked, gold, **extra) -> dict:
        m = metrics.evaluate(ranked, gold)
        row = {
            "stage": stage,
            "system": system,
            "split": split,
            "queries": len(gold),
            **extra,
        }
        for k in METRICS:
            row[k] = float(m[k].mean())
        if split == "test":
            rel = metrics.relevance_matrix(ranked, gold)
            n_rel = np.array([len(g) for g in gold])
            row["recall_curve"] = [
                float(metrics.recall_at_k(rel, n_rel, k).mean()) for k in range(1, 101)
            ]
            row["mrr@10_ci"] = metrics.bootstrap_ci(m["mrr@10"])
            row["recall@10_ci"] = metrics.bootstrap_ci(m["recall@10"])
            for k in METRICS:
                self.per_query[f"{system}|{k}"] = m[k].astype(np.float32)
        self.rows.append(row)
        log.info(
            "%-34s %-4s mrr@10 %.4f  r@10 %.4f  r@100 %.4f",
            system,
            split,
            row["mrr@10"],
            row["recall@10"],
            row["recall@100"],
        )
        return row

    def save(self) -> None:
        (self.out / "runs.json").write_text(json.dumps(self.rows, indent=1, ensure_ascii=False))
        if self.per_query:
            np.savez_compressed(self.out / "per_query_test.npz", **self.per_query)


def bm25_stage(ds, rec: Recorder) -> None:
    dev_q, dev_t = ds.split_queries("dev")
    dev_gold = [ds.qrels["dev"][q] for q in dev_q]
    test_q, test_t = ds.split_queries("test")
    test_gold = [ds.qrels["test"][q] for q in test_q]

    def ranked(bm, texts):
        idx, _ = bm.search(texts, k=100)
        return [[ds.doc_ids[i] for i in row] for row in idx]

    for name, tok in (
        ("syllables", partial(tokens, bigrams=False)),
        ("syllables + bigrams", tokens),
    ):
        bm = BM25(tokenizer=tok).fit(ds.doc_texts)
        rec.add(
            "tokenization",
            f"BM25 ({name}, default k1/b)",
            "dev",
            ranked(bm, dev_t),
            dev_gold,
            vocab=len(bm.vocab),
        )
    best = (-1.0, None)
    for k1, b in itertools.product([0.6, 0.9, 1.2, 1.6, 2.0], [0.3, 0.5, 0.75, 0.9]):
        bm.set_params(k1, b)
        row = rec.add(
            "bm25_grid",
            f"BM25 k1={k1} b={b}",
            "dev",
            ranked(bm, dev_t),
            dev_gold,
            k1=k1,
            b=b,
        )
        best = max(best, (row["mrr@10"], (k1, b)), key=lambda x: x[0])
    k1, b = best[1]
    bm.set_params(k1, b)
    rec.add("final", "BM25", "test", ranked(bm, test_t), test_gold, k1=k1, b=b)
    bm.set_params(1.2, 0.75)
    rec.add("final", "BM25 (untuned)", "test", ranked(bm, test_t), test_gold)


def neural_stage(ds, rec: Recorder, kaggle_dir: Path) -> None:
    runs = json.loads((kaggle_dir / "runs.json").read_text())
    for system, by_split in runs.items():
        for split in ("dev", "test"):
            if split not in by_split or system == "BM25":
                continue
            qids = sorted(ds.qrels[split])
            ranked = [by_split[q] for q in qids]
            stage = "finetune_epochs" if "epoch" in system else "final"
            if stage == "final" or split == "dev":
                rec.add(stage, system, split, ranked, [ds.qrels[split][q] for q in qids])


def significance(rec: Recorder) -> None:
    pq = rec.per_query
    systems = sorted({k.split("|")[0] for k in pq})
    best = max(systems, key=lambda s: pq[f"{s}|mrr@10"].mean())
    out = {
        "best": best,
        "metric": "mrr@10",
        "vs": {
            s: metrics.paired_bootstrap(pq[f"{best}|mrr@10"], pq[f"{s}|mrr@10"])
            for s in systems
            if s != best
        },
    }
    (rec.out / "significance.json").write_text(json.dumps(out, indent=2))


def run(stages, data_dir="data", out_dir="results", kaggle_dir="kaggle/output/main") -> None:
    ds = data.load(data_dir)
    rec = Recorder(Path(out_dir))
    if "bm25" in stages:
        bm25_stage(ds, rec)
    if "neural" in stages:
        neural_stage(ds, rec, Path(kaggle_dir))
    rec.save()
    if rec.per_query:
        significance(rec)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stages", nargs="*", default=["bm25", "neural"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--kaggle-dir", default="kaggle/output/main")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(a.stages, a.data_dir, a.out, a.kaggle_dir)


if __name__ == "__main__":
    main()
