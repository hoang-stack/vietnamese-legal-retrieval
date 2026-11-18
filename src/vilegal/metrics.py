"""Ranking metrics per query, plus bootstrap confidence intervals over queries."""

from __future__ import annotations

import numpy as np


def relevance_matrix(ranked_doc_ids: list[list[str]], relevant: list[set[str]]) -> np.ndarray:
    """rel[q, r] = 1 if the document at rank r is relevant to query q."""
    k = max((len(r) for r in ranked_doc_ids), default=0)
    rel = np.zeros((len(ranked_doc_ids), k), dtype=bool)
    for q, (docs, gold) in enumerate(zip(ranked_doc_ids, relevant, strict=True)):
        rel[q, : len(docs)] = [d in gold for d in docs]
    return rel


def recall_at_k(rel: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    return rel[:, :k].sum(axis=1) / n_relevant


def mrr_at_k(rel: np.ndarray, k: int = 10) -> np.ndarray:
    hit = rel[:, :k]
    first = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), 1.0 / (first + 1), 0.0)


def ndcg_at_k(rel: np.ndarray, n_relevant: np.ndarray, k: int = 10) -> np.ndarray:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel[:, :k] * disc[: rel[:, :k].shape[1]]).sum(axis=1)
    ideal = np.cumsum(disc)[np.minimum(n_relevant, k) - 1]
    return dcg / ideal


def evaluate(ranked_doc_ids, relevant, ks=(1, 5, 10, 100)) -> dict[str, np.ndarray]:
    rel = relevance_matrix(ranked_doc_ids, relevant)
    n_rel = np.array([len(g) for g in relevant])
    out = {f"recall@{k}": recall_at_k(rel, n_rel, k) for k in ks}
    out["mrr@10"] = mrr_at_k(rel, 10)
    out["ndcg@10"] = ndcg_at_k(rel, n_rel, 10)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = a - b
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p)),
    }
