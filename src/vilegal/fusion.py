"""Combining rankings from different retrievers."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """RRF (Cormack et al., SIGIR 2009): score(d) = sum over rankings of 1 / (k + rank(d)).

    Uses ranks only, so retrievers with incomparable score scales (BM25 vs cosine) can be mixed
    without any calibration. Ties keep the order of first appearance.
    """
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda d: -score[d])
