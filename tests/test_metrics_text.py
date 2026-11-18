import numpy as np
import pytest

from vilegal import metrics
from vilegal.fusion import reciprocal_rank_fusion
from vilegal.text import syllables, tokens


def test_tokens_add_syllable_bigrams_and_normalise():
    assert syllables("Giấy  PHÉP, lái-xe!") == ["giấy", "phép", "lái", "xe"]
    assert tokens("giấy phép lái") == ["giấy", "phép", "lái", "giấy_phép", "phép_lái"]
    decomposed = "Việt"
    assert syllables(decomposed) == ["việt"]


def test_ranking_metrics_hand_example():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r"]]
    gold = [{"b"}, {"x", "z"}, {"nope"}]
    m = metrics.evaluate(ranked, gold, ks=(1, 3))
    np.testing.assert_allclose(m["recall@1"], [0, 0.5, 0])
    np.testing.assert_allclose(m["recall@3"], [1, 1, 0])
    np.testing.assert_allclose(m["mrr@10"], [0.5, 1, 0])
    ideal = 1 + 1 / np.log2(3)
    np.testing.assert_allclose(m["ndcg@10"], [1 / np.log2(3), (1 + 1 / np.log2(4)) / ideal, 0])


def test_bootstrap_and_paired_test():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    lo, hi = metrics.bootstrap_ci(a)
    assert lo < a.mean() < hi
    res = metrics.paired_bootstrap(a + 0.1, a)
    assert res["diff"] == pytest.approx(0.1) and res["p_value"] < 0.01


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert fused[0] == "b"  # ranked well by both
    assert set(fused) == {"a", "b", "c", "d"}
    assert reciprocal_rank_fusion([["a", "b"]]) == ["a", "b"]
