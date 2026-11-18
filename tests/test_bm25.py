import math
from functools import partial

import numpy as np
import pytest

from vilegal.bm25 import BM25
from vilegal.text import tokens

DOCS = [
    "Giấy phép lái xe hạng A1 cấp cho người lái xe mô tô",
    "Mức phạt khi không mang giấy phép lái xe",
    "Thủ tục đăng ký kết hôn tại ủy ban nhân dân cấp xã",
    "Điều kiện kết hôn theo luật hôn nhân và gia đình",
]


def brute_bm25(query, docs, k1, b, tok):
    toks = [tok(d) for d in docs]
    avgdl = np.mean([len(t) for t in toks])
    n = len(docs)
    out = []
    for t in toks:
        s = 0.0
        for term in tok(query):
            df = sum(term in d for d in toks)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = t.count(term)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avgdl))
        out.append(s)
    return np.array(out)


@pytest.mark.parametrize("bigrams", [False, True])
@pytest.mark.parametrize(("k1", "b"), [(1.2, 0.75), (0.9, 0.3), (2.0, 1.0)])
def test_scores_match_the_formula(bigrams, k1, b):
    tok = partial(tokens, bigrams=bigrams)
    bm = BM25(k1, b, tokenizer=tok).fit(DOCS)
    for q in ["giấy phép lái xe", "kết hôn ở xã", "phạt"]:
        np.testing.assert_allclose(bm.scores([q])[0], brute_bm25(q, DOCS, k1, b, tok), rtol=1e-5)


def test_set_params_matches_a_fresh_fit():
    a = BM25(1.2, 0.75).fit(DOCS).set_params(0.5, 0.2)
    b = BM25(0.5, 0.2).fit(DOCS)
    np.testing.assert_allclose(a.scores(["kết hôn"]), b.scores(["kết hôn"]))


def test_search_ranks_the_obvious_document_first():
    bm = BM25().fit(DOCS)
    idx, sc = bm.search(["đăng ký kết hôn", "không mang giấy phép lái xe bị phạt"], k=2)
    assert idx[:, 0].tolist() == [2, 1]
    assert (np.diff(sc, axis=1) <= 0).all()


def test_unknown_query_terms_are_ignored():
    bm = BM25().fit(DOCS)
    assert bm.scores(["blockchain"]).sum() == 0
