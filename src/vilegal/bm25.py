"""Okapi BM25 over a sparse document-term matrix (Robertson & Zaragoza, 2009).

    score(q, d) = sum_{t in q} idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
                  where tf = tf(t, d)
    idf(t)      = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))        (Lucene's non-negative idf)

The whole collection is one CSR matrix of term frequencies. For given (k1, b) the BM25 weight of
every (document, term) cell is computed once, so scoring a batch of queries is a single sparse
matrix product: (docs x terms) @ (terms x queries).
"""

from __future__ import annotations

from array import array
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from vilegal.text import tokens


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, tokenizer: Callable = tokens):
        self.k1, self.b, self.tokenizer = k1, b, tokenizer

    def fit(self, docs: list[str]) -> BM25:
        vocab: dict[str, int] = {}
        # Typed arrays (4 bytes per entry) instead of lists of Python ints (~28 bytes each):
        # the bigram index has tens of millions of non-zeros.
        indptr, indices, data = array("q", [0]), array("i"), array("f")
        for doc in docs:
            counts: dict[int, int] = {}
            for tok in self.tokenizer(doc):
                j = vocab.setdefault(tok, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            indices.extend(counts)
            data.extend(counts.values())
            indptr.append(len(indices))
        self.vocab = vocab
        self.tf = sp.csr_matrix(
            (
                np.frombuffer(data, np.float32),
                np.frombuffer(indices, np.int32),
                np.frombuffer(indptr, np.int64),
            ),
            shape=(len(docs), len(vocab)),
        )
        n_docs = len(docs)
        df = np.bincount(self.tf.indices, minlength=len(vocab))
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.doc_len = np.asarray(self.tf.sum(axis=1)).ravel()
        self.set_params(self.k1, self.b)
        return self

    def set_params(self, k1: float, b: float) -> BM25:
        """Recompute the weight matrix for new (k1, b) without re-tokenising the corpus."""
        self.k1, self.b = k1, b
        norm = k1 * (1 - b + b * self.doc_len / max(self.doc_len.mean(), 1e-9))
        rows = np.repeat(np.arange(self.tf.shape[0]), np.diff(self.tf.indptr))
        tf = self.tf.data
        w = self.idf[self.tf.indices] * tf * (k1 + 1) / (tf + norm[rows])
        self.weights = sp.csr_matrix(
            (w.astype(np.float32), self.tf.indices, self.tf.indptr), shape=self.tf.shape
        )
        return self

    def query_matrix(self, queries: list[str]) -> sp.csr_matrix:
        """(terms x queries) indicator matrix; a repeated query term counts once per occurrence."""
        rows, cols = [], []
        for qi, q in enumerate(queries):
            for tok in self.tokenizer(q):
                j = self.vocab.get(tok)
                if j is not None:
                    rows.append(j)
                    cols.append(qi)
        data = np.ones(len(rows), np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(self.vocab), len(queries)))

    def scores(self, queries: list[str]) -> np.ndarray:
        """Dense (queries x docs) score matrix."""
        return np.asarray((self.weights @ self.query_matrix(queries)).T.todense())

    def search(self, queries: list[str], k: int = 100, batch: int = 256):
        """Top-k (doc indices, scores) per query, best first."""
        all_idx, all_sc = [], []
        for s in range(0, len(queries), batch):
            sc = self.scores(queries[s : s + batch])
            idx = np.argpartition(-sc, min(k, sc.shape[1] - 1), axis=1)[:, :k]
            top = np.take_along_axis(sc, idx, axis=1)
            order = np.argsort(-top, axis=1, kind="stable")
            all_idx.append(np.take_along_axis(idx, order, axis=1))
            all_sc.append(np.take_along_axis(top, order, axis=1))
        return np.vstack(all_idx), np.vstack(all_sc)
