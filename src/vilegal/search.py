"""A small search service over the corpus: BM25 ranking plus a best-matching snippet."""

from __future__ import annotations

from dataclasses import dataclass

from vilegal.bm25 import BM25
from vilegal.text import normalize, syllables


@dataclass
class Hit:
    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float


def best_snippet(text: str, query: str, width: int = 40) -> str:
    """The ``width``-word window of ``text`` that contains the most query syllables."""
    words = text.split()
    if len(words) <= width:
        return text
    q = set(syllables(query))
    hits = [int(normalize(w).strip(".,;:()\"'") in q) for w in words]
    window = sum(hits[:width])
    best, best_start = window, 0
    for start in range(1, len(words) - width + 1):
        window += hits[start + width - 1] - hits[start - 1]
        if window > best:
            best, best_start = window, start
    prefix = "… " if best_start else ""
    suffix = " …" if best_start + width < len(words) else ""
    return prefix + " ".join(words[best_start : best_start + width]) + suffix


class SearchEngine:
    def __init__(self, doc_ids: list[str], doc_texts: list[str], k1: float = 0.9, b: float = 0.75):
        self.doc_ids, self.doc_texts = doc_ids, doc_texts
        self.bm25 = BM25(k1, b).fit(doc_texts)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        idx, scores = self.bm25.search([query], k=k)
        hits = []
        for r, (i, s) in enumerate(zip(idx[0], scores[0], strict=True), start=1):
            if s <= 0:
                break
            title, _, body = self.doc_texts[i].partition("\n")
            hits.append(Hit(r, self.doc_ids[i], title, best_snippet(body, query), float(s)))
        return hits
