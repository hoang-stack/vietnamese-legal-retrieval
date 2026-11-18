"""Tokenisation for Vietnamese retrieval.

Vietnamese writes one syllable per space-separated token, and most words have two syllables
("pháp luật", "giấy phép"). Without a word segmenter, adjacent-syllable bigrams are a cheap
stand-in for words: "giấy phép lái xe" -> giấy, phép, lái, xe, giấy_phép, phép_lái, lái_xe.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[^\W_]+")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def syllables(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def tokens(text: str, bigrams: bool = True) -> list[str]:
    syl = syllables(text)
    if not bigrams:
        return syl
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:], strict=False)]
