"""KeyBERT-based keyword extraction (lazy-loaded)."""
from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def _get_kw_model():
    from keybert import KeyBERT
    return KeyBERT(model="all-MiniLM-L6-v2")


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    kw = _get_kw_model()
    results = kw.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
    )
    return [k for k, _ in results]
