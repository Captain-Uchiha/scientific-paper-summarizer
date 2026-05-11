"""Text cleaning for scientific articles (PubMed / ArXiv style).

Removes references sections, inline citations, LaTeX equations, figure/table
captions, and normalizes whitespace + unicode.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:  # pragma: no cover
    nltk = None
    sent_tokenize = None


# --- regex patterns -----------------------------------------------------------
_INLINE_CITATION_NUM = re.compile(r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]")
_INLINE_CITATION_AUTHOR = re.compile(
    r"\(\s*[A-Z][A-Za-z\-']+(?:\s+et\s+al\.?)?(?:\s*,)?\s*\d{4}[a-z]?\s*\)"
)
_LATEX_INLINE = re.compile(r"\$[^$]{0,500}\$")
_LATEX_BLOCK = re.compile(
    r"\\begin\{(?:equation|align|gather|multline)\*?\}.*?"
    r"\\end\{(?:equation|align|gather|multline)\*?\}",
    re.DOTALL,
)
_FIG_TABLE_LINE = re.compile(
    r"^(?:figure|fig\.?|table|tbl\.?)\s*\d+[:.\s].*?$",
    re.IGNORECASE | re.MULTILINE,
)
_REFERENCES_HEADER = re.compile(
    r"\n\s*(references|bibliography|acknowledgements?|acknowledgments?)\s*\n",
    re.IGNORECASE,
)
_MULTI_WS = re.compile(r"\s+")


def _ensure_punkt() -> None:
    if nltk is None:
        return
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # pragma: no cover - best effort
                pass


def strip_references(text: str) -> str:
    """Cut everything from the final 'references'/'bibliography' header onward."""
    matches = list(_REFERENCES_HEADER.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    return text[:cut]


def clean_article(text: str, lowercase: bool = False) -> str:
    """Apply the full cleaning pipeline to a single article string."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = strip_references(text)
    text = _LATEX_BLOCK.sub(" ", text)
    text = _LATEX_INLINE.sub(" ", text)
    text = _INLINE_CITATION_NUM.sub(" ", text)
    text = _INLINE_CITATION_AUTHOR.sub(" ", text)
    text = _FIG_TABLE_LINE.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def split_sentences(text: str) -> List[str]:
    """Sentence-tokenize using NLTK punkt; falls back to a naive split."""
    if sent_tokenize is None:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    _ensure_punkt()
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def is_record_valid(article: str, abstract: str,
                    min_article_words: int = 200,
                    min_abstract_words: int = 30) -> bool:
    """Filter out records that are too short to be useful for training."""
    return (
        len(article.split()) >= min_article_words
        and len(abstract.split()) >= min_abstract_words
    )


def clean_record(article: str, abstract: str) -> dict:
    """Clean an (article, abstract) pair and return a dict ready for JSONL."""
    art = clean_article(article)
    abs_ = clean_article(abstract)
    return {
        "input_text": art,
        "target_text": abs_,
        "sentences": split_sentences(art),
    }
