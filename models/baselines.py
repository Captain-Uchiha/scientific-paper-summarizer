"""Classical non-trainable summarization baselines: TF-IDF and TextRank."""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing.clean import split_sentences


def tfidf_summary(text: str, num_sentences: int = 7) -> str:
    """Score each sentence by mean TF-IDF; return top-k in original order."""
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(sentences)
    # mean of non-zero entries per sentence
    sums = matrix.sum(axis=1).A1
    nnz = (matrix != 0).sum(axis=1).A1
    scores = np.divide(sums, np.maximum(nnz, 1))
    top_idx = sorted(np.argsort(scores)[-num_sentences:])
    return " ".join(sentences[i] for i in top_idx)


def textrank_summary(text: str, num_sentences: int = 7) -> str:
    """TextRank summary via the ``sumy`` library."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    picked = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in picked)
