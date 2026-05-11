"""Greedy ORACLE sentence-label builder for extractive summarization (BERTSUM).

For each (article_sentences, abstract) pair, greedily picks up to ``max_sents``
sentences whose concatenation maximizes ROUGE-1 + ROUGE-2 F1 against the
gold abstract. The chosen sentence indices get label 1, all others 0.

Reference: Nallapati et al. 2017 (SummaRuNNer) / Liu 2019 (BERTSUM).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from rouge_score import rouge_scorer


_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)


def _combined_rouge(candidate: str, reference: str) -> float:
    s = _SCORER.score(reference, candidate)
    return s["rouge1"].fmeasure + s["rouge2"].fmeasure


def greedy_oracle(sentences: Sequence[str], abstract: str,
                  max_sents: int = 7) -> Tuple[List[int], List[int]]:
    """Return (selected_indices, binary_labels) for a single article."""
    n = len(sentences)
    labels = [0] * n
    if n == 0 or not abstract.strip():
        return [], labels

    selected: List[int] = []
    best_score = 0.0
    remaining = set(range(n))

    for _ in range(min(max_sents, n)):
        best_idx = -1
        best_new_score = best_score
        for idx in remaining:
            trial = selected + [idx]
            trial.sort()
            cand = " ".join(sentences[i] for i in trial)
            score = _combined_rouge(cand, abstract)
            if score > best_new_score:
                best_new_score = score
                best_idx = idx
        if best_idx == -1:
            break  # no improvement; stop early
        selected.append(best_idx)
        remaining.discard(best_idx)
        best_score = best_new_score

    for i in selected:
        labels[i] = 1
    return sorted(selected), labels
