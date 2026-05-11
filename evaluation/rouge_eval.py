"""Compute ROUGE-1/2/L given predictions and references.

Usage (programmatic):
    from evaluation.rouge_eval import compute_rouge
    scores = compute_rouge(predictions, references)

Usage (CLI):
    python -m evaluation.rouge_eval --preds preds.txt --refs refs.txt
"""
from __future__ import annotations

import argparse
from typing import Dict, Sequence

from rouge_score import rouge_scorer


def compute_rouge(predictions: Sequence[str],
                  references: Sequence[str]) -> Dict[str, float]:
    assert len(predictions) == len(references), "preds/refs length mismatch"
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        for k in agg:
            agg[k] += s[k].fmeasure
    n = max(len(predictions), 1)
    return {k: round(v / n * 100, 2) for k, v in agg.items()}


def _read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--refs", required=True)
    args = ap.parse_args()
    scores = compute_rouge(_read_lines(args.preds), _read_lines(args.refs))
    print(scores)


if __name__ == "__main__":
    main()
