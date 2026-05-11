"""Smoke test for everything that doesn't require torch/transformers.

Run:  python tests/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("1. preprocessing.clean")
print("=" * 60)
from preprocessing.clean import clean_article, split_sentences, is_record_valid

raw = """
Introduction
This paper studies transformer-based summarization [1, 2]
of scientific articles (Smith et al., 2019). We propose a new method.
\\begin{equation} E = mc^2 \\end{equation}
Figure 1: An example diagram of our pipeline.
Our results show that fine-tuning improves ROUGE by 3.2 points [12].

Methods
We fine-tune $BERT_{base}$ on PubMed with learning rate 2e-5.
We use 3000 samples for training and 400 for validation.
The optimizer is AdamW with weight decay.

Results
Our model achieves ROUGE-1 of 42.1 and ROUGE-2 of 18.5 on the test set.
This is a 3.2 point improvement over the previous baseline.

References
[1] Vaswani et al. Attention is all you need. NeurIPS 2017.
[2] Devlin et al. BERT. NAACL 2019.
"""
cleaned = clean_article(raw)
print("CLEANED:\n", cleaned)
assert "[1, 2]" not in cleaned, "inline citation [1,2] not removed"
assert "Smith et al., 2019" not in cleaned, "author citation not removed"
assert "References" not in cleaned, "references section not stripped"
assert "Figure 1" not in cleaned, "figure caption not removed"
assert "$BERT_{base}$" not in cleaned, "inline latex not removed"
assert "\\begin{equation}" not in cleaned, "equation block not removed"

sents = split_sentences(cleaned)
print(f"\n{len(sents)} sentences:")
for i, s in enumerate(sents):
    print(f"  [{i}] {s}")
assert len(sents) >= 3
print("PASS\n")

print("=" * 60)
print("2. preprocessing.oracle (greedy ORACLE labels)")
print("=" * 60)
from preprocessing.oracle import greedy_oracle
abstract = "We fine-tune BERT on PubMed and achieve ROUGE-1 of 42.1, improving over the baseline."
selected, labels = greedy_oracle(sents, abstract, max_sents=3)
print(f"selected indices: {selected}")
print(f"labels:           {labels}")
print("selected sentences:")
for i in selected:
    print(f"  - {sents[i]}")
assert sum(labels) == len(selected) >= 1
print("PASS\n")

print("=" * 60)
print("3. models.baselines (TF-IDF + TextRank)")
print("=" * 60)
from models.baselines import tfidf_summary, textrank_summary
tfidf_out = tfidf_summary(cleaned, num_sentences=2)
tr_out = textrank_summary(cleaned, num_sentences=2)
print("TF-IDF :", tfidf_out)
print("TextRank:", tr_out)
assert tfidf_out and tr_out
print("PASS\n")

print("=" * 60)
print("4. evaluation.rouge_eval")
print("=" * 60)
from evaluation.rouge_eval import compute_rouge
scores = compute_rouge([tfidf_out, tr_out], [abstract, abstract])
print("ROUGE:", scores)
assert "rouge1" in scores and scores["rouge1"] >= 0
print("PASS\n")

print("=" * 60)
print("5. Static import check for torch-dependent modules")
print("=" * 60)
import ast
for path in ("models/bertsum.py", "models/abstractive.py", "models/keywords.py", "ui/app.py"):
    full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    ast.parse(open(full, encoding="utf-8").read())
    print(f"  {path}: syntax OK")
print("PASS\n")

print("ALL SMOKE TESTS PASSED")
