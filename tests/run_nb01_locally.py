"""Run the notebook 01 logic (cells 4-7) locally to verify the pipeline.

Mirrors training/01_load_and_preprocess.ipynb but uses local paths and
skips Colab-only cells (git clone, drive mount).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_DIR, "dataset", "processed")
os.makedirs(OUT, exist_ok=True)

# --- Cell 4 equivalent: load a TINY subset for fast verification ---
print("=" * 60)
print("Cell 4 — load dataset")
print("=" * 60)
from datasets import load_dataset
DATASET = "ccdv/pubmed-summarization"
N = 20  # small for local smoke test
train_raw = load_dataset(DATASET, split=f"train[:{N}]")
val_raw   = load_dataset(DATASET, split=f"validation[:{N // 2}]")
test_raw  = load_dataset(DATASET, split=f"test[:{N // 2}]")
print(train_raw, val_raw, test_raw)
print("columns:", train_raw.column_names)

# --- Cell 5 equivalent: peek at one record ---
print("\n" + "=" * 60)
print("Cell 5 — sample record")
print("=" * 60)
rec = train_raw[0]
print("ARTICLE (first 500 chars):\n", rec["article"][:500])
print("\n--- ABSTRACT ---\n", rec["abstract"][:500])

# --- Cell 6 equivalent: clean + tokenize + save JSONL ---
print("\n" + "=" * 60)
print("Cell 6 — clean + save JSONL")
print("=" * 60)
from tqdm import tqdm
from preprocessing.clean import clean_article, split_sentences, is_record_valid


def process_split(ds, out_path):
    kept = dropped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc=os.path.basename(out_path)):
            art = clean_article(ex["article"])
            abs_ = clean_article(ex["abstract"])
            if not is_record_valid(art, abs_):
                dropped += 1
                continue
            rec = {
                "input_text": art,
                "target_text": abs_,
                "sentences": split_sentences(art),
            }
            f.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"{out_path}: kept={kept} dropped={dropped}")


process_split(train_raw, os.path.join(OUT, "train.jsonl"))
process_split(val_raw,   os.path.join(OUT, "val.jsonl"))
process_split(test_raw,  os.path.join(OUT, "test.jsonl"))

# --- Cell 7 equivalent: stats ---
print("\n" + "=" * 60)
print("Cell 7 — stats")
print("=" * 60)
import numpy as np


def length_stats(path):
    art_lens, abs_lens, sent_counts = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            art_lens.append(len(r["input_text"].split()))
            abs_lens.append(len(r["target_text"].split()))
            sent_counts.append(len(r["sentences"]))
    if not art_lens:
        print(f"{path}: empty after filtering")
        return
    print(path)
    print("  article words   median/p95:", int(np.median(art_lens)), int(np.percentile(art_lens, 95)))
    print("  abstract words  median/p95:", int(np.median(abs_lens)), int(np.percentile(abs_lens, 95)))
    print("  sentences/paper median/p95:", int(np.median(sent_counts)), int(np.percentile(sent_counts, 95)))


length_stats(os.path.join(OUT, "train.jsonl"))
length_stats(os.path.join(OUT, "val.jsonl"))
length_stats(os.path.join(OUT, "test.jsonl"))

print("\n" + "=" * 60)
print("LOCAL PIPELINE VERIFIED")
print("=" * 60)
