# Scientific Article Summarizer

Fine-tunes transformer models (BERTSUM extractive + PEGASUS/DistilBART abstractive) on the PubMed scientific papers dataset and exposes the result through a Streamlit web app that takes PDF uploads.

> **All training runs on Google Colab (free GPU).** Your laptop only runs the Streamlit UI for the final demo. The UI loads the fine-tuned model from Google Drive (or the Hugging Face Hub) — no local GPU needed.

---

## Workflow at a glance

```
┌────────────────────────────────┐         ┌──────────────────────────────┐
│  Google Colab (free T4 GPU)    │         │  Your laptop (CPU-only)      │
│  • download PubMed             │         │  • streamlit run ui/app.py   │
│  • preprocess + ORACLE labels  │         │  • PDF upload → summary      │
│  • fine-tune BERTSUM           │         │                              │
│  • fine-tune PEGASUS-PubMed    │  ─────► │  Loads checkpoint from:      │
│  • evaluate (ROUGE)            │ Drive / │   ./models/  (synced from    │
│  • saves to MyDrive/...        │ HF Hub  │     Drive) or HF Hub         │
└────────────────────────────────┘         └──────────────────────────────┘
```

---

## Step 1 — Push this project to GitHub

The Colab notebooks pull the code fresh on every run via `git clone`, so any local edit you push is picked up automatically.

```powershell
cd scientific-summarizer
git add -A
git commit -m "update"
git push
```

Repo: https://github.com/Captain-Uchiha/scientific-paper-summarizer

## Step 2 — Open the notebooks in Colab

In VS Code, open each `.ipynb` under `training/` and connect the kernel to a **Colab T4 GPU** runtime (kernel picker → Google Colab → hosted T4). Or open them in browser Colab via the `Open in Colab` button. Each notebook's first cell clones this GitHub repo, mounts Drive for persistent storage of datasets/checkpoints, and installs dependencies.

Run them in order:

| Notebook | What it does | Approx. runtime |
|---|---|---|
| `01_load_and_preprocess.ipynb` | Downloads PubMed (3k train / 400 val / 400 test), cleans, saves JSONL to Drive | 5–10 min |
| `02_build_oracle_labels.ipynb` | Builds ORACLE extractive labels for BERTSUM | 10–20 min |
| `03_finetune_bertsum.ipynb` | Fine-tunes BERTSUM (3 epochs) | 30–60 min |
| `04_finetune_pegasus.ipynb` | Fine-tunes `google/pegasus-pubmed` (3 epochs) | 1–2 h |
| `05_evaluate.ipynb` | Runs all 4 models on the test split, writes ROUGE table | 20–40 min |

All dataset splits and checkpoints land in `MyDrive/scientific-summarizer-data/`.

**If PEGASUS runs out of GPU memory:** open notebook 04 and change the `MODEL_NAME` line to `sshleifer/distilbart-cnn-12-6`. Smaller, fits comfortably on the free Colab T4.

## Step 3 — Run the Streamlit UI on your laptop

```powershell
cd scientific-summarizer
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Sync the fine-tuned model folder from Drive to local ./models/
#   (download via Drive web UI, or use Google Drive desktop)
# Expected layout after sync:
#   scientific-summarizer/models/pegasus-pubmed-ft/
#   scientific-summarizer/models/bertsum/bertsum_epoch3.pt
# (You can copy from MyDrive/scientific-summarizer-data/models/ into ./models/.)

streamlit run ui/app.py
```

The UI auto-detects checkpoints under `models/`. If none are present yet it falls back to public pretrained models (DistilBART + TextRank) so you can still demo the interface.

---

## Repository layout

```
scientific-summarizer/
├── dataset/           processed JSONL splits (lives on Drive; gitignored)
├── preprocessing/     cleaning + ORACLE label code (clean.py, oracle.py)
├── models/            checkpoints (gitignored) + model wrappers
│   ├── baselines.py   TF-IDF, TextRank
│   ├── bertsum.py     extractive transformer
│   ├── abstractive.py PEGASUS / DistilBART wrapper + hierarchical chunking
│   └── keywords.py    KeyBERT helper
├── training/          Colab notebooks 01–05
├── evaluation/        ROUGE scoring (rouge_eval.py)
├── ui/                Streamlit app (app.py)
├── notebooks/         exploration scratchpads
├── outputs/           generated summaries, results.csv (gitignored)
├── report/            literature survey, final report, slides
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

| Role | Model | Notes |
|---|---|---|
| Baseline 1 | TF-IDF top-sentence | no training |
| Baseline 2 | TextRank (`sumy`) | no training |
| Extractive | BERTSUM (`bert-base-uncased` + per-sentence head) | fine-tuned on PubMed ORACLE labels |
| Abstractive | `google/pegasus-pubmed` | fine-tuned on 3k PubMed samples |
| Fallback | `sshleifer/distilbart-cnn-12-6` | when GPU memory is tight |

## Metrics

ROUGE-1 / ROUGE-2 / ROUGE-L via `evaluate.load("rouge")`. After notebook 05, see `outputs/results.csv` for the final comparison table.

## Troubleshooting

- **Colab disconnects mid-training** — notebooks 03/04 save a checkpoint each epoch to Drive; re-open and re-run, training resumes from the last saved epoch.
- **`google/pegasus-pubmed` OOM** — switch `MODEL_NAME` to `sshleifer/distilbart-cnn-12-6` in notebook 04.
- **Articles longer than 1024 tokens** — the abstractive UI option "Hierarchical (long paper)" chunks → summarizes → re-summarizes.
- **Scanned PDFs** — the app shows a friendly error; OCR is out of scope.
