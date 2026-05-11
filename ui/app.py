"""Streamlit UI: upload a PDF or paste text, get a summary.

Run locally:
    streamlit run ui/app.py

It auto-detects fine-tuned checkpoints under ``models/`` and falls back to
public pretrained models (DistilBART / pegasus-pubmed) so the UI works even
before you've trained anything.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make project root importable when running `streamlit run ui/app.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.clean import clean_article, split_sentences  # noqa: E402
from models.baselines import textrank_summary, tfidf_summary    # noqa: E402
from models.abstractive import AbstractiveSummarizer            # noqa: E402


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "models"


def _find_abstractive_checkpoint() -> str:
    """Return a local fine-tuned checkpoint path if present, else a HF id."""
    for name in ("pegasus-pubmed-ft", "distilbart-cnn-12-6-ft"):
        p = MODELS_DIR / name
        if p.exists() and (p / "config.json").exists():
            return str(p)
    return "sshleifer/distilbart-cnn-12-6"  # fast public fallback


@st.cache_resource(show_spinner="Loading abstractive model…")
def load_abstractive() -> AbstractiveSummarizer:
    ckpt = _find_abstractive_checkpoint()
    return AbstractiveSummarizer(ckpt)


@st.cache_resource(show_spinner="Loading BERTSUM extractive model…")
def load_bertsum():
    import torch
    from transformers import AutoTokenizer
    from models.bertsum import BertSumExt

    base = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(base)
    model = BertSumExt(base)
    ckpt = MODELS_DIR / "bertsum" / "bertsum_epoch3.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        st.warning("No fine-tuned BERTSUM checkpoint found — using untrained head. "
                   "Run the Colab notebook `training/03_finetune_bertsum.ipynb` first.")
    return model, tok


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
def pdf_to_text(uploaded_file) -> str:
    import fitz  # pymupdf
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Section splitting (very simple heuristic)
# ---------------------------------------------------------------------------
SECTION_HEADERS = [
    "abstract", "introduction", "background", "related work",
    "methods", "methodology", "materials and methods",
    "results", "discussion", "conclusion", "conclusions",
]


def split_sections(text: str) -> dict:
    import re
    pattern = re.compile(
        r"\n\s*(" + "|".join(SECTION_HEADERS) + r")\s*\n",
        re.IGNORECASE,
    )
    parts = pattern.split(text)
    if len(parts) < 3:
        return {"Full text": text}
    sections = {}
    # parts = [pre, header1, body1, header2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip().title()] = parts[i + 1].strip()
    return sections


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Scientific Paper Summarizer", layout="wide")
st.title("Scientific Paper Summarizer")
st.caption("Transformer-based summarization (BERTSUM / PEGASUS) fine-tuned on PubMed.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Summarization model",
        ("Abstractive (PEGASUS / DistilBART)", "Extractive (BERTSUM)", "Baseline (TextRank)"),
    )
    num_sentences = st.slider("Extractive: # sentences", 3, 12, 7)
    abs_min = st.slider("Abstractive: min length", 40, 200, 80)
    abs_max = st.slider("Abstractive: max length", 100, 400, 256)
    show_keywords = st.checkbox("Show keywords (KeyBERT)", value=False)
    show_sections = st.checkbox("Section-wise summarization", value=False)
    long_paper = st.checkbox("Hierarchical (long paper > 1024 tokens)", value=True)

tab_in, tab_out = st.tabs(["Input", "Summary"])

with tab_in:
    uploaded = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    pasted = st.text_area("…or paste article text", height=250)
    gold = st.text_area("Optional: paste the gold abstract to compute ROUGE", height=120)
    go = st.button("Generate summary", type="primary")

with tab_out:
    if not go:
        st.info("Upload a PDF or paste text in the Input tab, then click **Generate summary**.")
    else:
        # 1. extract text
        if uploaded is not None:
            with st.spinner("Extracting text from PDF…"):
                raw = pdf_to_text(uploaded)
        else:
            raw = pasted or ""
        if not raw.strip():
            st.error("Could not extract any text. If the PDF is scanned/image-only, OCR is required (out of scope).")
            st.stop()

        # 2. clean
        text = clean_article(raw)
        st.write(f"**Article length:** ~{len(text.split())} words, "
                 f"{len(split_sentences(text))} sentences.")

        # 3. summarize
        def run_summary(t: str) -> str:
            if model_choice.startswith("Abstractive"):
                model = load_abstractive()
                if long_paper:
                    return model.hierarchical_summarize(t, final_min_length=abs_min, final_max_length=abs_max)
                return model.summarize(t, min_length=abs_min, max_length=abs_max)
            if model_choice.startswith("Extractive"):
                from models.bertsum import bertsum_summary
                model, tok = load_bertsum()
                return bertsum_summary(t, model, tok, num_sentences=num_sentences)
            return textrank_summary(t, num_sentences=num_sentences)

        with st.spinner("Generating summary…"):
            summary = run_summary(text)

        st.subheader("Summary")
        st.write(summary)
        st.download_button("Download summary (.txt)", summary, file_name="summary.txt")

        # 4. keywords
        if show_keywords:
            with st.spinner("Extracting keywords…"):
                from models.keywords import extract_keywords
                keywords = extract_keywords(text, top_n=10)
            st.subheader("Keywords")
            st.write(", ".join(keywords))

        # 5. section-wise
        if show_sections:
            st.subheader("Section-wise summaries")
            sections = split_sections(text)
            for name, body in sections.items():
                if len(body.split()) < 60:
                    continue
                with st.expander(name):
                    with st.spinner(f"Summarizing {name}…"):
                        section_summary = run_summary(body)
                    st.write(section_summary)

        # 6. ROUGE vs gold abstract
        if gold.strip():
            from evaluation.rouge_eval import compute_rouge
            scores = compute_rouge([summary], [gold])
            st.subheader("ROUGE vs gold abstract")
            st.json(scores)
