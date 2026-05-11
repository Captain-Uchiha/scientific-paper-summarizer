"""Abstractive summarization wrappers (PEGASUS / DistilBART / LED).

Provides a single ``AbstractiveSummarizer`` class and a hierarchical
chunk-then-summarize helper for papers longer than the model's context.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AbstractiveSummarizer:
    def __init__(self, model_name_or_path: str = "sshleifer/distilbart-cnn-12-6",
                 device: Optional[str] = None,
                 max_input_length: int = 1024,
                 max_target_length: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    @torch.no_grad()
    def summarize(self, text: str,
                  min_length: int = 80,
                  max_length: Optional[int] = None,
                  num_beams: int = 4) -> str:
        max_length = max_length or self.max_target_length
        inputs = self.tokenizer(
            text, max_length=self.max_input_length,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            num_beams=num_beams,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def chunk_text(self, text: str, overlap_tokens: int = 100) -> List[str]:
        """Split text into overlapping chunks that each fit in max_input_length."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        step = self.max_input_length - overlap_tokens
        chunks = []
        for start in range(0, len(ids), step):
            chunk_ids = ids[start:start + self.max_input_length]
            if not chunk_ids:
                break
            chunks.append(self.tokenizer.decode(chunk_ids, skip_special_tokens=True))
            if start + self.max_input_length >= len(ids):
                break
        return chunks

    def hierarchical_summarize(self, text: str,
                               final_min_length: int = 80,
                               final_max_length: Optional[int] = None) -> str:
        """Chunk → summarize each → summarize the concatenated chunk-summaries."""
        chunks = self.chunk_text(text)
        if len(chunks) <= 1:
            return self.summarize(text, min_length=final_min_length,
                                  max_length=final_max_length)
        partial = [self.summarize(c, min_length=40, max_length=180) for c in chunks]
        joined = " ".join(partial)
        return self.summarize(joined, min_length=final_min_length,
                              max_length=final_max_length)
