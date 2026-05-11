"""BERTSUM-style extractive summarizer.

A thin BERT encoder + per-sentence [CLS] classifier head. At inference time
each sentence gets a score; the top-k are returned in original order.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from preprocessing.clean import split_sentences


class BertSumExt(nn.Module):
    def __init__(self, base_model: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, cls_positions, cls_mask):
        """Return per-sentence logits.

        cls_positions: (B, S) indices of each [CLS] token in input_ids.
        cls_mask:      (B, S) 1 if sentence is real, 0 if padding sentence.
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # (B, T, H)
        b, s = cls_positions.shape
        idx = cls_positions.unsqueeze(-1).expand(-1, -1, last.size(-1))
        sent_vecs = last.gather(1, idx)  # (B, S, H)
        logits = self.classifier(sent_vecs).squeeze(-1)  # (B, S)
        logits = logits.masked_fill(cls_mask == 0, -1e4)
        return logits


def encode_for_bertsum(sentences: List[str], tokenizer, max_len: int = 512):
    """Concatenate sentences with [CLS] separators; return tensors + cls indices."""
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_ids: List[int] = []
    cls_positions: List[int] = []
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        if len(input_ids) + len(ids) + 2 > max_len:
            break
        cls_positions.append(len(input_ids))
        input_ids.append(cls_id)
        input_ids.extend(ids)
        input_ids.append(sep_id)
    input_ids = input_ids[:max_len]
    attention_mask = [1] * len(input_ids)
    # pad to max_len
    pad_len = max_len - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = attention_mask + [0] * pad_len
    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
        "cls_positions": torch.tensor([cls_positions]),
        "cls_mask": torch.ones(1, len(cls_positions), dtype=torch.long),
        "num_sentences": len(cls_positions),
    }


@torch.no_grad()
def bertsum_summary(text: str, model: BertSumExt, tokenizer,
                    num_sentences: int = 7, device: Optional[str] = None) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    enc = encode_for_bertsum(sentences, tokenizer)
    n = enc.pop("num_sentences")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc)[0, :n]
    scores = torch.sigmoid(logits).cpu().numpy()
    top_idx = sorted(scores.argsort()[-num_sentences:].tolist())
    return " ".join(sentences[i] for i in top_idx)
