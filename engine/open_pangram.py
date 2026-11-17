"""Open Pangram (EditLens), the free local stand-in for Pangram (plan 4, 5.5).

Two models, both 4-bucket classifiers (0 = human ... 3 = AI). The score is the
expected bucket divided by 3, exactly as in pangramlabs/EditLens
scripts/inference.py. Text is cleaned the same way the models were trained:
emoji spelled out, AI-style openers stripped, lowercased, whitespace squashed.
(So Open Pangram never sees CAPS. Pangram 4.1 might.)

    roberta  pangram/editlens_roberta-large, 355M, 512 tokens, fast
    llama    pangram/editlens_Llama-3.2-3B, a LoRA on Llama 3.2 3B, 1024 tokens.
             Trained on a 4-bit base; loaded here in bf16 on the ungated
             unsloth copy of the same model.

Texts longer than the model's window get split at sentence breaks and scored
per chunk; the text's score is the mean of its chunks, each counted by its tokens.
"""
import re
from pathlib import Path

import emoji
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .settings import setting_path


def models_dir(source=None):
    """The folder holding the EditLens model files: OPEN_PANGRAM_DIR, else open-pangram/ in MODELS_DIR. It holds
    editlens_roberta-large/, and for the llama kind editlens_Llama-3.2-3B/ and Llama-3.2-3B/."""
    return setting_path("OPEN_PANGRAM_DIR", source)


BOILERPLATE_STARTS = ["Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly"]


class NormedLinear(torch.nn.Module):
    """Copied from pangramlabs/EditLens scripts/train.py: LayerNorm then Linear."""

    def __init__(self, hidden_size, num_labels, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))


def clean_text(text):
    """Mirror of EditLens preprocess.clean_text."""
    text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    paras = [p for p in text.split("\n") if p.strip()]
    if paras:
        first = emoji.replace_emoji(re.sub(r"^[^a-zA-Z0-9]*", "", paras[0]), "")
        if any(first.startswith(p) for p in BOILERPLATE_STARTS) and len(paras) > 1:
            text = "\n".join(paras[1:])
    return re.sub(r"\s+", " ", text.lower()).strip()


class OpenPangram:
    def __init__(self, kind="roberta", device=None, folder=None, settings=None):
        self.kind = kind
        root = Path(folder) if folder else models_dir(settings)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if kind == "roberta":
            path = str(root / "editlens_roberta-large")
            self.max_len = 512
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        elif kind == "llama":
            from peft import PeftModel
            adapter = str(root / "editlens_Llama-3.2-3B")
            self.max_len = 1024
            self.tok = AutoTokenizer.from_pretrained(adapter)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"
            base = AutoModelForSequenceClassification.from_pretrained(
                str(root / "Llama-3.2-3B"), num_labels=4, torch_dtype=torch.bfloat16)
            base.config.pad_token_id = self.tok.pad_token_id
            base.score = NormedLinear(base.config.hidden_size, 4, dtype=torch.bfloat16)
            self.model = PeftModel.from_pretrained(base, adapter).to(self.device)
        else:
            raise ValueError(kind)
        self.model.eval()
        self.n = self.model.config.num_labels if kind == "roberta" else 4

    def _chunks(self, text):
        """Split cleaned text into pieces that fit the window, at sentence breaks."""
        sents = re.split(r"(?<=[.!?])\s+", text)
        out, cur = [], ""
        budget = self.max_len - 8
        for s in sents:
            cand = f"{cur} {s}".strip()
            if cur and len(self.tok(cand, add_special_tokens=False)["input_ids"]) > budget:
                out.append(cur)
                cur = s
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out or [text]

    @torch.no_grad()
    def _score_pieces(self, pieces, batch=16):
        res = []
        for i in range(0, len(pieces), batch):
            enc = self.tok(pieces[i:i + batch], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float().cpu().numpy()
            probs = softmax(logits, axis=1)
            for p, ids in zip(probs, enc["attention_mask"].sum(1).tolist()):
                res.append({"score": float(p @ np.arange(self.n) / (self.n - 1)),
                            "bucket": int(p.argmax()), "probs": p.round(4).tolist(), "tokens": ids})
        return res

    def score(self, texts, batch=16):
        """[{score, bucket, chunks}] for each text."""
        flat, owner = [], []
        for i, t in enumerate(texts):
            for c in self._chunks(clean_text(t)):
                flat.append(c)
                owner.append(i)
        scored = self._score_pieces(flat, batch)
        out = [{"chunks": []} for _ in texts]
        for i, r in zip(owner, scored):
            out[i]["chunks"].append(r)
        for o in out:
            w = sum(c["tokens"] for c in o["chunks"]) or 1
            o["score"] = sum(c["score"] * c["tokens"] for c in o["chunks"]) / w
            o["bucket"] = max(o["chunks"], key=lambda c: c["tokens"])["bucket"] if o["chunks"] else None
        return out

    def unload(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
