"""Qwen3-Reranker backend (`reranking.backend: qwen_direct`).

Qwen3-Reranker models are causal LMs, not cross-encoders: relevance is the
probability of the token "yes" versus "no" at the final position of a chat
prompt that carries an instruction, the query and the document. This module
implements that scoring directly with transformers, in-process, batch size 1.
The model is loaded once per process and cached.

Prompt format follows the Qwen3-Reranker model card.
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Given a code search query, judge whether the code chunk is the implementation "
    "the query is looking for. Function, method and class definitions that do the "
    "work count; mere mentions, constants, config, imports and tests do not."
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_qwen_cache: dict[str, "QwenReranker"] = {}


def build_prompt(query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return f"{PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{SUFFIX}"


def yes_probability(yes_logit: float, no_logit: float) -> float:
    """Softmax over the two answer logits, returned as P(yes). Overflow-safe."""
    m = max(yes_logit, no_logit)
    ey, en = math.exp(yes_logit - m), math.exp(no_logit - m)
    return ey / (ey + en)


class QwenReranker:
    def __init__(self, model_name: str, max_length: int = 1024, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        # Fixed padding lengths (see score()); bounded shape set => bounded MPS graph cache.
        self.buckets = tuple(b for b in (256, 512, 768, 1024) if b < max_length) + (max_length,)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        dtype = torch.float16 if device == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        unk = self.tokenizer.unk_token_id
        if self.yes_id is None or self.no_id is None or unk in (self.yes_id, self.no_id):
            raise ValueError(f"{model_name}: tokenizer has no single 'yes'/'no' tokens; not a Qwen3-Reranker vocabulary")

    def _fit_document(self, query: str, document: str, instruction: str) -> str:
        """Truncate the document (never the prompt scaffold) to the token budget."""
        fixed = len(self.tokenizer(build_prompt(query, "", instruction))["input_ids"])
        budget = max(64, self.max_length - fixed)
        ids = self.tokenizer(document, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return document
        return self.tokenizer.decode(ids[:budget])

    def score(self, query: str, documents: list[str], instruction: str = DEFAULT_INSTRUCTION) -> list[float]:
        import torch

        out: list[float] = []
        with torch.no_grad():
            for doc in documents:
                prompt = build_prompt(query, self._fit_document(query, doc, instruction), instruction)
                # Left-pad to a small set of fixed lengths. PyTorch's MPS backend
                # caches a compiled graph per distinct input shape; with one shape
                # per candidate that cache grew without bound (~12 GB per eval run).
                # Left padding keeps the real last token at position -1.
                n = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                bucket = next((b for b in self.buckets if b >= n), self.max_length)
                enc = self.tokenizer(
                    prompt, return_tensors="pt", padding="max_length", max_length=bucket, truncation=True,
                ).to(self.device)
                # Only the final position is scored: logits_to_keep avoids
                # materialising [seq_len x 151k-vocab] logits (~300 MB in fp16 per
                # 1024-token candidate). use_cache=False skips the KV cache, which
                # is dead weight for a single forward pass.
                try:
                    logits = self.model(**enc, logits_to_keep=1, use_cache=False).logits[0, -1]
                except TypeError:  # older transformers without those kwargs
                    logits = self.model(**enc).logits[0, -1]
                out.append(yes_probability(float(logits[self.yes_id]), float(logits[self.no_id])))
                del logits, enc
        if self.device == "mps":
            torch.mps.empty_cache()
        return out


def rerank_qwen(query: str, candidates: list, model_name: str, max_length: int = 1024) -> list:
    """Rerank HybridResult candidates by P(yes). Ties keep the incoming (RRF) order.
    On any failure, log and return the candidates unchanged."""
    if not candidates:
        return candidates
    try:
        reranker = _qwen_cache.get(model_name)
        if reranker is None:
            reranker = QwenReranker(model_name, max_length=max_length)
            _qwen_cache[model_name] = reranker
        scores = reranker.score(query, [c.text for c in candidates])
    except Exception as e:  # noqa: BLE001 - degrade to RRF order, same contract as _rerank
        log.warning("Qwen reranking failed: %s", e)
        return candidates
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)
        c.score = float(s)
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)  # sorted() is stable
