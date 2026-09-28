"""Qwen3-Reranker through Ollama (`reranking.backend: qwen_ollama`).

Same scoring as `qwen_reranker.py` (P("yes") vs P("no") at the first generated
token of the reranker prompt), but the model runs inside Ollama: no torch, no
transformers, no model loading in the CLI process, and the weights stay
resident in Ollama's server between calls. Requires an Ollama build that
returns `logprobs` from /api/generate (0.34+ does) and a GGUF reranker pulled
into Ollama, e.g. `dengcao/Qwen3-Reranker-0.6B:Q8_0`.

The prompt is sent with `raw: true` so Ollama does not wrap it in its own chat
template; the reranker's scaffold (system turn, empty <think> block) is part of
the prompt text.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests

from flowmap.search.qwen_reranker import DEFAULT_INSTRUCTION, build_prompt, yes_probability

log = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Characters, not tokens: the CLI has no tokenizer for the GGUF. Measured on
# Ollama 0.34 / Apple silicon (docs/PLAN_GOLDEN_EVAL_TDD.md): with CONCURRENCY
# parallel requests Ollama batches prompt evaluation and scores 30 distinct
# candidates in ~1.4 s as long as each prompt stays under ~1000 tokens; past
# that it falls off a cliff to 17-28 s. Dense code tokenises at ~1.8 chars per
# token worst case, so 1500 chars keeps the ~120-token scaffold + document
# under the cliff. A smaller num_ctx also keeps the runner's KV/compute buffers
# small (2048 ctx let the runner grow to ~9 GB during long-prompt runs).
MAX_DOC_CHARS = 1500
NUM_CTX = 1280
TOP_LOGPROBS = 10
CONCURRENCY = 4
# Agent-facing budget: a reranked search must never hang. Per-request timeout
# covers a cold model load (~2-5 s); the overall deadline caps the whole batch.
TIMEOUT_S = 20
DEADLINE_S = 20

_YES = {"yes", " yes", "Yes", " Yes"}
_NO = {"no", " no", "No", " No"}


def _pick(top: list[dict], wanted: set[str]) -> float | None:
    best = None
    for t in top:
        if t.get("token") in wanted:
            lp = float(t["logprob"])
            best = lp if best is None else max(best, lp)
    return best


def score_from_logprobs(top: list[dict]) -> float | None:
    """P(yes) from Ollama's top_logprobs list for the first generated token.
    A missing answer token is treated as far below the lowest listed one.
    Returns None when *neither* answer token appears: the model did not answer
    the question at all (broken build, wrong model), which callers must not
    mistake for "not relevant"."""
    if not top:
        return None
    floor = min(float(t["logprob"]) for t in top) - 10.0
    yes = _pick(top, _YES)
    no = _pick(top, _NO)
    if yes is None and no is None:
        return None
    return yes_probability(floor if yes is None else yes, floor if no is None else no)


def _generate(session: requests.Session, url: str, model: str, prompt: str) -> float | None:
    r = session.post(
        f"{url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": NUM_CTX},
        },
        timeout=TIMEOUT_S,
    )
    if r.status_code == 404:
        raise RuntimeError(f"model {model!r} is not in Ollama; run: ollama pull {model}")
    r.raise_for_status()
    payload = r.json()
    lps = payload.get("logprobs") or []
    if not lps:
        raise RuntimeError("Ollama returned no logprobs; needs Ollama >= 0.34 and a model that supports them")
    return score_from_logprobs(lps[0].get("top_logprobs") or [])


class RerankTimeout(RuntimeError):
    pass


def score(query: str, documents: list[str], model: str, ollama_url: str = DEFAULT_OLLAMA_URL,
          instruction: str = DEFAULT_INSTRUCTION, deadline_s: float | None = None) -> list[float | None]:
    """Score every document, in parallel, within one overall deadline. On the
    deadline, queued requests are cancelled and RerankTimeout is raised; the few
    in-flight requests finish on their own per-request timeout in the background.
    A request that errors counts as one unanswered candidate (None) so a single
    bad response does not discard the rest; only an all-failed batch raises."""
    if deadline_s is None:
        deadline_s = DEADLINE_S  # resolved at call time so it can be tuned/patched
    prompts = [build_prompt(query, d[:MAX_DOC_CHARS], instruction) for d in documents]
    t0 = time.monotonic()
    session = requests.Session()
    pool = ThreadPoolExecutor(max_workers=CONCURRENCY)
    try:
        futures = {pool.submit(_generate, session, ollama_url, model, p): i for i, p in enumerate(prompts)}
        out: list[float | None] = [None] * len(prompts)
        pending = set(futures)
        errors: list[Exception] = []
        while pending:
            remaining = deadline_s - (time.monotonic() - t0)
            if remaining <= 0:
                raise RerankTimeout(f"reranking exceeded {deadline_s:.0f}s for {len(prompts)} candidates ({len(pending)} unfinished)")
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    out[futures[f]] = f.result()
                except Exception as e:  # noqa: BLE001 - one bad response must not sink the batch
                    errors.append(e)
                    out[futures[f]] = None
        if errors and len(errors) == len(prompts):
            raise errors[-1]
        if errors:
            log.warning("Qwen/Ollama reranking: %d of %d requests failed (last: %s); those candidates scored as unanswered",
                        len(errors), len(prompts), errors[-1])
        return out
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        session.close()


def rerank_qwen_ollama(query: str, candidates: list, model_name: str, ollama_url: str = DEFAULT_OLLAMA_URL) -> list:
    """Rerank HybridResult candidates by P(yes) via Ollama. Ties keep the incoming
    (RRF) order. On any failure, log and return the candidates unchanged."""
    if not candidates:
        return candidates
    try:
        scores = score(query, [c.text for c in candidates], model_name, ollama_url)
    except RerankTimeout as e:
        log.warning("Qwen/Ollama reranking timed out (%s): %s; returning fusion order", model_name, e)
        return candidates
    except Exception as e:  # noqa: BLE001 - degrade to RRF order, same contract as the other backends
        log.warning("Qwen/Ollama reranking failed (%s): %s", model_name, e)
        return candidates
    answered = [s for s in scores if s is not None]
    if not answered:
        log.warning(
            "Qwen/Ollama reranking produced no yes/no answer for any of %d candidates with model %s; "
            "the model build is probably broken (try: ollama pull %s). Returning fusion order.",
            len(candidates), model_name, "hf.co/Mungert/Qwen3-Reranker-0.6B-GGUF:Q8_0",
        )
        return candidates
    if len(answered) < len(scores):
        log.warning("Qwen/Ollama reranking: %d of %d candidates got no yes/no answer; scored as 0",
                    len(scores) - len(answered), len(scores))
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s or 0.0)
        c.score = float(s or 0.0)
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)  # sorted() is stable
