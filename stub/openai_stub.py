"""Deterministic stand-in for the embedding and chat model endpoints.

It speaks the two OpenAI-compatible routes the application uses, so the whole
flow runs without a GPU, a model download or a paid API:

    POST /v1/embeddings         hashed bag-of-words vectors (lexical similarity)
    POST /v1/chat/completions   extractive answer: sentences from the context
                                block that best matches the question, with
                                that block's [n] marker
    GET  /v1/models
    GET  /stats, POST /stats/reset   request counters used by the tests

It is NOT a model. Retrieval quality measured against it shows that the wiring
is correct, not how a real embedding model will rank the corpus.

    uvicorn stub.openai_stub:app --port 9000
"""
from __future__ import annotations

import hashlib
import math
import re
import time

from fastapi import FastAPI
from pydantic import BaseModel

DIM = 384
_STOP = set("""a an and are as at be but by can do does for from how i if in into is it its many
me much my of on or so than that the their then there these this to up was what when where which
who why will with within you your long often get need""".split())

app = FastAPI(title="OpenAI-compatible stub")
STATS = {"embedding_requests": 0, "embedding_inputs": 0, "chat_requests": 0}


def tokens(text: str) -> list:
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP:
            continue
        if len(tok) > 5 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        else:
            for suffix in ("ing", "es", "ed", "s"):
                if len(tok) > len(suffix) + 3 and tok.endswith(suffix):
                    tok = tok[: -len(suffix)]
                    break
        out.append(tok)
    return out


def embed(text: str) -> list:
    vec = [0.0] * DIM
    counts: dict = {}
    for tok in tokens(text):
        counts[tok] = counts.get(tok, 0) + 1
    for tok, n in counts.items():
        digest = hashlib.md5(tok.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % DIM
        sign = 1.0 if digest[4] % 2 else -1.0
        vec[index] += sign * (1.0 + math.log(n))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class EmbeddingRequest(BaseModel):
    model: str = "embed"
    input: str | list
    encoding_format: str | None = None
    dimensions: int | None = None


class ChatRequest(BaseModel):
    model: str = "chat"
    messages: list
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "embed", "object": "model"},
                                       {"id": "chat", "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else list(req.input)
    STATS["embedding_requests"] += 1
    STATS["embedding_inputs"] += len(inputs)
    return {
        "object": "list",
        "model": req.model,
        "data": [{"object": "embedding", "index": i, "embedding": embed(str(t))}
                 for i, t in enumerate(inputs)],
        "usage": {"prompt_tokens": sum(len(str(t).split()) for t in inputs),
                  "total_tokens": sum(len(str(t).split()) for t in inputs)},
    }


def _content(message) -> str:
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):  # OpenAI content-part form
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


_QUANTITY = re.compile(r"\b(how (long|many|much|often)|when|what time|at what|within)\b", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z])\d+([.:]\d+)?(?![A-Za-z])")


def extractive_answer(prompt: str) -> str:
    """Parse the middleware prompt (numbered "[n] source: ..." blocks followed
    by "Question: ...") and answer with the two sentences of the best matching
    block that overlap most with the question, followed by the block's marker."""
    question = prompt.rsplit("Question:", 1)[-1].strip() if "Question:" in prompt else prompt
    q = set(tokens(question))
    wants_number = bool(_QUANTITY.search(question))
    best_score, best_sentences, best_number = 0.0, [], 0
    for block in re.split(r"\n(?=\[\d+\] source:)", prompt):
        m = re.match(r"\[(\d+)\] source:[^\n]*\n(.*)", block, flags=re.DOTALL)
        if not m:
            continue
        body = m.group(2).split("\nQuestion:")[0]
        block_score = len(q & set(tokens(body))) / math.sqrt(len(set(tokens(body))) or 1)
        scored = []
        for position, sentence in enumerate(re.split(r"(?<=[.!?])\s+|\n+", body)):
            sentence = re.sub(r"^(\d+\.|-)\s+", "", sentence.strip())
            if len(sentence) < 12 or sentence.startswith("#"):
                continue
            words = set(tokens(sentence))
            score = float(len(q & words))
            if wants_number and _NUMBER.search(sentence):
                score *= 2.0
            scored.append((score, position, sentence))
        if scored and block_score > best_score:
            # best two by score, earlier sentence first on a tie, in reading order
            top = sorted(scored, key=lambda t: (-t[0], t[1]))[:2]
            best_sentences = [text for score, _, text in sorted(top, key=lambda t: t[1])
                              if score > 0]
            best_score, best_number = block_score, int(m.group(1))
    if not best_number or not best_sentences:
        return "The provided documents do not answer this question."
    return "%s [%d]" % (" ".join(best_sentences), best_number)


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    STATS["chat_requests"] += 1
    user = [m for m in req.messages if isinstance(m, dict) and m.get("role") == "user"]
    answer = extractive_answer(_content(user[-1]) if user else "")
    return {
        "id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/stats")
def stats():
    return STATS


@app.post("/stats/reset")
def reset():
    for key in STATS:
        STATS[key] = 0
    return STATS


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
