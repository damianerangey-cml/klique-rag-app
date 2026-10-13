"""Retrieval-quality check. The question set travels with the corpus as
eval/questions.jsonl inside the dataset version, one JSON object per line:
{"question": "...", "expected": "docs/<path>.md"}."""
from __future__ import annotations

import json
from pathlib import Path

from .clients import make_embed_model, make_qdrant
from .config import Settings
from .retrieval import Retriever


def load_questions(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_retrieval(questions: list, settings: Settings, top_k: int | None = None) -> dict:
    """hit@1, hit@k and mean reciprocal rank of the expected document."""
    k = top_k or settings.top_k
    client = make_qdrant(settings)
    try:
        retriever = Retriever(client, make_embed_model(settings), settings)
        hits1 = hitsk = 0
        rr = 0.0
        misses = []
        for q in questions:
            paths = [r["doc_path"] for r in retriever.retrieve(q["question"], top_k=k)]
            rank = paths.index(q["expected"]) + 1 if q["expected"] in paths else 0
            hits1 += rank == 1
            hitsk += rank > 0
            rr += 1.0 / rank if rank else 0.0
            if rank != 1:
                misses.append({"question": q["question"], "expected": q["expected"],
                               "rank": rank, "retrieved": paths})
    finally:
        client.close()
    n = max(len(questions), 1)
    return {
        "questions": len(questions),
        "top_k": k,
        "hit_at_1": round(hits1 / n, 4),
        "hit_at_k": round(hitsk / n, 4),
        "mrr": round(rr / n, 4),
        "not_top_ranked": misses,
    }
