"""RAG middleware: a FastAPI service that retrieves from the vector store with
LlamaIndex and calls a chat model through an OpenAI-compatible base URL.

    GET  /healthz   liveness: the process is up (used as the container probe)
    GET  /readyz    readiness: the vector store answers and the collection has chunks
    GET  /info      non-secret configuration and index size
    POST /query     {"question": "...", "top_k": 4}  ->  answer + numbered sources

Configuration comes from environment variables only (see rag/config.py). In
deployment LLM_BASE_URL and EMBED_BASE_URL point at the Klique LLM gateway and
LLM_MODEL / EMBED_MODEL are gateway aliases, so the model behind an alias can
change without touching this service.

    uvicorn service.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field

from rag.clients import make_embed_model, make_llm, make_qdrant
from rag.config import Settings
from rag.retrieval import IndexNotReady, Retriever
from rag.store import count_points

SYSTEM_PROMPT = (
    "You answer questions about Pellwood products using only the numbered sources "
    "in the user message. Cite the sources you used with their numbers in square "
    "brackets, for example [1]. If the sources do not contain the answer, say that "
    "the documents do not answer the question. Do not use other knowledge."
)


def build_user_prompt(question: str, sources: list) -> str:
    """Numbered context blocks followed by the question. The stub chat backend
    parses this exact layout, so change both together."""
    blocks = ["[%d] source: %s (%s)\n%s" % (i, s["doc_path"], s["title"], s["text"])
              for i, s in enumerate(sources, start=1)]
    return "Sources:\n\n%s\n\nQuestion: %s" % ("\n\n".join(blocks), question)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    n: int
    doc_path: str
    title: str
    section: str
    score: float
    dataset_version: str
    snippet: str
    cited: bool


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    cited: list[int]
    model: str


class RagService:
    """Holds the clients. The embedded Qdrant mode is not thread-safe, so calls
    are serialised with a lock; a Qdrant server handles concurrency itself."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = make_qdrant(settings)
        self.retriever = Retriever(self.client, make_embed_model(settings), settings)
        self.llm = make_llm(settings)
        self._lock = threading.Lock()

    def points(self) -> int:
        with self._lock:
            return count_points(self.client, self.settings.collection)

    def query(self, question: str, top_k: int | None) -> QueryResponse:
        with self._lock:
            sources = self.retriever.retrieve(question, top_k)
        if not sources:
            return QueryResponse(answer="The documents do not answer the question.",
                                 sources=[], cited=[], model=self.settings.llm_model)
        reply = self.llm.chat([
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_user_prompt(question, sources)),
        ])
        answer = (reply.message.content or "").strip()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)
                        if 1 <= int(n) <= len(sources)})
        return QueryResponse(
            answer=answer, cited=cited, model=self.settings.llm_model,
            sources=[Source(n=i, doc_path=s["doc_path"], title=s["title"], section=s["section"],
                            score=s["score"], dataset_version=s["dataset_version"],
                            snippet=s["text"][:400], cited=i in cited)
                     for i, s in enumerate(sources, start=1)])

    def close(self) -> None:
        self.client.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = RagService(settings or Settings.from_env())
        yield
        app.state.rag.close()

    app = FastAPI(title="Klique RAG middleware", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        try:
            points = app.state.rag.points()
        except Exception as exc:
            raise HTTPException(503, "vector store not reachable: %s" % type(exc).__name__)
        if points == 0:
            raise HTTPException(503, "index is empty; run the ingestion pipeline")
        return {"status": "ready", "points": points}

    @app.get("/info")
    def info():
        rag = app.state.rag
        return {"settings": rag.settings.public(), "points": rag.points()}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        try:
            return app.state.rag.query(req.question.strip(), req.top_k)
        except IndexNotReady as exc:
            raise HTTPException(503, "index not ready: %s" % exc)

    return app


app = create_app()
