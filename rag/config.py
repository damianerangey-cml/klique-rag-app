"""Configuration, read from environment variables only.

No endpoint, key or token is written in code. On Klique the non-secret values
come from launch-form environment variables or pipeline parameters, and the
secret values (API keys, tokens) come from a vault.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace

# Fields that must never be printed, logged or returned by an API.
SECRET_FIELDS = ("embed_api_key", "llm_api_key", "qdrant_api_key", "qdrant_bearer_token")

# The OpenAI client refuses an empty key. Endpoints without authentication
# (the local stub) ignore this placeholder.
_NO_KEY = "not-set"


@dataclass(frozen=True)
class Settings:
    # Embedding endpoint: OpenAI-compatible, POST {base}/embeddings.
    embed_base_url: str = "http://localhost:9000/v1"
    embed_api_key: str = _NO_KEY
    embed_model: str = "embed"
    embed_batch_size: int = 32
    # Chat endpoint: OpenAI-compatible, POST {base}/chat/completions.
    llm_base_url: str = "http://localhost:9000/v1"
    llm_api_key: str = _NO_KEY
    llm_model: str = "chat"
    llm_max_tokens: int = 512
    # Vector store. QDRANT_URL selects a Qdrant server; without it the embedded
    # on-disk mode at QDRANT_PATH is used (one process at a time).
    qdrant_url: str = ""
    qdrant_api_key: str = ""        # Qdrant's own "api-key" header
    qdrant_bearer_token: str = ""   # "Authorization: Bearer" for a gateway route
    qdrant_path: str = ".local/qdrant"
    collection: str = "pellwood_docs"
    # Chunking and retrieval.
    chunk_size: int = 220           # words
    chunk_overlap: int = 30         # words
    top_k: int = 4

    @classmethod
    def from_env(cls, overrides: dict | None = None) -> "Settings":
        """Environment first, then explicit non-secret overrides (used by the
        pipeline to pass its parameters to each step)."""
        env = os.environ
        s = cls(
            embed_base_url=env.get("EMBED_BASE_URL", cls.embed_base_url),
            embed_api_key=env.get("EMBED_API_KEY", "") or _NO_KEY,
            embed_model=env.get("EMBED_MODEL", cls.embed_model),
            embed_batch_size=int(env.get("EMBED_BATCH_SIZE", cls.embed_batch_size)),
            llm_base_url=env.get("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=env.get("LLM_API_KEY", "") or _NO_KEY,
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            llm_max_tokens=int(env.get("LLM_MAX_TOKENS", cls.llm_max_tokens)),
            qdrant_url=env.get("QDRANT_URL", ""),
            qdrant_api_key=env.get("QDRANT_API_KEY", ""),
            qdrant_bearer_token=env.get("QDRANT_BEARER_TOKEN", ""),
            qdrant_path=env.get("QDRANT_PATH", cls.qdrant_path),
            collection=env.get("RAG_COLLECTION", cls.collection),
            chunk_size=int(env.get("RAG_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(env.get("RAG_CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(env.get("RAG_TOP_K", cls.top_k)),
        )
        clean = {
            k: type(getattr(s, k))(v)
            for k, v in (overrides or {}).items()
            if v not in (None, "") and hasattr(s, k) and k not in SECRET_FIELDS
        }
        return replace(s, **clean) if clean else s

    def public(self) -> dict:
        """Settings that are safe to log or return."""
        return {k: v for k, v in asdict(self).items() if k not in SECRET_FIELDS}
