"""Factories for the three external dependencies: embedding endpoint, chat
endpoint and vector store. Everything goes through LlamaIndex classes so the
pipeline and the middleware share one client implementation."""
from __future__ import annotations

from pathlib import Path

from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .config import Settings


def make_embed_model(s: Settings) -> OpenAILikeEmbedding:
    return OpenAILikeEmbedding(
        model_name=s.embed_model,
        api_base=s.embed_base_url,
        api_key=s.embed_api_key,
        embed_batch_size=s.embed_batch_size,
        max_retries=3,
        timeout=60.0,
    )


def make_llm(s: Settings) -> OpenAILike:
    return OpenAILike(
        model=s.llm_model,
        api_base=s.llm_base_url,
        api_key=s.llm_api_key,
        is_chat_model=True,
        temperature=0.0,
        max_tokens=s.llm_max_tokens,
        max_retries=3,
        timeout=120.0,
    )


def make_qdrant(s: Settings) -> QdrantClient:
    """Server mode when QDRANT_URL is set, embedded on-disk mode otherwise.

    Behind the Klique application gateway an HTTP route expects a bearer token,
    which is sent with auth_token_provider. A Qdrant server protected by its own
    API key uses api_key instead.
    """
    if s.qdrant_url:
        kwargs = {"url": s.qdrant_url, "timeout": 30}
        if s.qdrant_api_key:
            kwargs["api_key"] = s.qdrant_api_key
        if s.qdrant_bearer_token:
            token = s.qdrant_bearer_token
            kwargs["auth_token_provider"] = lambda: token
        return QdrantClient(**kwargs)
    Path(s.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=s.qdrant_path)


def make_vector_store(client: QdrantClient, s: Settings) -> QdrantVectorStore:
    return QdrantVectorStore(client=client, collection_name=s.collection)
