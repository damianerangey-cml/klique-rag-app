"""Retrieval through a LlamaIndex VectorStoreIndex over the Qdrant collection.
Used by the middleware for queries and by the pipeline for the quality check,
so the check measures the same code path that serves users."""
from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode

from .clients import make_vector_store
from .config import Settings
from .store import collection_exists


class IndexNotReady(RuntimeError):
    """The collection does not exist yet (the ingestion pipeline has not run)."""


class Retriever:
    def __init__(self, client, embed_model, settings: Settings):
        self._client = client
        self._embed_model = embed_model
        self._settings = settings
        self._index = None

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            if not collection_exists(self._client, self._settings.collection):
                raise IndexNotReady("collection %r does not exist" % self._settings.collection)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=make_vector_store(self._client, self._settings),
                embed_model=self._embed_model)
        return self._index

    def retrieve(self, question: str, top_k: int | None = None) -> list:
        """Return the top chunks as plain dicts, best first."""
        retriever = self._get_index().as_retriever(
            similarity_top_k=top_k or self._settings.top_k)
        results = []
        for hit in retriever.retrieve(question):
            meta = hit.node.metadata or {}
            results.append({
                "doc_path": meta.get("doc_path", ""),
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "dataset_version": meta.get("dataset_version", ""),
                "score": round(float(hit.score or 0.0), 4),
                "text": hit.node.get_content(metadata_mode=MetadataMode.NONE),
            })
        return results
