"""Ingestion steps. Each function takes and returns plain JSON-serialisable
values, so it can run as a pipeline step on its own compute resource (the
return value travels to the next step as an artifact) or in-process.

  plan_changes      compare the dataset version with what the vector store holds
  chunk_changed     chunk only added and modified documents
  embed_and_upsert  embed those chunks, upsert, delete stale and removed chunks
"""
from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import TextNode

from .chunking import chunk_document
from .clients import make_embed_model, make_qdrant, make_vector_store
from .config import Settings
from .datasets import diff_manifests
from .store import (count_points, delete_documents, ensure_payload_index, indexed_state)
from qdrant_client.http import models as qm

DOCS_PREFIX = "docs/"
EVAL_FILE = "eval/questions.jsonl"


def indexable_files(files: dict) -> dict:
    return {p: h for p, h in files.items() if p.startswith(DOCS_PREFIX) and p.endswith(".md")}


def plan_changes(dataset: dict, settings: Settings) -> dict:
    """Diff the dataset manifest against the hashes stored with the indexed
    chunks. The vector store is the reference, so the plan is correct whichever
    dataset version was indexed last, and a re-run after a failure resumes."""
    wanted = indexable_files(dataset["files"])
    client = make_qdrant(settings)
    try:
        state = indexed_state(client, settings.collection)
    finally:
        client.close()
    diff = diff_manifests({p: e["hash"] for p, e in state.items()}, wanted)
    to_embed = diff["added"] + diff["modified"]
    return {
        "dataset_id": dataset["id"],
        "dataset_version": dataset["version"],
        "added": diff["added"],
        "modified": diff["modified"],
        "removed": diff["removed"],
        "unchanged": diff["unchanged"],
        "hashes": {p: wanted[p] for p in to_embed},
    }


def chunk_changed(dataset_dir, plan: dict, settings: Settings) -> list:
    """Chunk only the documents the plan marks as added or modified."""
    chunks = []
    for doc_path in plan["added"] + plan["modified"]:
        text = (Path(dataset_dir) / doc_path).read_text(encoding="utf-8")
        chunks.extend(chunk_document(
            doc_path, text, plan["hashes"][doc_path], plan["dataset_version"],
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap))
    return chunks


def embed_and_upsert(chunks: list, plan: dict, settings: Settings) -> dict:
    """Embed the chunks through the OpenAI-compatible endpoint and upsert them.
    New chunks are written before stale ones are deleted, so a modified document
    stays answerable during the update."""
    client = make_qdrant(settings)
    try:
        if chunks:
            embed_model = make_embed_model(settings)
            vectors = embed_model.get_text_embedding_batch([c["embed_text"] for c in chunks])
            nodes = [
                TextNode(id_=c["id"], text=c["text"], metadata=c["metadata"], embedding=v)
                for c, v in zip(chunks, vectors)
            ]
            # Creates the collection on first use, sized from the vectors.
            make_vector_store(client, settings).add(nodes)
            if settings.qdrant_url:
                ensure_payload_index(client, settings.collection)

        # Old chunks of modified documents: same path, different content hash.
        stale = 0
        for doc_path in plan["modified"]:
            before = count_points(client, settings.collection)
            client.delete(
                settings.collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_path", match=qm.MatchValue(value=doc_path))],
                    must_not=[qm.FieldCondition(
                        key="doc_hash", match=qm.MatchValue(value=plan["hashes"][doc_path]))])),
                wait=True)
            stale += before - count_points(client, settings.collection)
        removed = delete_documents(client, settings.collection, plan["removed"])
        return {
            "embedded_chunks": len(chunks),
            "embedded_documents": len(plan["added"]) + len(plan["modified"]),
            "skipped_documents": len(plan["unchanged"]),
            "deleted_stale_chunks": stale,
            "deleted_removed_chunks": removed,
            "points_total": count_points(client, settings.collection),
        }
    finally:
        client.close()
