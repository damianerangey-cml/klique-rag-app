"""Vector store helpers that go beyond what the LlamaIndex wrapper offers:
reading which document versions are indexed, and deleting a document's chunks.

Every point carries the payload keys doc_path and doc_hash (set in
chunking.py). The store is therefore the record of what is embedded: the
pipeline compares it with the dataset version and embeds only the difference.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def collection_exists(client: QdrantClient, collection: str) -> bool:
    return client.collection_exists(collection)


def count_points(client: QdrantClient, collection: str) -> int:
    if not collection_exists(client, collection):
        return 0
    return client.count(collection, exact=True).count


def indexed_state(client: QdrantClient, collection: str) -> dict:
    """Return {doc_path: {"hash": doc_hash, "point_ids": [...]}} for all points."""
    state: dict = {}
    if not collection_exists(client, collection):
        return state
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=256, offset=offset,
            with_payload=["doc_path", "doc_hash"], with_vectors=False)
        for p in points:
            payload = p.payload or {}
            path = payload.get("doc_path")
            if not path:
                continue
            entry = state.setdefault(path, {"hash": payload.get("doc_hash"), "point_ids": []})
            entry["point_ids"].append(str(p.id))
        if offset is None:
            break
    for entry in state.values():
        entry["point_ids"].sort()
    return state


def delete_documents(client: QdrantClient, collection: str, doc_paths: list) -> int:
    """Delete all chunks of the given documents. Returns the number of points removed."""
    if not doc_paths or not collection_exists(client, collection):
        return 0
    before = count_points(client, collection)
    client.delete(
        collection,
        points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="doc_path", match=qm.MatchAny(any=list(doc_paths)))])),
        wait=True)
    return before - count_points(client, collection)


def ensure_payload_index(client: QdrantClient, collection: str) -> None:
    """Keyword index on doc_path for filtered deletes. Server mode only; the
    embedded mode has no payload indexes and does not need them."""
    try:
        client.create_payload_index(collection, "doc_path", qm.PayloadSchemaType.KEYWORD)
    except Exception as exc:  # index already exists, or embedded mode
        print("payload index not created:", str(exc)[:120])
