"""Markdown chunking with LlamaIndex node parsers.

A document is split on its headings first, then any section longer than
chunk_size words is split on sentence boundaries. Chunk ids are derived from
(path, content hash, position), so re-chunking unchanged content gives the same
ids and an upsert is idempotent.
"""
from __future__ import annotations

import re
import uuid

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import MetadataMode

_NAMESPACE = uuid.UUID("6f1c1f0e-6f0b-4b7e-9f39-2c9d1f3a7b11")


def _word_tokens(text: str) -> list:
    # Word counts instead of a model tokenizer: no download, no model coupling.
    return text.split()


def _title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else fallback


def chunk_document(doc_path: str, text: str, doc_hash: str, dataset_version: str,
                   chunk_size: int = 220, chunk_overlap: int = 30) -> list:
    """Return a list of JSON-serialisable chunks:
    {"id", "text", "embed_text", "metadata": {doc_path, doc_hash, title, section,
    chunk_index, dataset_version}}."""
    title = _title(text, doc_path)
    document = Document(text=text, metadata={"doc_path": doc_path})
    sections = MarkdownNodeParser(include_metadata=True).get_nodes_from_documents([document])
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                tokenizer=_word_tokens)
    chunks = []
    for section in sections:
        body = section.get_content(metadata_mode=MetadataMode.NONE).strip()
        if not body:
            continue
        heading = _title(re.sub(r"^#{2,6}", "#", body, flags=re.MULTILINE), title)
        for piece in splitter.split_text(body):
            piece = piece.strip()
            # A section that is only a heading line carries no content.
            if not piece or re.fullmatch(r"#{1,6}\s+.*", piece):
                continue
            index = len(chunks)
            chunks.append({
                "id": str(uuid.uuid5(_NAMESPACE, "%s|%s|%d" % (doc_path, doc_hash, index))),
                "text": piece,
                # The title is prepended for embedding so that short sections
                # still carry the document subject.
                "embed_text": piece if heading == title else "%s\n%s" % (title, piece),
                "metadata": {
                    "doc_path": doc_path,
                    "doc_hash": doc_hash,
                    "title": title,
                    "section": heading,
                    "chunk_index": index,
                    "dataset_version": dataset_version,
                },
            })
    return chunks
