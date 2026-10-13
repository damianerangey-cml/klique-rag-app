"""Shared library for the Klique RAG example: configuration, dataset access,
chunking, vector store helpers, ingestion steps and the retrieval check.

The pipeline (pipeline/), the middleware (service/) and the tests all import
from here, so ingestion and query use the same chunk format and the same
embedding client.
"""
