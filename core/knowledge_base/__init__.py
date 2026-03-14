# ==== core/knowledge_base/__init__.py ====
"""
Personal LLM Orchestrator — Knowledge Base Package.

Provides the full RAG ingestion and retrieval stack:

- ``document_loader`` — File discovery, text normalisation, smart chunking.
- ``vector_store``    — Embedding, ChromaDB persistence, similarity search.

Typical pipeline::

    from core.knowledge_base.document_loader import DocumentLoader
    from core.knowledge_base.vector_store import VectorStore

    loader = DocumentLoader(data_dir="./data/documents")
    store  = VectorStore(persist_dir="./data/chroma_store")

    chunks = loader.load_and_chunk()
    store.index_documents([c.to_dict() for c in chunks])

    results = store.similarity_search("What is RAG?", k=5)
"""

from core.knowledge_base.document_loader import (
    DocumentChunk,
    DocumentLoader,
    RawDocument,
    TextChunker,
)
from core.knowledge_base.vector_store import IndexResult, SearchResult, VectorStore

__all__: list[str] = [
    "DocumentLoader",
    "DocumentChunk",
    "RawDocument",
    "TextChunker",
    "VectorStore",
    "SearchResult",
    "IndexResult",
]
