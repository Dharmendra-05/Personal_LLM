# ==== core/knowledge_base/vector_store.py ====
"""
ChromaDB-backed vector store with local sentence-transformer embeddings.

Architecture
------------
:class:`VectorStore` is the single component responsible for:

1. **Embedding** — Converts raw text into dense vectors using a local
   ``sentence-transformers`` model.  The embedding model is loaded once
   and held in memory for the lifetime of the ``VectorStore`` instance.

2. **Persistence** — Wraps a ``chromadb.PersistentClient`` whose data files
   survive process restarts.  The ChromaDB directory is created automatically
   if absent.

3. **Deduplication** — Every document chunk is assigned a deterministic ID
   computed as the SHA-256 digest of its text content.  Calling
   :meth:`index_documents` with documents already in the store is a no-op
   for those documents; only genuinely new chunks are written.

4. **Retrieval** — :meth:`similarity_search` embeds the query with the same
   model and returns the *k* most semantically similar chunks, each
   accompanied by their metadata and a normalised similarity score in
   ``[0.0, 1.0]``.

Deduplication strategy
----------------------
ChromaDB ``collection.upsert()`` is used instead of ``add()``.  When the
same SHA-256 ID already exists in the collection, ChromaDB silently skips
the write.  This means:

* Re-indexing the same directory is idempotent and fast.
* Modifying a document's text changes its hash → produces a new entry
  (the old entry is *not* automatically removed).

Distance metric
---------------
ChromaDB collections are created with ``cosine`` distance.  Raw distances
returned by ChromaDB (range ``[0.0, 2.0]`` for cosine) are converted to a
similarity score in ``[0.0, 1.0]`` using:

.. code-block:: python

    similarity = 1.0 - (distance / 2.0)

Batch processing
----------------
Both embedding and ChromaDB upserts are performed in configurable batches
(``batch_size``) to avoid OOM errors when indexing large corpora.

Usage
-----
.. code-block:: python

    from core.knowledge_base.vector_store import VectorStore

    store = VectorStore(
        persist_dir="./data/chroma_store",
        collection_name="my_knowledge_base",
        embedding_model_name="all-MiniLM-L6-v2",
    )
    store.index_documents(chunks_as_dicts)
    results = store.similarity_search("What is RAG?", k=3)
    for r in results:
        print(r["score"], r["text"][:80])
"""

from __future__ import annotations

import os
import hashlib
import threading
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from core.exceptions import (
    CollectionNotFoundError,
    EmbeddingError,
    OrchestratorValidationError,
    VectorStoreError,
)
from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# --- ChromaDB Telemetry & ONNX Suppression ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ORT_LOG_LEVEL"] = "3" 

try:
    import chromadb.telemetry.product
    import chromadb.telemetry.posthog
    import chromadb.telemetry.segment
    def _noop(*args, **kwargs): 
        return None
    for mod in [chromadb.telemetry.product, chromadb.telemetry.posthog, chromadb.telemetry.segment]:
        for cls_name in ["Posthog", "AnonymousTelemetry", "Sentry"]:
            if hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                cls.capture = _noop
                cls.send = _noop
                cls.__init__ = _noop
except Exception:
    pass
# ---------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of similarity results returned per query.
_DEFAULT_TOP_K: Final[int] = 5

#: Minimum cosine similarity score to include in results (applied post-query).
_MIN_SIMILARITY_SCORE: Final[float] = 0.0

#: ChromaDB distance metric used for all collections.
_DISTANCE_METRIC: Final[str] = "cosine"

#: Metadata key used to store the original text alongside the embedding.
_TEXT_METADATA_KEY: Final[str] = "text"

#: Maximum number of characters of chunk text logged at DEBUG level.
_LOG_TEXT_PREVIEW_LEN: Final[int] = 60


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single result from a :meth:`VectorStore.similarity_search` call.

    Attributes:
        doc_id: The deterministic SHA-256-based document ID.
        text: The chunk's original text content.
        score: Cosine similarity score in ``[0.0, 1.0]``.  Higher is better.
        metadata: All metadata fields stored alongside this chunk in
            ChromaDB (``filename``, ``chunk_index``, ``source_path``, etc.).
        distance: The raw ChromaDB cosine distance ``[0.0, 2.0]`` before
            conversion to *score*.
    """

    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    distance: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Returns:
            Dictionary with all fields of this search result.
        """
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "distance": self.distance,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """ChromaDB-backed semantic vector store with local embedding generation.

    Args:
        persist_dir: Filesystem path where ChromaDB persists its SQLite
            database and segment data.  Created automatically if absent.
        collection_name: Name of the ChromaDB collection to use.  The
            collection is created (with cosine distance) if it does not
            already exist.
        embedding_model_name: HuggingFace sentence-transformers model
            identifier or local path.  Defaults to ``"all-MiniLM-L6-v2"``.
        device: Compute device for the embedding model.  One of
            ``"cpu"``, ``"cuda"``, ``"mps"``.  Defaults to ``"cpu"``.
        embedding_batch_size: Number of texts encoded in a single forward
            pass.  Tune based on available memory.  Defaults to 64.
        top_k: Default number of results for :meth:`similarity_search`
            when ``k`` is not specified.  Defaults to 5.

    Raises:
        VectorStoreError: If the ChromaDB client cannot be initialised or
            the collection cannot be created/retrieved.
        EmbeddingError: If the sentence-transformer model fails to load.

    Example:
        >>> store = VectorStore(persist_dir="./data/chroma")
        >>> store.index_documents([{"text": "Hello world", "filename": "a.txt"}])
        >>> results = store.similarity_search("greeting", k=2)
    """

    def __init__(
        self,
        persist_dir: str | Path = "./data/chroma_store",
        collection_name: str = "llm_orchestrator_default",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        embedding_batch_size: int = 64,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self.persist_dir: Path = Path(persist_dir).resolve()
        self.collection_name: str = collection_name
        self.embedding_model_name: str = embedding_model_name
        self.device: str = device
        self.embedding_batch_size: int = embedding_batch_size
        self.top_k: int = top_k

        # Lazy-initialised internals (initialised on first property access)
        self._embedding_model: Any = None   # sentence_transformers.SentenceTransformer
        self._chroma_client: Any = None     # chromadb.PersistentClient
        self._collection: Any = None        # chromadb.Collection
        self._init_lock = threading.RLock()

        # Only the directory needs to exist immediately.
        self._init_persist_directory()

        logger.info(
            "VectorStore initialized (lazy): collection=%s persist_dir=%s",
            self.collection_name,
            self.persist_dir,
        )

    # ------------------------------------------------------------------
    # Lazy Properties
    # ------------------------------------------------------------------

    @property
    def embedding_model(self) -> Any:
        """Access the embedding model, loading it if necessary."""
        if self._embedding_model is None:
            with self._init_lock:
                if self._embedding_model is None:
                    self._init_embedding_model()
        return self._embedding_model

    @property
    def chroma_client(self) -> Any:
        """Access the ChromaDB client, loading it if necessary."""
        if self._chroma_client is None:
            with self._init_lock:
                if self._chroma_client is None:
                    self._init_chroma_client()
        return self._chroma_client

    @property
    def collection(self) -> Any:
        """Access the ChromaDB collection, loading it if necessary."""
        if self._collection is None:
            with self._init_lock:
                if self._collection is None:
                    # Depends on client
                    _ = self.chroma_client
                    self._init_collection()
        return self._collection

    @property
    def is_loaded(self) -> bool:
        """Return True if all lazy components have been initialised."""
        return self._embedding_model is not None and self._collection is not None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[dict[str, Any]]) -> IndexResult:
        """Embed and upsert a list of document chunk dicts into the collection.

        Documents that are already present (same SHA-256 content hash) are
        silently skipped, making this operation fully idempotent.

        Args:
            documents: List of dicts, each representing one text chunk.
                **Required key**: ``"text"`` — the chunk's string content.
                **Optional keys** (stored as metadata):
                ``"filename"``, ``"source_path"``, ``"chunk_index"``,
                ``"total_chunks"``, ``"char_start"``, ``"char_end"``,
                ``"has_overlap"``, and any other keys in ``"metadata"``.

        Returns:
            An :class:`IndexResult` summary with counts of added, skipped,
            and failed documents.

        Raises:
            OrchestratorValidationError: If ``documents`` is empty.
            EmbeddingError: If the embedding model fails on any batch.
            VectorStoreError: If ChromaDB upsert fails.

        Example:
            >>> result = store.index_documents(chunks)
            >>> print(f"Added {result.added}, skipped {result.skipped}")
        """
        if not documents:
            raise OrchestratorValidationError(
                message="index_documents() received an empty document list.",
                field="documents",
                received=[],
            )

        # --- Validate and pre-filter ---
        valid_docs: list[dict[str, Any]] = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning("Document at index %d is not a dict; skipping.", i)
                continue
            if "text" not in doc or not doc["text"] or not str(doc["text"]).strip():
                logger.warning(
                    "Document at index %d missing or empty 'text' field; skipping.", i
                )
                continue
            valid_docs.append(doc)

        if not valid_docs:
            raise OrchestratorValidationError(
                message="No valid documents (with non-empty 'text') found in the list.",
                field="documents",
                received=f"{len(documents)} items, all invalid",
            )

        logger.info(
            "VectorStore.index_documents(): %d valid docs to process "
            "(%d invalid skipped)",
            len(valid_docs),
            len(documents) - len(valid_docs),
        )

        # --- Process in batches ---
        total_added: int = 0
        total_skipped: int = 0
        total_failed: int = 0

        # Process in parallel batches
        batches = [
            valid_docs[i : i + self.embedding_batch_size]
            for i in range(0, len(valid_docs), self.embedding_batch_size)
        ]

        with self._init_lock:
            # Process batches
            if len(batches) <= 1:
                # Sequential processing for a single batch to avoid thread overhead
                for batch in batches:
                    added, skipped, failed = self._upsert_batch(batch)
                    total_added += added
                    total_skipped += skipped
                    total_failed += failed
            else:
                # Parallel processing for multiple batches
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self._upsert_batch, batch) for batch in batches]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            added, skipped, failed = future.result()
                            total_added += added
                            total_skipped += skipped
                            total_failed += failed
                        except (EmbeddingError, VectorStoreError):
                            raise
                        except Exception as exc:
                            raise VectorStoreError(
                                message=f"Unexpected error during batch upsert: {exc}",
                                collection=self.collection_name,
                                error_code="VS_001",
                            ) from exc

        result = IndexResult(
            added=total_added,
            skipped=total_skipped,
            failed=total_failed,
            total_processed=len(valid_docs),
            collection_name=self.collection_name,
        )
        logger.info(
            "VectorStore.index_documents() complete: %s", result
        )
        return result

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        min_score: float = _MIN_SIMILARITY_SCORE,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve the *k* most semantically similar chunks for *query*.

        The query is embedded with the same model used during indexing and
        the resulting vector is used to query ChromaDB's approximate nearest
        neighbour index.

        Args:
            query: The natural-language search query string.
            k: Number of results to return.  Defaults to :attr:`top_k`.
                Clamped to the number of documents in the collection.
            min_score: Minimum similarity score (inclusive) for a result to
                be included.  Defaults to ``0.0`` (no filter).
            where: Optional ChromaDB ``where`` metadata filter dict.
                Example: ``{"filename": {"$eq": "report.txt"}}``.

        Returns:
            Ordered list of :class:`SearchResult` objects (highest score
            first).  May be shorter than *k* if fewer matching documents
            exist or ``min_score`` filters some out.

        Raises:
            OrchestratorValidationError: If ``query`` is empty.
            EmbeddingError: If query embedding fails.
            VectorStoreError: If the ChromaDB query fails.
            CollectionNotFoundError: If the collection was dropped externally
                between construction and this call.

        Example:
            >>> results = store.similarity_search("What is retrieval augmentation?")
            >>> for r in results:
            ...     print(f"{r.score:.3f}  {r.text[:60]}")
        """
        if not query or not query.strip():
            raise OrchestratorValidationError(
                message="similarity_search() query must not be empty.",
                field="query",
                received=query,
            )

        effective_k: int = k if k is not None else self.top_k
        if effective_k < 1:
            raise OrchestratorValidationError(
                message="k must be ≥ 1.",
                field="k",
                received=effective_k,
            )

        # Clamp k to collection size to avoid ChromaDB errors
        collection_count: int = self._safe_collection_count()
        if collection_count == 0:
            logger.warning(
                "similarity_search(): collection '%s' is empty; returning []",
                self.collection_name,
            )
            return []

        effective_k = min(effective_k, collection_count)

        # --- Embed query ---
        logger.debug(
            "similarity_search(): embedding query (len=%d chars)", len(query)
        )
        query_embedding: list[float] = self._embed_texts([query])[0]

        # --- Query ChromaDB ---
        logger.debug(
            "similarity_search(): querying collection '%s' for k=%d",
            self.collection_name,
            effective_k,
        )
        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": effective_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                query_kwargs["where"] = where

            raw: dict[str, Any] = self.collection.query(**query_kwargs)
        except Exception as exc:
            raise VectorStoreError(
                message=f"ChromaDB query failed: {exc}",
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc

        # --- Parse results ---
        results: list[SearchResult] = self._parse_query_results(raw, min_score)

        logger.debug(
            "similarity_search(): returned %d result(s) for query='%s...'",
            len(results),
            query[:_LOG_TEXT_PREVIEW_LEN],
        )
        return results

    def collection_count(self) -> int:
        """Return the number of documents currently indexed in the collection.

        Returns:
            Integer document count, or 0 if the collection is empty or
            an error occurs.
        """
        return self._safe_collection_count()

    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB.

        This is irreversible.  The :class:`VectorStore` instance becomes
        unusable after this call unless :meth:`_init_collection` is called
        again.

        Raises:
            VectorStoreError: If ChromaDB raises during deletion.
        """
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self._collection = None
            logger.warning(
                "VectorStore.delete_collection(): '%s' has been deleted.",
                self.collection_name,
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Failed to delete collection '{self.collection_name}': {exc}",
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc

    def get_document_by_id(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a single document by its deterministic ID.

        Args:
            doc_id: The SHA-256-based document ID as returned by
                :func:`generate_doc_id`.

        Returns:
            A dict with keys ``"doc_id"``, ``"text"``, and ``"metadata"``,
            or ``None`` if no document with that ID exists.

        Raises:
            VectorStoreError: If the ChromaDB get fails.
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"ChromaDB get failed for id '{doc_id}': {exc}",
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc

        ids: list[str] = result.get("ids", [])
        if not ids:
            return None

        docs: list[str | None] = result.get("documents", [None])
        metas: list[dict[str, Any]] = result.get("metadatas", [{}])

        return {
            "doc_id": ids[0],
            "text": docs[0] or "",
            "metadata": metas[0] if metas else {},
        }

    # ------------------------------------------------------------------
    # Static / class-level utilities
    # ------------------------------------------------------------------

    @staticmethod
    def generate_doc_id(text: str) -> str:
        """Compute a deterministic, collision-resistant document ID.

        Uses the SHA-256 digest of the UTF-8 encoded text.  The same text
        always produces the same ID; different texts (with overwhelming
        probability) produce different IDs.

        Args:
            text: The chunk's text content.

        Returns:
            A 64-character lowercase hexadecimal string (256-bit SHA-256
            digest).

        Example:
            >>> VectorStore.generate_doc_id("Hello world")
            'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e'
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_persist_directory(self) -> None:
        """Create the ChromaDB persistence directory if it does not exist.

        Raises:
            VectorStoreError: If the directory cannot be created.
        """
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "VectorStore: persist directory ready: %s", self.persist_dir
            )
        except OSError as exc:
            raise VectorStoreError(
                message=(
                    f"Cannot create ChromaDB persist directory "
                    f"'{self.persist_dir}': {exc}"
                ),
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc

    def _init_embedding_model(self) -> None:
        """Load the sentence-transformer embedding model.

        Raises:
            EmbeddingError: If the model cannot be loaded (missing package,
                invalid model name, OOM, etc.).
        """
        logger.info(
            "VectorStore: loading embedding model '%s' on device='%s'",
            self.embedding_model_name,
            self.device,
        )
        try:
            # sentence-transformers is imported here (not at module level)
            # to avoid a hard import-time dependency when the module is
            # imported in a context where the package may not be installed.
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )
            logger.info(
                "VectorStore: embedding model '%s' loaded successfully.",
                self.embedding_model_name,
            )
        except ImportError as exc:
            raise EmbeddingError(
                message=(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ),
                model_name=self.embedding_model_name,
                error_code="VS_003",
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                message=(
                    f"Failed to load embedding model "
                    f"'{self.embedding_model_name}': {exc}"
                ),
                model_name=self.embedding_model_name,
                error_code="VS_003",
            ) from exc

    def _init_chroma_client(self) -> None:
        """Initialise the ChromaDB PersistentClient.

        Raises:
            VectorStoreError: If ChromaDB cannot be initialised.
        """
        logger.debug(
            "VectorStore: initialising ChromaDB PersistentClient at %s",
            self.persist_dir,
        )
        try:
            import chromadb  # noqa: PLC0415

            self._chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            logger.debug("VectorStore: ChromaDB PersistentClient ready.")
        except ImportError as exc:
            raise VectorStoreError(
                message="chromadb is not installed. Run: pip install chromadb",
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc
        except Exception as exc:
            raise VectorStoreError(
                message=f"ChromaDB PersistentClient failed to initialise: {exc}",
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc

    def _init_collection(self) -> None:
        """Get or create the ChromaDB collection with cosine distance.

        Raises:
            VectorStoreError: If the collection cannot be obtained or created.
        """
        logger.debug(
            "VectorStore: getting/creating collection '%s' (metric=%s)",
            self.collection_name,
            _DISTANCE_METRIC,
        )
        try:
            self._collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": _DISTANCE_METRIC},
            )
            count: int = self.collection.count()
            logger.info(
                "VectorStore: collection '%s' ready (%d documents indexed).",
                self.collection_name,
                count,
            )
        except Exception as exc:
            raise VectorStoreError(
                message=(
                    f"Failed to get/create ChromaDB collection "
                    f"'{self.collection_name}': {exc}"
                ),
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of strings into embedding vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of embedding vectors (each a ``list[float]``), in the same
            order as *texts*.

        Raises:
            EmbeddingError: If the model raises during encoding.
        """
        if not texts:
            return []

        logger.debug(
            "VectorStore._embed_texts(): encoding %d text(s) with '%s'",
            len(texts),
            self.embedding_model_name,
        )
        try:
            embeddings_np = self.embedding_model.encode(
                texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2-normalise for cosine similarity
            )
            # Convert NumPy array rows to plain Python lists for ChromaDB
            return [vec.tolist() for vec in embeddings_np]
        except Exception as exc:
            raise EmbeddingError(
                message=f"Embedding model '{self.embedding_model_name}' failed: {exc}",
                model_name=self.embedding_model_name,
                error_code="VS_003",
            ) from exc

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    def _upsert_batch(
        self, batch: list[dict[str, Any]]
    ) -> tuple[int, int, int]:
        """Embed and upsert one batch of document dicts.

        Args:
            batch: A slice of the validated documents list.

        Returns:
            A 3-tuple ``(added, skipped, failed)`` counting outcomes.

        Raises:
            EmbeddingError: On embedding failure.
            VectorStoreError: On ChromaDB upsert failure.
        """
        texts: list[str] = [str(doc["text"]) for doc in batch]

        # Generate deterministic IDs
        ids: list[str] = [self.generate_doc_id(t) for t in texts]

        # Check which IDs already exist (avoid redundant embedding work)
        existing_ids: set[str] = self._get_existing_ids(ids)

        new_indices: list[int] = [
            i for i, doc_id in enumerate(ids) if doc_id not in existing_ids
        ]
        skipped: int = len(ids) - len(new_indices)

        if not new_indices:
            logger.debug(
                "VectorStore._upsert_batch(): all %d documents already exist; "
                "skipping batch.",
                len(batch),
            )
            return 0, skipped, 0

        # Embed only new documents
        new_texts: list[str] = [texts[i] for i in new_indices]
        new_ids: list[str] = [ids[i] for i in new_indices]
        new_docs: list[dict[str, Any]] = [batch[i] for i in new_indices]

        embeddings: list[list[float]] = self._embed_texts(new_texts)

        # Build metadata dicts (exclude 'text', 'metadata' nesting)
        metadatas: list[dict[str, Any]] = [
            self._build_metadata(doc) for doc in new_docs
        ]

        # Upsert into ChromaDB
        try:
            self.collection.upsert(
                ids=new_ids,
                embeddings=embeddings,
                documents=new_texts,   # stored as the "document" field
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                message=f"ChromaDB upsert failed for batch of {len(new_ids)}: {exc}",
                collection=self.collection_name,
                error_code="VS_001",
            ) from exc

        logger.debug(
            "VectorStore._upsert_batch(): upserted %d new, skipped %d existing",
            len(new_ids),
            skipped,
        )
        return len(new_ids), skipped, 0

    def _get_existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of *ids* that already exist in the collection.

        Args:
            ids: List of candidate document IDs to check.

        Returns:
            Set of IDs that are already present.  Returns an empty set on
            any error (erring on the side of re-upserting rather than
            silently skipping).
        """
        try:
            result = self.collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception as exc:
            logger.warning(
                "VectorStore._get_existing_ids(): check failed (%s); "
                "proceeding without deduplication for this batch.",
                exc,
            )
            return set()

    @staticmethod
    def _build_metadata(doc: dict[str, Any]) -> dict[str, Any]:
        """Extract and flatten metadata from a document dict for ChromaDB.

        ChromaDB metadata values must be ``str``, ``int``, ``float``, or
        ``bool``.  Any other types are coerced to strings.

        Args:
            doc: A document chunk dict from the caller.

        Returns:
            A flat dict of metadata key/value pairs safe for ChromaDB storage.
            The ``"text"`` key is intentionally excluded (stored as the
            ``documents`` field in ChromaDB, not duplicated in metadata).
        """
        # Keys to lift directly into metadata
        _DIRECT_KEYS: frozenset[str] = frozenset({
            "filename", "source_path", "chunk_index", "total_chunks",
            "char_start", "char_end", "has_overlap", "encoding",
        })
        _CHROMA_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool)

        meta: dict[str, Any] = {}

        for key in _DIRECT_KEYS:
            if key in doc:
                val = doc[key]
                meta[key] = (
                    val if isinstance(val, _CHROMA_SCALAR_TYPES) else str(val)
                )

        # Merge any nested "metadata" dict from DocumentChunk.to_dict()
        nested: dict[str, Any] = doc.get("metadata", {})
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k not in meta:   # direct keys take precedence
                    meta[k] = (
                        v if isinstance(v, _CHROMA_SCALAR_TYPES) else str(v)
                    )

        return meta

    # ------------------------------------------------------------------
    # Query result parsing
    # ------------------------------------------------------------------

    def _parse_query_results(
        self,
        raw: dict[str, Any],
        min_score: float,
    ) -> list[SearchResult]:
        """Convert raw ChromaDB query output into :class:`SearchResult` objects.

        ChromaDB returns nested lists (one sub-list per query vector).  Since
        we always query with a single vector, we unwrap the first sub-list.

        Args:
            raw: The raw dict returned by ``collection.query()``.
            min_score: Minimum similarity score filter (inclusive).

        Returns:
            Filtered and sorted list of :class:`SearchResult` objects.
        """
        ids_outer: list[list[str]] = raw.get("ids", [[]])
        docs_outer: list[list[str | None]] = raw.get("documents", [[]])
        metas_outer: list[list[dict[str, Any]]] = raw.get("metadatas", [[]])
        dists_outer: list[list[float]] = raw.get("distances", [[]])

        ids: list[str] = ids_outer[0] if ids_outer else []
        docs: list[str | None] = docs_outer[0] if docs_outer else []
        metas: list[dict[str, Any]] = metas_outer[0] if metas_outer else []
        dists: list[float] = dists_outer[0] if dists_outer else []

        results: list[SearchResult] = []

        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Convert cosine distance [0, 2] → similarity [0, 1]
            # (L2-normalised embeddings: dist=0 → identical, dist=2 → opposite)
            score: float = max(0.0, min(1.0, 1.0 - dist / 2.0))

            if score < min_score:
                logger.debug(
                    "similarity_search(): dropping id=%s score=%.4f < min=%.4f",
                    doc_id,
                    score,
                    min_score,
                )
                continue

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    text=text or "",
                    score=score,
                    metadata=meta or {},
                    distance=dist,
                )
            )

        # Already ordered by distance from ChromaDB, but re-sort to be safe
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _safe_collection_count(self) -> int:
        """Return the collection size without raising on errors.

        Returns:
            Integer document count, or 0 on any error.
        """
        try:
            return self.collection.count()
        except Exception as exc:
            logger.warning(
                "VectorStore: could not get collection count: %s", exc
            )
            return 0

    def __repr__(self) -> str:
        count: int = self._safe_collection_count()
        return (
            f"VectorStore("
            f"collection={self.collection_name!r}, "
            f"model={self.embedding_model_name!r}, "
            f"device={self.device!r}, "
            f"docs={count})"
        )


# ---------------------------------------------------------------------------
# IndexResult summary dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Summary of a completed :meth:`VectorStore.index_documents` call.

    Attributes:
        added: Number of new documents written to the collection.
        skipped: Number of documents that already existed (same SHA-256 ID).
        failed: Number of documents that could not be indexed due to errors.
        total_processed: Total number of valid documents submitted.
        collection_name: Name of the target collection.
    """

    added: int
    skipped: int
    failed: int
    total_processed: int
    collection_name: str

    def __str__(self) -> str:
        return (
            f"IndexResult(added={self.added}, skipped={self.skipped}, "
            f"failed={self.failed}, total={self.total_processed}, "
            f"collection={self.collection_name!r})"
        )


# ---------------------------------------------------------------------------
# Public re-export surface
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "VectorStore",
    "SearchResult",
    "IndexResult",
]
