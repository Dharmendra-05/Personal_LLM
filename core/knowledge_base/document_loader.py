# ==== core/knowledge_base/document_loader.py ====
"""
Document loading and intelligent text chunking for the RAG knowledge base.

Overview
--------
This module is responsible for the first two stages of any RAG ingestion
pipeline:

1. **Loading** — Discover and read ``.txt`` files from a source directory,
   producing structured :class:`RawDocument` objects with rich file metadata.
2. **Chunking** — Split each document's text into overlapping segments that
   fit within an embedding model's context window, using a multi-tier
   boundary detection strategy to avoid mid-word and mid-sentence splits.

Chunking strategy
-----------------
The :class:`TextChunker` implements a *cascading boundary preference*:

.. code-block:: text

    Priority 1: Paragraph boundary  (double-newline  "\\n\\n")
    Priority 2: Sentence boundary   (". " / "? " / "! " / "\\n")
    Priority 3: Word boundary       (single space)
    Priority 4: Hard cut            (last resort, no boundary found)

For each chunk window, the splitter scans *backwards* from the ideal
``chunk_size`` position to find the best available boundary within a
configurable ``boundary_search_window``.  This guarantees that:

* No token or word is split in half at the chunk edge.
* Sentence and paragraph structure is preserved wherever possible.
* The overlap region always starts at a clean sentence/word boundary.

Overlap semantics
-----------------
``overlap`` tokens (characters) of the *previous* chunk's tail are prepended
to each new chunk.  The overlap region is trimmed to the nearest word
boundary so it never starts mid-word.

Output schema
-------------
Every chunk is emitted as a :class:`DocumentChunk` dataclass and can be
converted to a plain ``dict`` via :meth:`DocumentChunk.to_dict` for
downstream consumption by :class:`~core.knowledge_base.vector_store.VectorStore`.

Usage
-----
.. code-block:: python

    from core.knowledge_base.document_loader import DocumentLoader

    loader = DocumentLoader(
        data_dir="./data/documents",
        chunk_size=512,
        overlap=64,
    )
    chunks = loader.load_and_chunk()
    for chunk in chunks:
        print(chunk.chunk_index, chunk.text[:80])
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterator, Sequence

from core.exceptions import OrchestratorValidationError, PersonalLLMException
from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: File extensions that the loader will attempt to read.
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".txt", ".pdf", ".docx", ".doc", ".xlsx",
    ".png", ".jpg", ".jpeg", ".mp3", ".wav"
})

#: Regex that matches one or more blank lines (paragraph separator).
_PARAGRAPH_BREAK_RE: Final[re.Pattern[str]] = re.compile(r"\n{2,}")

#: Sentence-ending punctuation patterns used for boundary detection.
_SENTENCE_ENDS: Final[tuple[str, ...]] = (". ", "? ", "! ", ".\n", "?\n", "!\n")

#: Minimum chunk size (in characters) below which a chunk is discarded as
#: too small to carry meaningful semantic content.
_MIN_CHUNK_CHARS: Final[int] = 20


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawDocument:
    """Immutable value object representing one loaded source file.

    Attributes:
        source_path: Absolute path to the original file.
        filename: Bare filename (``stem + suffix``), e.g. ``"README.txt"``.
        text: Full UTF-8 decoded text content of the file.
        size_bytes: File size in bytes at load time.
        encoding: The character encoding that was used to decode the file.
    """

    source_path: Path
    filename: str
    text: str
    size_bytes: int
    encoding: str = "utf-8"


@dataclass(slots=True)
class DocumentChunk:
    """A single text chunk produced by :class:`TextChunker`.

    Attributes:
        text: The chunk's text content (may include overlap prefix).
        chunk_index: Zero-based position of this chunk within its source
            document.
        total_chunks: Total number of chunks produced from the source document.
        source_path: Absolute path to the originating file.
        filename: Bare filename of the source document.
        char_start: Character offset in the *original* (pre-overlap) text
            where this chunk begins.
        char_end: Character offset in the original text where this chunk ends.
        has_overlap: ``True`` if this chunk carries an overlap prefix from
            the previous chunk.
        metadata: Arbitrary extra key/value pairs for downstream use.
    """

    text: str
    chunk_index: int
    total_chunks: int
    source_path: Path
    filename: str
    char_start: int
    char_end: int
    has_overlap: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise this chunk to a plain ``dict`` for vector store ingestion.

        Returns:
            A flat dictionary with all chunk fields plus a flattened
            ``"source_path"`` string (not a ``Path`` object) for JSON
            compatibility.

        Example:
            >>> chunk.to_dict()
            {
                "text": "...",
                "chunk_index": 0,
                "total_chunks": 4,
                "source_path": "/abs/path/doc.txt",
                "filename": "doc.txt",
                "char_start": 0,
                "char_end": 512,
                "has_overlap": False,
                "metadata": {},
            }
        """
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_path": str(self.source_path),
            "filename": self.filename,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "has_overlap": self.has_overlap,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_text(raw: str) -> str:
    """Apply lightweight Unicode normalisation and whitespace cleanup.

    Performs:
    * NFC Unicode normalisation (combines combining characters).
    * Replacement of Windows-style ``\\r\\n`` line endings with ``\\n``.
    * Replacement of non-breaking spaces (U+00A0) with regular spaces.
    * Removal of null bytes and other C0 control characters (except ``\\t``,
      ``\\n``).
    * Collapsing of runs of more than 3 consecutive blank lines into exactly
      2 (preserving intentional section breaks without excessive whitespace).

    Args:
        raw: The raw string read directly from disk.

    Returns:
        Normalised string suitable for chunking.
    """
    # NFC: combine decomposed characters (e.g. é = e + combining acute)
    text: str = unicodedata.normalize("NFC", raw)

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Non-breaking space → regular space
    text = text.replace("\u00a0", " ")

    # Remove C0 control characters except \t (0x09) and \n (0x0A)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)

    # Collapse excessive blank lines (4+ newlines → 2)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Text chunker
# ---------------------------------------------------------------------------


class TextChunker:
    """Splits a long string into overlapping chunks with smart boundary detection.

    The chunker uses a cascading priority system when deciding where to end
    each chunk:

    1. **Paragraph boundary** (``\\n\\n``) — highest semantic signal.
    2. **Sentence boundary** (``". "``, ``"? "``, ``"! "``, etc.).
    3. **Word boundary** (space character).
    4. **Hard cut** — only if no boundary is found in the search window.

    For each chunk, the algorithm scans *backwards* from the ideal end
    position by at most ``boundary_search_window`` characters looking for the
    best available boundary.

    Args:
        chunk_size: Target maximum character count per chunk (excluding the
            overlap prefix).  Chunks may be slightly shorter if a boundary
            is found before the target.  Must be ≥ 50.
        overlap: Number of characters from the tail of the previous chunk
            to prepend to the next chunk.  Enables continuity across chunk
            boundaries.  Must be ≥ 0 and < ``chunk_size``.
        boundary_search_window: Maximum number of characters to scan
            backwards from the ideal end position when looking for a
            clean boundary.  Defaults to ``min(chunk_size // 4, 200)``.

    Raises:
        OrchestratorValidationError: If ``chunk_size`` or ``overlap`` values
            violate their constraints.

    Example:
        >>> chunker = TextChunker(chunk_size=512, overlap=64)
        >>> segments = chunker.chunk("Very long document text ...")
        >>> len(segments)
        3
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        boundary_search_window: int | None = None,
    ) -> None:
        if chunk_size < 50:
            raise OrchestratorValidationError(
                message="chunk_size must be ≥ 50 characters.",
                field="chunk_size",
                received=chunk_size,
            )
        if overlap < 0:
            raise OrchestratorValidationError(
                message="overlap must be ≥ 0.",
                field="overlap",
                received=overlap,
            )
        if overlap >= chunk_size:
            raise OrchestratorValidationError(
                message="overlap must be strictly less than chunk_size.",
                field="overlap",
                received=overlap,
            )

        self.chunk_size: int = chunk_size
        self.overlap: int = overlap
        self.boundary_search_window: int = (
            boundary_search_window
            if boundary_search_window is not None
            else min(chunk_size // 4, 200)
        )

        logger.debug(
            "TextChunker initialised: chunk_size=%d overlap=%d search_window=%d",
            self.chunk_size,
            self.overlap,
            self.boundary_search_window,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of overlapping chunk strings.

        Empty or whitespace-only text returns an empty list.  Text shorter
        than ``chunk_size`` is returned as a single chunk with no splitting.

        Args:
            text: The normalised source text to split.

        Returns:
            Ordered list of chunk strings.  Each string (except possibly
            the first) begins with up to ``overlap`` characters from the
            tail of the previous chunk.
        """
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks: list[str] = []
        cursor: int = 0  # Current position in ``text``
        text_len: int = len(text)

        while cursor < text_len:
            # Ideal end position (hard ceiling)
            ideal_end: int = min(cursor + self.chunk_size, text_len)

            if ideal_end == text_len:
                # Last chunk — take everything that remains
                segment: str = text[cursor:].strip()
                if len(segment) >= _MIN_CHUNK_CHARS:
                    chunks.append(segment)
                break

            # Find the best boundary at or before ideal_end
            split_pos: int = self._find_boundary(text, ideal_end)

            segment = text[cursor:split_pos].strip()
            if len(segment) >= _MIN_CHUNK_CHARS:
                chunks.append(segment)

            # Advance cursor: skip past the split position, then step back
            # by ``overlap`` to create the overlap region for the next chunk.
            next_start: int = split_pos
            if self.overlap > 0 and next_start > self.overlap:
                # Step back by overlap, but land on a clean word boundary
                overlap_start: int = next_start - self.overlap
                overlap_start = self._snap_to_word_boundary(
                    text, overlap_start, direction="forward"
                )
                next_start = max(overlap_start, cursor + 1)

            # Prevent infinite loop: guarantee forward progress
            if next_start <= cursor:
                next_start = cursor + max(1, self.chunk_size // 2)

            cursor = next_start

        logger.debug(
            "TextChunker.chunk(): %d chars → %d chunks "
            "(chunk_size=%d, overlap=%d)",
            len(text),
            len(chunks),
            self.chunk_size,
            self.overlap,
        )
        return chunks

    # ------------------------------------------------------------------
    # Private boundary detection helpers
    # ------------------------------------------------------------------

    def _find_boundary(self, text: str, ideal_end: int) -> int:
        """Scan backwards from *ideal_end* to find the best split position.

        Boundary priority (highest to lowest):
        1. Paragraph break (``\\n\\n``)
        2. Sentence-end punctuation (``". "``, ``"? "``, ``"! "``, etc.)
        3. Single newline (``\\n``)
        4. Space (word boundary)
        5. Hard cut at *ideal_end* (fallback)

        Args:
            text: The full source text string.
            ideal_end: The character index of the ideal chunk end (inclusive).

        Returns:
            The chosen split position (character index in *text*).  The
            returned index points to the character *after* the boundary
            token, so the boundary itself is included in the current chunk.
        """
        search_start: int = max(0, ideal_end - self.boundary_search_window)
        window: str = text[search_start:ideal_end]

        # --- Priority 1: paragraph boundary (\n\n) ---
        pos: int = window.rfind("\n\n")
        if pos != -1:
            return search_start + pos + 2  # include the double-newline

        # --- Priority 2: sentence-ending punctuation ---
        best_sentence: int = -1
        for marker in _SENTENCE_ENDS:
            p = window.rfind(marker)
            if p != -1 and p > best_sentence:
                best_sentence = p
        if best_sentence != -1:
            return search_start + best_sentence + len(
                next(m for m in _SENTENCE_ENDS if window.rfind(m) == best_sentence)
            )

        # --- Priority 3: single newline ---
        pos = window.rfind("\n")
        if pos != -1:
            return search_start + pos + 1

        # --- Priority 4: word boundary (space) ---
        pos = window.rfind(" ")
        if pos != -1:
            return search_start + pos + 1

        # --- Fallback: hard cut ---
        return ideal_end

    def _snap_to_word_boundary(
        self,
        text: str,
        position: int,
        direction: str = "forward",
    ) -> int:
        """Move *position* to the nearest word boundary.

        Args:
            text: The source string.
            position: Starting character index.
            direction: ``"forward"`` advances to the next space/newline;
                ``"backward"`` retreats to the previous one.

        Returns:
            Adjusted position at a word boundary, clamped to valid indices.
        """
        text_len: int = len(text)
        pos: int = max(0, min(position, text_len - 1))

        if direction == "forward":
            while pos < text_len and text[pos] not in (" ", "\n"):
                pos += 1
            return min(pos + 1, text_len)
        else:
            while pos > 0 and text[pos] not in (" ", "\n"):
                pos -= 1
            return max(pos, 0)


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------


class DocumentLoader:
    """Discovers, reads, and chunks ``.txt`` files from a source directory.

    This class orchestrates the full ingestion pipeline:

    1. Scan ``data_dir`` (optionally recursively) for ``*.txt`` files.
    2. Read each file with configurable encoding fallback.
    3. Normalise the text (Unicode, whitespace).
    4. Pass the normalised text to :class:`TextChunker`.
    5. Emit :class:`DocumentChunk` objects tagged with source metadata.

    Args:
        data_dir: Path to the directory containing ``.txt`` source files.
            Must exist and be readable.
        chunk_size: Target chunk size in characters.  Passed to
            :class:`TextChunker`.  Defaults to 512.
        overlap: Overlap in characters between consecutive chunks.  Defaults
            to 64.
        recursive: If ``True``, scan subdirectories recursively.  Defaults
            to ``False``.
        encodings: Ordered list of encodings to try when opening each file.
            The first encoding that decodes without error is used.  Defaults
            to ``["utf-8", "latin-1"]``.
        boundary_search_window: Passed directly to :class:`TextChunker`.
            See its documentation for semantics.

    Raises:
        OrchestratorValidationError: If ``data_dir`` does not exist or is
            not a directory.

    Example:
        >>> loader = DocumentLoader(data_dir="./data/documents", chunk_size=512)
        >>> chunks = loader.load_and_chunk()
        >>> print(f"Loaded {len(chunks)} chunks from {loader.data_dir}")
    """

    def __init__(
        self,
        data_dir: str | Path,
        chunk_size: int = 512,
        overlap: int = 64,
        recursive: bool = False,
        encodings: list[str] | None = None,
        boundary_search_window: int | None = None,
        silent: bool = False,
    ) -> None:
        self.data_dir: Path = Path(data_dir).resolve()
        self.recursive: bool = recursive
        self.encodings: list[str] = encodings or ["utf-8", "latin-1"]
        self.silent: bool = silent
        self._manifest_path: Path = self.data_dir / ".scan_cache.json"
        self._manifest: dict[str, Any] = self._load_manifest()

        if not self.data_dir.exists():
            raise OrchestratorValidationError(
                message=f"data_dir does not exist: {self.data_dir}",
                field="data_dir",
                received=str(self.data_dir),
            )
        if not self.data_dir.is_dir():
            raise OrchestratorValidationError(
                message=f"data_dir is not a directory: {self.data_dir}",
                field="data_dir",
                received=str(self.data_dir),
            )

        self._chunker: TextChunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            boundary_search_window=boundary_search_window,
        )

        logger.info(
            "DocumentLoader initialised: dir=%s chunk_size=%d overlap=%d recursive=%s",
            self.data_dir,
            chunk_size,
            overlap,
            recursive,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_and_chunk(self) -> list[DocumentChunk]:
        """Load all ``.txt`` files and return a flat list of chunks.

        This is the primary entry point.  It calls :meth:`iter_chunks`
        internally and materialises the full list.

        Returns:
            A flat list of :class:`DocumentChunk` objects ordered by
            (filename, chunk_index).  Returns an empty list if no supported
            files are found in ``data_dir``.

        Raises:
            PersonalLLMException: Propagated from :meth:`_read_file` if a
                file cannot be decoded with any of the configured encodings.

        Example:
            >>> chunks = loader.load_and_chunk()
            >>> chunks[0].filename
            'document_a.txt'
        """
        all_chunks: list[DocumentChunk] = []
        
        # Parallel file loading and chunking
        txt_files: list[Path] = self._discover_files()
        if not txt_files:
            return []

        # Track which files were seen in this scan to prune the manifest later
        seen_files: set[str] = set()
        
        # Lazy threading: Avoid overhead for very small file sets
        if len(txt_files) <= 3:
            for f in txt_files:
                rel_path = str(f.relative_to(self.data_dir))
                seen_files.add(rel_path)
                file_chunks = self._process_file(f)
                if file_chunks:
                    all_chunks.extend(file_chunks)
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map _process_file to each path
                future_to_file = {executor.submit(self._process_file, f): f for f in txt_files}
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    seen_files.add(str(file_path.relative_to(self.data_dir)))
                    file_chunks = future.result()
                    if file_chunks:
                        all_chunks.extend(file_chunks)

        # Sort by filename and then index to maintain consistent ordering
        all_chunks.sort(key=lambda c: (c.filename, c.chunk_index))

        # Save manifest for next time
        self._save_manifest(seen_files)

        logger.info(
            "DocumentLoader.load_and_chunk(): %d chunks from %s",
            len(all_chunks),
            self.data_dir,
        )
        return all_chunks

    def _process_file(self, file_path: Path) -> list[DocumentChunk]:
        """Load and chunk a single file. Helper for parallel execution."""
        rel_path = str(file_path.relative_to(self.data_dir))
        try:
            stat = file_path.stat()
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            return []

        # Check manifest cache
        cached = self._manifest.get(rel_path)
        if cached and cached.get("mtime") == mtime and cached.get("size") == size:
            # Reconstruct chunks from manifest
            cached_chunks = cached.get("chunks", [])
            if cached_chunks:
                return [
                    DocumentChunk(
                        text=c["text"],
                        chunk_index=c["chunk_index"],
                        total_chunks=c["total_chunks"],
                        source_path=Path(c["source_path"]),
                        filename=c["filename"],
                        char_start=c["char_start"],
                        char_end=c["char_end"],
                        has_overlap=c["has_overlap"],
                    )
                    for c in cached_chunks
                ]

        # Not in cache or changed
        raw_doc: RawDocument | None = self._load_file(file_path)
        if raw_doc is None:
            return []
        
        chunks = list(self._chunk_document(raw_doc))
        
        # Update manifest entry (in-memory)
        self._manifest[rel_path] = {
            "mtime": mtime,
            "size": size,
            "chunks": [c.to_dict() for c in chunks]
        }
        
        return chunks

    def _load_manifest(self) -> dict[str, Any]:
        """Load the scan manifest from disk."""
        if not self._manifest_path.exists():
            return {}
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to load manifest %s: %s", self._manifest_path, exc)
            return {}

    def _save_manifest(self, current_files: set[str]) -> None:
        """Save the current manifest to disk, pruning missing files."""
        # Prune manifest: keep only files that were seen in the latest scan
        pruned_manifest = {
            k: v for k, v in self._manifest.items() if k in current_files
        }
        self._manifest = pruned_manifest
        
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2)
        except OSError as exc:
            logger.debug("Failed to save manifest %s: %s", self._manifest_path, exc)

    def iter_chunks(self) -> Iterator[DocumentChunk]:
        """Lazy iterator that yields :class:`DocumentChunk` objects one by one.

        Useful for large corpora where materialising the full chunk list
        would consume excessive memory.

        Yields:
            :class:`DocumentChunk` objects in file-discovery order.

        Raises:
            PersonalLLMException: If a file cannot be read or decoded.
        """
        txt_files: list[Path] = self._discover_files()

        if not txt_files:
            if not self.silent:
                logger.warning(
                    "DocumentLoader: no supported files found in %s (recursive=%s)",
                    self.data_dir,
                    self.recursive,
                )
            return

        for file_path in txt_files:
            yield from self._process_file(file_path)

    def load_raw_documents(self) -> list[RawDocument]:
        """Load all files without chunking and return :class:`RawDocument` objects.

        Useful for inspection, statistics, or when a different chunking
        strategy is applied externally.

        Returns:
            List of :class:`RawDocument` objects, one per discovered file.
        """
        docs: list[RawDocument] = []
        for file_path in self._discover_files():
            raw = self._load_file(file_path)
            if raw is not None:
                docs.append(raw)
        logger.debug("DocumentLoader.load_raw_documents(): %d docs loaded", len(docs))
        return docs

    def get_stats(self) -> dict[str, object]:
        """Return corpus statistics without fully loading all files into memory.

        Scans file metadata (size, count) without reading file contents.

        Returns:
            Dictionary with keys:
            ``"file_count"``, ``"total_size_bytes"``, ``"files"`` (list of
            ``{"filename": str, "size_bytes": int}`` dicts).
        """
        files: list[Path] = self._discover_files()
        file_stats: list[dict[str, object]] = [
            {"filename": f.name, "size_bytes": f.stat().st_size}
            for f in files
        ]
        return {
            "file_count": len(files),
            "total_size_bytes": sum(f["size_bytes"] for f in file_stats),  # type: ignore[arg-type]
            "files": file_stats,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_files(self) -> list[Path]:
        """Return a sorted list of supported text files in ``data_dir``.

        Returns:
            Alphabetically sorted list of ``Path`` objects for all files
            whose suffix is in :data:`SUPPORTED_EXTENSIONS`.
        """
        if self.recursive:
            found: list[Path] = [
                p for p in self.data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        else:
            found = [
                p for p in self.data_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        return sorted(found)

    def _load_file(self, file_path: Path) -> RawDocument | None:
        """Read a single file from disk, trying each configured encoding.

        Errors are logged but do not propagate — the caller receives ``None``
        and processing continues with the next file.

        Args:
            file_path: Absolute path to the file to read.

        Returns:
            A :class:`RawDocument` on success, or ``None`` on failure.
        """
        logger.debug("DocumentLoader: reading %s", file_path.name)

        size_bytes: int
        try:
            size_bytes = file_path.stat().st_size
        except OSError as exc:
            logger.error("Cannot stat '%s': %s", file_path, exc)
            return None

        if size_bytes == 0:
            logger.warning("Skipping empty file: %s", file_path.name)
            return None

        ext = file_path.suffix.lower()
        
        # Branch on file extension
        if ext == ".pdf":
            from core.knowledge_base.parsers import parse_pdf
            raw_text = parse_pdf(file_path)
            used_encoding = "binary/pdf"
        elif ext in (".docx", ".doc"):
            from core.knowledge_base.parsers import parse_docx
            raw_text = parse_docx(file_path)
            used_encoding = "binary/docx"
        elif ext == ".xlsx":
            from core.knowledge_base.parsers import parse_xlsx
            raw_text = parse_xlsx(file_path)
            used_encoding = "binary/xlsx"
        elif ext in (".png", ".jpg", ".jpeg"):
            from core.knowledge_base.parsers import parse_image
            raw_text = parse_image(file_path)
            used_encoding = "binary/image"
        elif ext in (".mp3", ".wav"):
            from core.knowledge_base.parsers import parse_audio
            raw_text = parse_audio(file_path)
            used_encoding = "binary/audio"
        else:
            # Fallback for plain text (.txt)
            for enc in self.encodings:
                try:
                    raw_text = file_path.read_text(encoding=enc)
                    used_encoding = enc
                    break
                except UnicodeDecodeError:
                    logger.debug(
                        "Encoding '%s' failed for '%s'; trying next.", enc, file_path.name
                    )
                except OSError as exc:
                    logger.error("I/O error reading '%s': %s", file_path, exc)
                    return None

        if raw_text is None:
            if not self.silent:
                logger.error(
                    "All extraction methods failed for '%s'. Skipping.",
                    file_path.name,
                )
            else:
                logger.debug(
                    "Extraction failed for '%s'.", file_path.name
                )
            return None

        normalised: str = _normalise_text(raw_text)

        if not normalised:
            logger.warning(
                "File '%s' is non-empty but contains no usable text after "
                "normalisation. Skipping.",
                file_path.name,
            )
            return None

        return RawDocument(
            source_path=file_path,
            filename=file_path.name,
            text=normalised,
            size_bytes=size_bytes,
            encoding=used_encoding,
        )

    def _chunk_document(self, doc: RawDocument) -> Iterator[DocumentChunk]:
        """Chunk a single :class:`RawDocument` and yield :class:`DocumentChunk` objects.

        Args:
            doc: The loaded and normalised document to chunk.

        Yields:
            :class:`DocumentChunk` objects with source metadata attached.
        """
        raw_segments: list[str] = self._chunker.chunk(doc.text)

        if not raw_segments:
            logger.warning(
                "Document '%s' produced zero chunks after splitting.", doc.filename
            )
            return

        total: int = len(raw_segments)

        # Track character positions in the original text for each segment.
        search_cursor: int = 0

        for idx, segment_text in enumerate(raw_segments):
            # Find this segment's approximate start position in the original
            # text (used for metadata; not exact due to overlap/stripping).
            stripped_head: str = segment_text.lstrip()[:30]
            char_start: int = doc.text.find(stripped_head, search_cursor)
            if char_start == -1:
                char_start = search_cursor  # Fallback
            char_end: int = char_start + len(segment_text)

            yield DocumentChunk(
                text=segment_text,
                chunk_index=idx,
                total_chunks=total,
                source_path=doc.source_path,
                filename=doc.filename,
                char_start=char_start,
                char_end=min(char_end, len(doc.text)),
                has_overlap=(idx > 0 and self._chunker.overlap > 0),
                metadata={
                    "file_size_bytes": doc.size_bytes,
                    "encoding": doc.encoding,
                    "source_path": str(doc.source_path),
                    "media_type": doc.source_path.suffix.lower().lstrip("."),
                    "date_added": int(time.time()),
                },
            )

            # Advance search cursor past the non-overlap portion of this segment
            advance: int = max(
                len(segment_text) - self._chunker.overlap, 1
            )
            search_cursor = max(char_start + advance, search_cursor + 1)

        logger.debug(
            "DocumentLoader: '%s' → %d chunks", doc.filename, total
        )


# ---------------------------------------------------------------------------
# Public re-export surface
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "SUPPORTED_EXTENSIONS",
    "RawDocument",
    "DocumentChunk",
    "TextChunker",
    "DocumentLoader",
]
