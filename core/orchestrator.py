# ==== core/orchestrator.py ====
"""
Central coordinator for the Personal LLM Orchestrator.

The :class:`SystemOrchestrator` is the *brain* of the application.  It wires
together every subsystem built in the previous steps and exposes a single,
clean :meth:`~SystemOrchestrator.process_query` method that the CLI calls for
every user turn.  All business logic lives here; the CLI handles only I/O.

Responsibilities
----------------
1. **Bootstrap** — Load settings, initialise the model registry, vector store,
   document loader, and query router in the correct order.
2. **Routing** — Delegate query classification to :class:`~core.router.QueryRouter`
   and select the appropriate execution path.
3. **Retrieval** — For ``RAG`` queries, search the vector store, collect the
   top-k chunks, and inject them as context into the prompt.
4. **Conversation memory** — Maintain a rolling window of the last
   ``memory_turns`` (default 5) user/assistant exchanges.  The window is
   serialised into the system prompt so the model is always aware of recent
   context.
5. **Generation** — Build a typed :class:`~models.base.GenerationRequest`,
   dispatch it to the correct :class:`~models.base.BaseLLMClient`, and return
   a structured :class:`OrchestratorResponse`.
6. **Lifecycle management** — Provide :meth:`reload_knowledge_base` for
   hot-reloading indexed documents without restarting the process.

Prompt templates
----------------
Three system-prompt templates are defined as module-level constants:

* :data:`_SYSTEM_PROMPT_CHAT` — General assistant persona.
* :data:`_SYSTEM_PROMPT_CODE` — Code-expert persona with formatting guidance.
* :data:`_SYSTEM_PROMPT_RAG`  — Evidence-grounded persona with injected context.

Each template supports ``{context}`` and ``{history}`` placeholder substitution.

Error handling
--------------
All domain exceptions propagate to the caller as :class:`OrchestratorResponse`
objects with ``success=False`` and a populated ``error`` field.  The CLI
never sees raw exceptions from this layer.

Threading
---------
The orchestrator is **not** thread-safe by default.  The conversation history
deque is mutated on every turn.  If concurrent calls are required, wrap in a
per-caller instance or add external locking.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from core.exceptions import (
    LLMInferenceError,
    OllamaConnectionError,
    PersonalLLMException,
    VectorStoreError,
)
from core.router import QueryRouter, RouteMode, RoutingDecision
from core.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from core.utils.logger import get_logger
from models.base import GenerationRequest, GenerationResponse

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of conversation turns kept in short-term memory.
_DEFAULT_MEMORY_TURNS: Final[int] = 5

#: Maximum number of RAG context chunks injected per query.
_DEFAULT_RAG_TOP_K: Final[int] = 3

#: Maximum character length of a single RAG context chunk included in prompt.
_MAX_CONTEXT_CHUNK_CHARS: Final[int] = 600

#: Separator inserted between RAG context chunks in the prompt.
_CONTEXT_SEPARATOR: Final[str] = "\n---\n"

# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CHAT: Final[str] = """\
You are a helpful, knowledgeable, and concise AI assistant.
Engage naturally with the user. If you are unsure of something, say so.
{history}""".strip()

_SYSTEM_PROMPT_CODE: Final[str] = """\
You are an expert software engineer and code reviewer with deep knowledge of \
Python, system design, algorithms, and best practices.
When writing code:
  • Use clear variable names and add concise inline comments.
  • Prefer idiomatic, readable solutions over clever one-liners.
  • Always include type hints for Python code.
  • If asked to fix a bug, first briefly diagnose the root cause, then provide \
the corrected code.
{history}""".strip()

_SYSTEM_PROMPT_RAG: Final[str] = """\
You are a precise and evidence-based AI assistant.
Answer questions using ONLY the context excerpts provided below.
If the context does not contain enough information to answer the question, \
say "I don't have enough information in the knowledge base to answer that."
Do not speculate beyond what the context supports.

CONTEXT:
{context}
{history}""".strip()

# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConversationTurn:
    """A single user/assistant exchange stored in conversation memory.

    Attributes:
        role: Either ``"user"`` or ``"assistant"``.
        content: The text of the turn.
        route_mode: The routing mode that was used for this turn.
        timestamp: Unix timestamp of when this turn was created.
    """

    role: str
    content: str
    route_mode: RouteMode
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class OrchestratorResponse:
    """Immutable result of a single :meth:`SystemOrchestrator.process_query` call.

    Attributes:
        text: The LLM-generated response text.  Empty string on failure.
        route_mode: The routing mode that was applied.
        model_name: The model that generated the response.
        decision: The full :class:`~core.router.RoutingDecision`.
        rag_context_used: Number of RAG context chunks injected.  ``0`` for
            non-RAG queries.
        duration_seconds: Wall-clock time from query receipt to response.
        prompt_tokens: Token count of the prompt (provider-reported).
        completion_tokens: Token count of the completion (provider-reported).
        success: ``True`` if generation completed without error.
        error: Error message string if ``success`` is ``False``.
    """

    text: str
    route_mode: RouteMode
    model_name: str
    decision: RoutingDecision
    rag_context_used: int = 0
    duration_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    success: bool = True
    error: str = ""

    def __str__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error}"
        return (
            f"OrchestratorResponse("
            f"mode={self.route_mode.name}, "
            f"model={self.model_name!r}, "
            f"rag_chunks={self.rag_context_used}, "
            f"dur={self.duration_seconds:.2f}s, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# SystemOrchestrator
# ---------------------------------------------------------------------------


class SystemOrchestrator:
    """Coordinates all subsystems to process user queries end-to-end.

    Args:
        model_configs_dir: Path to the YAML model configs directory.
            Defaults to ``"models/model_configs"``.
        data_dir: Path to the ``.txt`` source documents directory.
            Defaults to ``"assets/"``.
        chroma_persist_dir: Path for ChromaDB persistence.
            Defaults to ``"data/chroma_store"``.
        chroma_collection: ChromaDB collection name.
        embedding_model: Sentence-transformer model identifier.
        embedding_device: Compute device (``"cpu"`` / ``"cuda"`` / ``"mps"``).
        default_model_name: Registry key of the default LLM model to use.
            When ``None``, the first registered model is used.
        rag_top_k: Number of context chunks retrieved per RAG query.
        memory_turns: Number of conversation turns to keep in short-term memory.
        chunk_size: Document chunk size passed to ``DocumentLoader``.
        chunk_overlap: Document chunk overlap passed to ``DocumentLoader``.
        advanced_threshold: Router ADVANCED eligibility score threshold.
        auto_index: If ``True`` (default), index documents on first
            PERSONAL_MEMORY query if the collection is empty.

    Raises:
        PersonalLLMException: If any subsystem fails to initialise.

    Example:
        >>> orch = SystemOrchestrator()
        >>> resp = orch.process_query("Explain what embeddings are")
        >>> print(resp.text)
    """

    def __init__(
        self,
        model_configs_dir: str | Path = "models/model_configs",
        data_dir: str | Path = "assets/",
        chroma_persist_dir: str | Path = "data/chroma_store",
        chroma_collection: str = "llm_orchestrator_default",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        default_model_name: str | None = None,
        rag_top_k: int = _DEFAULT_RAG_TOP_K,
        memory_turns: int = _DEFAULT_MEMORY_TURNS,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        advanced_threshold: float = 1.0,
        auto_index: bool = True,
    ) -> None:
        self._model_configs_dir: Path = Path(model_configs_dir)
        self._data_dir: Path = Path(data_dir)
        self._chroma_persist_dir: Path = Path(chroma_persist_dir)
        self._chroma_collection: str = chroma_collection
        self._embedding_model: str = embedding_model
        self._embedding_device: str = embedding_device
        self._default_model_name: str | None = default_model_name
        self.rag_top_k: int = rag_top_k
        self.memory_turns: int = memory_turns
        self._chunk_size: int = chunk_size
        self._chunk_overlap: int = chunk_overlap
        self._auto_index: bool = auto_index

        # Short-term conversation memory (bounded deque)
        self._history: deque[ConversationTurn] = deque(maxlen=memory_turns * 2)

        # Subsystem references (populated in _initialise_*)
        self._registry: Any = None      # ModelRegistry
        self._vector_store: Any = None  # VectorStore
        self._router: QueryRouter = QueryRouter(
            advanced_threshold=advanced_threshold,
        )
        self._auto_indexed: bool = False  # Track if auto-index has been attempted

        # One-time warning flags — suppresses per-query log spam
        self._advanced_api_warning_shown: bool = False

        logger.info(
            "SystemOrchestrator: starting up "
            "(model_configs=%s, data_dir=%s, rag_top_k=%d, memory=%d)",
            self._model_configs_dir,
            self._data_dir,
            self.rag_top_k,
            self.memory_turns,
        )

        self._initialise_registry()
        self._initialise_vector_store()

        logger.info("SystemOrchestrator: ready (vector store deferred).")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def route_query(self, query: str) -> RoutingDecision:
        """Classify *query* without executing any LLM call.

        Exposed so the CLI can preview the routing decision (e.g. to show a
        "Thinking …" indicator) **without** triggering a second route call
        inside :meth:`process_query`.

        Args:
            query: Raw user input string.

        Returns:
            A :class:`RoutingDecision` from the underlying :class:`QueryRouter`.
        """
        return self._router.route(query)

    def process_query(
        self,
        query: str,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        decision: RoutingDecision | None = None,
    ) -> OrchestratorResponse:
        """Process a single user query through the full orchestration pipeline.

        Steps performed:

        1. Route the query (or accept a pre-computed :class:`RoutingDecision`
           from the caller — avoids the double-route penalty when the CLI
           already called :meth:`route_query`).
        2. Trigger deferred auto-index *only* for ``PERSONAL_MEMORY`` queries.
        3. If ``PERSONAL_MEMORY``: retrieve relevant context from the vector store.
        4. Build the appropriate system prompt (with history and context).
        5. Construct a :class:`~models.base.GenerationRequest`.
        6. Dispatch to the LLM client and capture the response.
        7. Update conversation history.
        8. Return a structured :class:`OrchestratorResponse`.

        Args:
            query: The user's input string.  Must be non-empty.
            model_name: Override the default model for this query.
            temperature: Override the default temperature for this query.
            max_tokens: Override the default max tokens for this query.
            decision: Optional pre-computed routing decision.  When provided
                the router is **not** called again, eliminating the double-
                route log entries visible in previous sessions.

        Returns:
            An :class:`OrchestratorResponse` — always returned, never raises.
            Check ``response.success`` and ``response.error`` for failure.
        """
        if not query or not query.strip():
            return self._error_response(
                query="",
                error="Query must not be empty.",
                decision=RoutingDecision(
                    mode=RouteMode.GENERAL_CHAT,
                    confidence=0.0,
                    scores={},
                    matched_signals=[],
                    query_preview="",
                ),
            )

        wall_start: float = time.perf_counter()

        # --- Step 1: Route (skip if caller already routed) ---
        if decision is None:
            try:
                decision = self._router.route(query)
            except Exception as exc:
                logger.error("Router failed: %s", exc)
                decision = RoutingDecision(
                    mode=RouteMode.PERSONAL_MEMORY,
                    confidence=0.0,
                    scores={},
                    matched_signals=["fallback:router_error"],
                    query_preview=query[:80],
                )

        logger.info(
            "process_query(): mode=%s conf=%.2f | query=%r",
            decision.mode.name,
            decision.confidence,
            query[:60],
        )

        # --- Step 2: Deferred Auto-Index (PERSONAL_MEMORY only) ---
        # GENERAL_CHAT queries never need the vector store — do not pay the
        # ChromaDB + embedding model cold-start cost for simple conversations.
        if (
            self._auto_index
            and not self._auto_indexed
            and decision.mode == RouteMode.PERSONAL_MEMORY
        ):
            self._auto_index_if_empty()

        # --- Step 3: Retrieve context (PERSONAL_MEMORY only) ---
        context_chunks: list[dict[str, Any]] = []
        if decision.mode == RouteMode.PERSONAL_MEMORY and self._vector_store is not None:
            context_chunks = self._retrieve_context(query)

        # --- Step 4: Build system prompt ---
        system_prompt: str = self._build_system_prompt(
            mode=decision.mode,
            context_chunks=context_chunks,
        )

        # --- Step 5: Build prompt string ---
        prompt: str = self._build_prompt(query)

        # --- Step 6: Select model ---
        # ADVANCED queries → auto-advanced priority chain via ModelRegistry.
        # All other modes → configured default (local Ollama).
        if decision.mode == RouteMode.ADVANCED_KNOWLEDGE:
            target_model = "auto-advanced"
        else:
            target_model = model_name or self._default_model_name or self._first_model()

        gen_response: GenerationResponse | None = None
        error_message: str = ""

        try:
            client = self._registry.get_model(target_model)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            request_kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
            }
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens

            gen_request = GenerationRequest(**request_kwargs)
            gen_response = client.generate(gen_request)

            # --- Agentic Tool Execution Loop ---
            # Maximum of 5 consecutive tool execution turns to prevent infinite loops
            turn_count = 0
            while (
                gen_response.finish_reason == "tool_calls"
                and gen_response.tool_calls
                and turn_count < 5
            ):
                turn_count += 1

                # 1. Append the model's tool calls to the history context
                messages.append({
                    "role": "assistant",
                    "content": gen_response.text or "",
                    "tool_calls": gen_response.tool_calls,
                })

                # 2. Execute each requested tool locally
                for tcall in gen_response.tool_calls:
                    func_name = tcall.get("function", {}).get("name")
                    args = tcall.get("function", {}).get("arguments", {})

                    logger.info("Executing Tool: %s(args=%r)", func_name, args)
                    if func_name in TOOL_REGISTRY:
                        try:
                            result = TOOL_REGISTRY[func_name](**args)
                        except Exception as e:
                            logger.error("Tool execution failed: %s", e)
                            result = f"Error executing tool {func_name}: {e}"
                    else:
                        result = f"Error: Tool {func_name} not found in the registry."

                    # 3. Append the execution result to the history context
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                    })

                # 4. Fire the context back to the LLM so it can answer the user
                request_kwargs["messages"] = messages
                gen_request = GenerationRequest(**request_kwargs)
                gen_response = client.generate(gen_request)

        except OllamaConnectionError as exc:
            error_message = (
                f"Cannot reach Ollama ({exc.url or 'unknown URL'}). "
                f"Is `ollama serve` running? [{exc.error_code}]"
            )
            logger.error("OllamaConnectionError: %s", exc)
        except LLMInferenceError as exc:
            error_message = f"LLM inference failed: {exc.message} [{exc.error_code}]"
            logger.error("LLMInferenceError: %s", exc)
        except PersonalLLMException as exc:
            error_message = f"Orchestrator error: {exc.message}"
            logger.error("PersonalLLMException: %s", exc)
        except Exception as exc:
            error_message = f"Unexpected error during generation: {exc}"
            logger.exception("Unexpected error in process_query")

        wall_elapsed: float = time.perf_counter() - wall_start

        # --- Step 7: Update history ---
        self._history.append(
            ConversationTurn(role="user", content=query, route_mode=decision.mode)
        )
        if gen_response is not None:
            self._history.append(
                ConversationTurn(
                    role="assistant",
                    content=gen_response.text,
                    route_mode=decision.mode,
                )
            )

        # --- Step 8: Build response ---
        if gen_response is not None:
            return OrchestratorResponse(
                text=gen_response.text,
                route_mode=decision.mode,
                model_name=gen_response.model_name,
                decision=decision,
                rag_context_used=len(context_chunks),
                duration_seconds=wall_elapsed,
                prompt_tokens=gen_response.prompt_tokens,
                completion_tokens=gen_response.completion_tokens,
                success=True,
                error="",
            )
        else:
            return self._error_response(
                query=query,
                error=error_message,
                decision=decision,
                model_name=target_model,
                duration_seconds=wall_elapsed,
            )

    def reload_knowledge_base(self, confirm_reindex: bool = True) -> dict[str, Any]:
        """Re-scan the document directory and re-index all chunks.

        This is the backend for the CLI ``/reload`` command.  Existing
        documents with the same SHA-256 hash are skipped; only genuinely
        new or modified chunks are added.

        Args:
            confirm_reindex: If ``True`` (default), proceeds with indexing.
                Pass ``False`` to do a dry-run (scan only, no writes).

        Returns:
            A summary dict with keys: ``"files_found"``, ``"chunks_found"``,
            ``"added"``, ``"skipped"``, ``"failed"``, ``"duration_seconds"``.

        Raises:
            PersonalLLMException: If document loading or indexing fails.
        """
        logger.info(
            "SystemOrchestrator.reload_knowledge_base(): "
            "scanning %s (reindex=%s)",
            self._data_dir,
            confirm_reindex,
        )
        start: float = time.perf_counter()

        from core.knowledge_base.document_loader import DocumentLoader  # noqa: PLC0415

        loader = DocumentLoader(
            data_dir=self._data_dir,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
            recursive=True,
            silent=True,
        )

        stats: dict[str, Any] = loader.get_stats()
        chunks = loader.load_and_chunk()
        chunk_dicts: list[dict[str, Any]] = [c.to_dict() for c in chunks]

        result: dict[str, Any] = {
            "files_found": stats["file_count"],
            "chunks_found": len(chunks),
            "added": 0,
            "skipped": 0,
            "failed": 0,
            "duration_seconds": 0.0,
        }

        if confirm_reindex and chunk_dicts:
            try:
                index_result = self._vector_store.index_documents(chunk_dicts)
                result["added"] = index_result.added
                result["skipped"] = index_result.skipped
                result["failed"] = index_result.failed
            except VectorStoreError as exc:
                logger.error("reload_knowledge_base() indexing failed: %s", exc)
                raise

        result["duration_seconds"] = round(time.perf_counter() - start, 2)
        logger.info("reload_knowledge_base() complete: %s", result)
        return result

    def list_models(self) -> list[dict[str, Any]]:
        """Return summary metadata for all registered models."""
        return self._registry.list_model_details()

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics about the vector store collection.

        Returns:
            Dict with ``"collection_name"`` and ``"document_count"`` keys.
            ``document_count`` is -1 if the vector store is not yet loaded.
        """
        count = -1
        if self._vector_store.is_loaded:
            count = self._vector_store.collection_count()

        return {
            "collection_name": self._chroma_collection,
            "document_count": count,
        }

    def clear_history(self) -> None:
        """Wipe the in-memory conversation history."""
        self._history.clear()
        logger.info("SystemOrchestrator: conversation history cleared.")

    def get_history(self) -> list[dict[str, Any]]:
        """Return the current conversation history as a list of dicts."""
        return [
            {
                "role": t.role,
                "content": t.content[:200],
                "mode": t.route_mode.name,
                "timestamp": t.timestamp,
            }
            for t in self._history
        ]

    def health_check(self) -> dict[str, bool]:
        """Probe all subsystems and return a health status map.

        Returns:
            Dict mapping subsystem names to boolean health status.
        """
        status: dict[str, bool] = {}

        status["registry"] = len(self._registry) > 0

        try:
            if self._vector_store.is_loaded:
                self._vector_store.collection_count()
                status["vector_store"] = True
            else:
                status["vector_store"] = self._vector_store.persist_dir.exists()
        except Exception:
            status["vector_store"] = False

        try:
            model_name = self._default_model_name or self._first_model()
            if model_name:
                client = self._registry.get_model(model_name)
                status["default_model"] = client.health_check()
            else:
                status["default_model"] = False
        except Exception:
            status["default_model"] = False

        return status

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _initialise_registry(self) -> None:
        """Load the ModelRegistry from YAML config files."""
        from models.registry import ModelRegistry  # noqa: PLC0415

        logger.debug(
            "SystemOrchestrator: loading ModelRegistry from %s",
            self._model_configs_dir,
        )
        try:
            self._registry = ModelRegistry(
                configs_dir=self._model_configs_dir,
                auto_load=True,
            )
        except PersonalLLMException:
            raise
        except Exception as exc:
            from core.exceptions import ConfigurationError  # noqa: PLC0415
            raise ConfigurationError(
                message=f"Failed to initialise ModelRegistry: {exc}",
                error_code="CFG_010",
            ) from exc

        if self._default_model_name is None:
            models = self._registry.list_models()
            if models:
                self._default_model_name = models[0]
                logger.info(
                    "SystemOrchestrator: default model set to '%s'",
                    self._default_model_name,
                )

        logger.info(
            "SystemOrchestrator: ModelRegistry loaded — %d model(s): %s",
            len(self._registry),
            self._registry.list_models(),
        )

    def _initialise_vector_store(self) -> None:
        """Initialise the ChromaDB vector store (lazy — no I/O until first use)."""
        from core.knowledge_base.vector_store import VectorStore  # noqa: PLC0415

        logger.debug(
            "SystemOrchestrator: initialising VectorStore "
            "(model=%s, device=%s, persist=%s)",
            self._embedding_model,
            self._embedding_device,
            self._chroma_persist_dir,
        )
        self._vector_store = VectorStore(
            persist_dir=self._chroma_persist_dir,
            collection_name=self._chroma_collection,
            embedding_model_name=self._embedding_model,
            device=self._embedding_device,
            top_k=self.rag_top_k,
        )
        logger.info("SystemOrchestrator: VectorStore initialized (lazy).")

    def _auto_index_if_empty(self) -> None:
        """Index documents on first PERSONAL_MEMORY query if the collection is empty.

        This is intentionally deferred — GENERAL_CHAT queries never pay
        the ChromaDB + embedding model cold-start cost.
        """
        if self._auto_indexed:
            return
        self._auto_indexed = True

        try:
            count: int = self._vector_store.collection_count()

            if count == 0:
                if self._data_dir.exists():
                    logger.info(
                        "SystemOrchestrator: collection empty — auto-indexing from %s",
                        self._data_dir,
                    )
                    try:
                        result = self.reload_knowledge_base(confirm_reindex=True)
                        logger.info(
                            "SystemOrchestrator: auto-index complete: %s", result
                        )
                    except Exception as exc:
                        logger.error("SystemOrchestrator: auto-index failed: %s", exc)
                else:
                    logger.info(
                        "SystemOrchestrator: data_dir '%s' does not exist — "
                        "skipping auto-index.",
                        self._data_dir,
                    )
            else:
                logger.info(
                    "SystemOrchestrator: collection has %d docs — skipping auto-index.",
                    count,
                )

        except Exception as exc:
            logger.error("Failed to check collection count for auto-index: %s", exc)
            from core.exceptions import PipelineError
            raise PipelineError(
                message=f"Auto-index failed on startup: {exc}",
                stage="bootstrap",
            ) from exc

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> list[dict[str, Any]]:
        """Fetch the top-k semantically similar chunks for *query*.

        Args:
            query: The user's query string.

        Returns:
            List of result dicts.  Returns an empty list on any retrieval error.
        """
        try:
            results = self._vector_store.similarity_search(
                query=query,
                k=self.rag_top_k,
                min_score=0.1,
            )
            logger.debug(
                "_retrieve_context(): %d chunks retrieved (top score=%.3f)",
                len(results),
                results[0].score if results else 0.0,
            )
            return [r.to_dict() for r in results]
        except VectorStoreError as exc:
            logger.error("_retrieve_context() failed with vector store error: %s", exc)
            from core.exceptions import ContextRetrievalError
            raise ContextRetrievalError(
                message=f"Retrieval failed due to vector store error: {exc}",
                query=query,
            ) from exc
        except Exception as exc:
            logger.error("_retrieve_context() unexpected error: %s", exc)
            from core.exceptions import ContextRetrievalError
            raise ContextRetrievalError(
                message=f"Retrieval failed due to unexpected error: {exc}",
                query=query,
            ) from exc

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        mode: RouteMode,
        context_chunks: list[dict[str, Any]],
    ) -> str:
        """Assemble the system prompt for the given routing mode."""
        history_block: str = self._format_history_block()

        if mode == RouteMode.PERSONAL_MEMORY:
            context_text: str = self._format_context_block(context_chunks)
            return _SYSTEM_PROMPT_RAG.format(
                context=context_text or "(no relevant context found)",
                history=history_block,
            )
        elif mode == RouteMode.ADVANCED_KNOWLEDGE:
            return _SYSTEM_PROMPT_CODE.format(history=history_block)
        else:
            return _SYSTEM_PROMPT_CHAT.format(history=history_block)

    def _build_prompt(self, query: str) -> str:
        """Build the user-facing prompt from the raw query."""
        return query.strip()

    def _format_context_block(self, chunks: list[dict[str, Any]]) -> str:
        """Serialise RAG context chunks into an annotated text block."""
        if not chunks:
            return ""

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            filename: str = str(chunk.get("metadata", {}).get("filename", "unknown"))
            score: float = float(chunk.get("score", 0.0))
            text: str = str(chunk.get("text", ""))[:_MAX_CONTEXT_CHUNK_CHARS]
            parts.append(
                f"[Excerpt {i} | source: {filename} | relevance: {score:.2f}]\n{text}"
            )

        return _CONTEXT_SEPARATOR.join(parts)

    def _format_history_block(self) -> str:
        """Serialise recent conversation history for inclusion in system prompt."""
        if not self._history:
            return ""

        lines: list[str] = ["\n\nConversation so far:"]
        for turn in self._history:
            prefix: str = "[user]" if turn.role == "user" else "[assistant]"
            content: str = turn.content[:300]
            if len(turn.content) > 300:
                content += " [...]"
            lines.append(f"{prefix}: {content}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _first_model(self) -> str:
        """Return the name of the first registered model or empty string."""
        models = self._registry.list_models()
        return models[0] if models else ""

    @staticmethod
    def _error_response(
        query: str,
        error: str,
        decision: RoutingDecision,
        model_name: str = "unknown",
        duration_seconds: float = 0.0,
    ) -> OrchestratorResponse:
        """Construct a failed :class:`OrchestratorResponse`."""
        logger.warning(
            "SystemOrchestrator: error response for query=%r error=%s",
            query[:60],
            error,
        )
        return OrchestratorResponse(
            text="",
            route_mode=decision.mode,
            model_name=model_name,
            decision=decision,
            rag_context_used=0,
            duration_seconds=duration_seconds,
            success=False,
            error=error,
        )


# ---------------------------------------------------------------------------
# Public re-export surface
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "ConversationTurn",
    "OrchestratorResponse",
    "SystemOrchestrator",
]
