# ==== core/orchestrator.py ====
"""
Central coordinator for the Personal LLM Orchestrator.

The :class:`SystemOrchestrator` is the *brain* of the application.  It wires
together every subsystem built in the previous steps and exposes a single,
clean :meth:`~SystemOrchestrator.process_query` method that the CLI calls for
every user turn.  All business logic lives here; the CLI handles only I/O.

Responsibilities
----------------
1. Bootstrap — Load settings, model registry, vector store, document loader,
   and query router in the correct order.
2. Routing — Delegate classification to QueryRouter; accept pre-computed
   decisions to avoid the double-route log entries seen in earlier sessions.
3. Streaming — Fast path via stream_query() for GENERAL_CHAT and
   ADVANCED_KNOWLEDGE; tokens arrive in ~1-2 s vs ~40 s blocking.
4. Retrieval — For PERSONAL_MEMORY queries search the vector store, collect
   top-k chunks, and inject them as context.
5. Conversation memory — Rolling window of last memory_turns exchanges.
6. Generation — Build GenerationRequest, dispatch to LLM, return
   OrchestratorResponse.
7. Lifecycle — reload_knowledge_base() for hot-reloading without restart.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator

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

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MEMORY_TURNS: Final[int] = 5
_DEFAULT_RAG_TOP_K: Final[int] = 3
_MAX_CONTEXT_CHUNK_CHARS: Final[int] = 600
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
    """A single user/assistant exchange stored in conversation memory."""

    role: str
    content: str
    route_mode: RouteMode
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class OrchestratorResponse:
    """Immutable result of a single process_query() call."""

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
    """Coordinates all subsystems to process user queries end-to-end."""

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

        self._history: deque[ConversationTurn] = deque(maxlen=memory_turns * 2)
        self._registry: Any = None
        self._vector_store: Any = None
        self._router: QueryRouter = QueryRouter(advanced_threshold=advanced_threshold)
        self._auto_indexed: bool = False

        logger.info(
            "SystemOrchestrator: starting up "
            "(model_configs=%s, data_dir=%s, rag_top_k=%d, memory=%d)",
            self._model_configs_dir, self._data_dir, self.rag_top_k, self.memory_turns,
        )

        self._initialise_registry()
        self._initialise_vector_store()
        logger.info("SystemOrchestrator: ready (vector store deferred).")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def route_query(self, query: str) -> RoutingDecision:
        """Classify *query* without executing any LLM call.

        Exposed so the CLI can inspect the routing decision once and pass it
        directly to stream_query() or process_query() — eliminating the
        double-route log entries visible in earlier sessions.
        """
        return self._router.route(query)

    def stream_query(
        self,
        query: str,
        model_name: str | None = None,
        decision: RoutingDecision | None = None,
    ) -> Iterator[str]:
        """Stream response tokens for GENERAL_CHAT and ADVANCED_KNOWLEDGE queries.

        This is the **fast path**.  Tokens are yielded as they arrive from
        Ollama so the first word appears in ~1–2 s instead of ~40 s.

        For PERSONAL_MEMORY queries — which need RAG retrieval and may trigger
        tool-call loops — use process_query() instead.

        Args:
            query: The user's input string.
            model_name: Optional model override.
            decision: Pre-computed RoutingDecision from route_query().
                When provided the router is not called again.

        Yields:
            str: Text chunks as they arrive from the LLM.  On error an
                ``[Error: ...]`` string is yielded so the terminal always
                receives something to display.
        """
        if not query or not query.strip():
            yield "[Error: query must not be empty]"
            return

        if decision is None:
            try:
                decision = self._router.route(query)
            except Exception as exc:
                logger.error("stream_query(): router failed: %s", exc)
                decision = RoutingDecision(
                    mode=RouteMode.GENERAL_CHAT,
                    confidence=0.0,
                    scores={},
                    matched_signals=["fallback:router_error"],
                    query_preview=query[:80],
                )

        logger.info(
            "stream_query(): mode=%s conf=%.2f | query=%r",
            decision.mode.name, decision.confidence, query[:60],
        )

        system_prompt: str = self._build_system_prompt(
            mode=decision.mode, context_chunks=[]
        )
        prompt: str = self._build_prompt(query)

        target_model = (
            "auto-advanced"
            if decision.mode == RouteMode.ADVANCED_KNOWLEDGE
            else (model_name or self._default_model_name or self._first_model())
        )

        try:
            client = self._registry.get_model(target_model)
        except Exception as exc:
            logger.error("stream_query(): failed to get model %r: %s", target_model, exc)
            yield f"[Error: could not load model '{target_model}' — {exc}]"
            return

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request = GenerationRequest(messages=messages)
        accumulated: list[str] = []

        try:
            if hasattr(client, "stream_generate"):
                for chunk in client.stream_generate(request):
                    accumulated.append(chunk)
                    yield chunk
            else:
                # Fallback for API clients without streaming support.
                logger.debug(
                    "stream_query(): %s has no stream_generate — "
                    "falling back to blocking generate()",
                    type(client).__name__,
                )
                resp: GenerationResponse = client.generate(request)
                accumulated.append(resp.text)
                yield resp.text

        except OllamaConnectionError as exc:
            logger.error("stream_query(): OllamaConnectionError: %s", exc)
            yield (
                f"\n[Error: Cannot reach Ollama. "
                f"Is `ollama serve` running? ({exc.error_code})]"
            )
        except LLMInferenceError as exc:
            logger.error("stream_query(): LLMInferenceError: %s", exc)
            yield f"\n[Error: LLM inference failed — {exc.message}]"
        except Exception as exc:
            logger.exception("stream_query(): unexpected error")
            yield f"\n[Error: Unexpected error — {exc}]"

        # Update conversation history with the full accumulated response.
        full_text = "".join(accumulated)
        self._history.append(
            ConversationTurn(role="user", content=query, route_mode=decision.mode)
        )
        if full_text:
            self._history.append(
                ConversationTurn(
                    role="assistant", content=full_text, route_mode=decision.mode
                )
            )

    def process_query(
        self,
        query: str,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        decision: RoutingDecision | None = None,
    ) -> OrchestratorResponse:
        """Blocking pipeline for PERSONAL_MEMORY (RAG + tool-call loop).

        For GENERAL_CHAT and ADVANCED_KNOWLEDGE use stream_query() — it
        delivers the first token ~20-40× faster.

        Args:
            query: The user's input string.
            model_name: Override the default model for this query.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            decision: Pre-computed RoutingDecision — skips the internal
                router call when the CLI already called route_query().

        Returns:
            OrchestratorResponse — always returned, never raises.
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

        # Step 1: Route
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
            decision.mode.name, decision.confidence, query[:60],
        )

        # Step 2: Deferred auto-index — only for PERSONAL_MEMORY queries.
        # GENERAL_CHAT never needs the vector store; don't pay cold-start cost.
        if (
            self._auto_index
            and not self._auto_indexed
            and decision.mode == RouteMode.PERSONAL_MEMORY
        ):
            self._auto_index_if_empty()

        # Step 3: Retrieve context
        context_chunks: list[dict[str, Any]] = []
        if decision.mode == RouteMode.PERSONAL_MEMORY and self._vector_store is not None:
            context_chunks = self._retrieve_context(query)

        # Step 4 & 5: Build prompts
        system_prompt: str = self._build_system_prompt(
            mode=decision.mode, context_chunks=context_chunks
        )
        prompt: str = self._build_prompt(query)

        # Step 6: Select model and dispatch
        target_model = (
            "auto-advanced"
            if decision.mode == RouteMode.ADVANCED_KNOWLEDGE
            else (model_name or self._default_model_name or self._first_model())
        )

        gen_response: GenerationResponse | None = None
        error_message: str = ""

        try:
            client = self._registry.get_model(target_model)

            messages: list[dict[str, Any]] = []
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

            # Agentic tool-execution loop (max 5 turns)
            turn_count = 0
            while (
                gen_response.finish_reason == "tool_calls"
                and gen_response.tool_calls
                and turn_count < 5
            ):
                turn_count += 1
                messages.append({
                    "role": "assistant",
                    "content": gen_response.text or "",
                    "tool_calls": gen_response.tool_calls,
                })
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
                        result = f"Error: Tool {func_name} not found."
                    messages.append({"role": "tool", "content": str(result)})
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

        # Step 7: Update history
        self._history.append(
            ConversationTurn(role="user", content=query, route_mode=decision.mode)
        )
        if gen_response is not None:
            self._history.append(
                ConversationTurn(
                    role="assistant", content=gen_response.text, route_mode=decision.mode
                )
            )

        # Step 8: Return
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
        return self._error_response(
            query=query,
            error=error_message,
            decision=decision,
            model_name=target_model,
            duration_seconds=wall_elapsed,
        )

    def reload_knowledge_base(self, confirm_reindex: bool = True) -> dict[str, Any]:
        """Re-scan the document directory and re-index all chunks."""
        logger.info(
            "SystemOrchestrator.reload_knowledge_base(): scanning %s (reindex=%s)",
            self._data_dir, confirm_reindex,
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
        return self._registry.list_model_details()

    def get_collection_stats(self) -> dict[str, Any]:
        count = -1
        if self._vector_store.is_loaded:
            count = self._vector_store.collection_count()
        return {"collection_name": self._chroma_collection, "document_count": count}

    def clear_history(self) -> None:
        self._history.clear()
        logger.info("SystemOrchestrator: conversation history cleared.")

    def get_history(self) -> list[dict[str, Any]]:
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
        from models.registry import ModelRegistry  # noqa: PLC0415

        logger.debug("SystemOrchestrator: loading ModelRegistry from %s", self._model_configs_dir)
        try:
            self._registry = ModelRegistry(configs_dir=self._model_configs_dir, auto_load=True)
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
                    "SystemOrchestrator: default model set to '%s'", self._default_model_name
                )
        logger.info(
            "SystemOrchestrator: ModelRegistry loaded — %d model(s): %s",
            len(self._registry), self._registry.list_models(),
        )

    def _initialise_vector_store(self) -> None:
        from core.knowledge_base.vector_store import VectorStore  # noqa: PLC0415

        logger.debug(
            "SystemOrchestrator: initialising VectorStore (model=%s, device=%s, persist=%s)",
            self._embedding_model, self._embedding_device, self._chroma_persist_dir,
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
                        logger.info("SystemOrchestrator: auto-index complete: %s", result)
                    except Exception as exc:
                        logger.error("SystemOrchestrator: auto-index failed: %s", exc)
                else:
                    logger.info(
                        "SystemOrchestrator: data_dir '%s' does not exist — skipping.",
                        self._data_dir,
                    )
            else:
                logger.info(
                    "SystemOrchestrator: collection has %d docs — skipping auto-index.", count
                )
        except Exception as exc:
            logger.error("Failed to check collection count for auto-index: %s", exc)
            from core.exceptions import PipelineError
            raise PipelineError(
                message=f"Auto-index failed on startup: {exc}", stage="bootstrap"
            ) from exc

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> list[dict[str, Any]]:
        try:
            results = self._vector_store.similarity_search(
                query=query, k=self.rag_top_k, min_score=0.1
            )
            logger.debug(
                "_retrieve_context(): %d chunks retrieved (top score=%.3f)",
                len(results), results[0].score if results else 0.0,
            )
            return [r.to_dict() for r in results]
        except VectorStoreError as exc:
            logger.error("_retrieve_context() vector store error: %s", exc)
            from core.exceptions import ContextRetrievalError
            raise ContextRetrievalError(
                message=f"Retrieval failed: {exc}", query=query
            ) from exc
        except Exception as exc:
            logger.error("_retrieve_context() unexpected error: %s", exc)
            from core.exceptions import ContextRetrievalError
            raise ContextRetrievalError(
                message=f"Retrieval failed: {exc}", query=query
            ) from exc

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self, mode: RouteMode, context_chunks: list[dict[str, Any]]
    ) -> str:
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
        return query.strip()

    def _format_context_block(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return ""
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            filename = str(chunk.get("metadata", {}).get("filename", "unknown"))
            score = float(chunk.get("score", 0.0))
            text = str(chunk.get("text", ""))[:_MAX_CONTEXT_CHUNK_CHARS]
            parts.append(
                f"[Excerpt {i} | source: {filename} | relevance: {score:.2f}]\n{text}"
            )
        return _CONTEXT_SEPARATOR.join(parts)

    def _format_history_block(self) -> str:
        if not self._history:
            return ""
        lines: list[str] = ["\n\nConversation so far:"]
        for turn in self._history:
            prefix = "[user]" if turn.role == "user" else "[assistant]"
            content = turn.content[:300]
            if len(turn.content) > 300:
                content += " [...]"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _first_model(self) -> str:
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
        logger.warning(
            "SystemOrchestrator: error response for query=%r error=%s", query[:60], error
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


__all__: list[str] = [
    "ConversationTurn",
    "OrchestratorResponse",
    "SystemOrchestrator",
]