# ==== models/base.py ====
"""
Abstract base contract for every LLM client in the Personal LLM Orchestrator.

Design rationale
----------------
Following the **Dependency Inversion Principle** (SOLID "D"), all higher-level
modules (RAG pipeline, chat engine, agents) depend exclusively on the
abstractions defined here — never on a concrete provider implementation.
Swapping Ollama for an OpenAI-compatible endpoint, a llama.cpp server, or any
future provider requires only a new leaf class; zero changes to callers.

Key abstractions
----------------
* :class:`GenerationRequest` — Typed, validated Pydantic model representing
  everything a caller can say about a single completion request.
* :class:`GenerationResponse` — Typed, validated Pydantic model wrapping the
  provider's response with normalised metadata.
* :class:`BaseLLMClient` — Abstract Base Class (ABC) that every concrete client
  must implement.  Defines ``generate()``, ``stream_generate()``,
  ``health_check()``, and several concrete helpers.

Threading / async note
----------------------
All methods are synchronous in Step 2 (``requests``-based).  The interface is
designed so that async adapters can be layered on top later without breaking
existing callers (Liskov Substitution Principle).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Final, Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Package-wide sentinel value for "no maximum token limit set in call site".
UNSET_MAX_TOKENS: Final[int] = -1


# ---------------------------------------------------------------------------
# Transfer objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Immutable value object describing a single LLM completion request.

    ``GenerationRequest`` is the *only* object passed across the boundary
    between the orchestration layer and an LLM client.  Keeping it a plain
    frozen dataclass (rather than a dict) enforces schema discipline without
    pulling in Pydantic at the transport layer.

    Attributes:
        prompt: The full prompt string to send to the model.  Must be
            non-empty.
        model_name: Target model identifier as registered in the provider
            (e.g. ``"llama3"``, ``"gemma:7b"``).  When ``None`` the client
            uses its configured default.
        temperature: Sampling temperature in the range ``[0.0, 2.0]``.
            Lower → more deterministic; higher → more creative.
            Defaults to ``0.7``.
        max_tokens: Hard cap on generated tokens.  ``UNSET_MAX_TOKENS``
            (``-1``) signals the client to use its own default.
        top_p: Nucleus sampling probability mass.  ``1.0`` disables it.
        top_k: Top-k sampling pool size.  ``0`` disables it.
        repeat_penalty: Penalty factor applied to already-generated tokens
            to discourage repetition.  ``1.0`` means no penalty.
        stop_sequences: List of strings that, when generated, immediately
            terminate inference.  Empty list means no custom stop strings.
        system_prompt: Optional system / instruction prompt prepended to
            the conversation.  Not all providers honour this field.
        extra_params: Escape hatch for provider-specific parameters not
            covered by the standard fields.  Values are passed through
            verbatim to the provider's API payload.

    Raises:
        ValueError: At construction time if ``prompt`` is empty or
            ``temperature`` is outside ``[0.0, 2.0]``.

    Example:
        >>> req = GenerationRequest(
        ...     prompt="What is the capital of France?",
        ...     model_name="llama3",
        ...     temperature=0.1,
        ...     max_tokens=256,
        ... )
    """

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    model_name: str | None = None
    temperature: float = 0.7
    max_tokens: int = UNSET_MAX_TOKENS
    top_p: float = 1.0
    top_k: int = 0
    repeat_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field invariants after dataclass construction.

        Raises:
            ValueError: If neither `prompt` nor `messages` is provided, or if
                ``temperature`` is outside the valid range ``[0.0, 2.0]``.
        """
        if not self.prompt and not self.messages:
            raise ValueError("GenerationRequest must have either prompt or messages.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"GenerationRequest.temperature must be in [0.0, 2.0], "
                f"got {self.temperature}."
            )
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError(
                f"GenerationRequest.top_p must be in [0.0, 1.0], got {self.top_p}."
            )


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """Immutable value object wrapping a completed LLM generation result.

    Normalises provider-specific response shapes into a single schema that
    all downstream consumers can depend on.

    Attributes:
        text: The generated completion text.  Stripped of leading/trailing
            whitespace by the producing client before storage here.
        model_name: The exact model identifier reported by the provider
            (may differ from the requested name, e.g. when a tag alias is
            resolved).
        prompt_tokens: Number of tokens in the input prompt as counted by
            the provider.  ``None`` if the provider did not report it.
        completion_tokens: Number of tokens in the generated output.
            ``None`` if the provider did not report it.
        total_tokens: Sum of prompt and completion tokens.  ``None`` if
            either component is absent.
        duration_seconds: Wall-clock time (in seconds) from the moment the
            HTTP request was dispatched to the moment the response was fully
            received.
        raw_response: The provider's raw JSON response dict, preserved for
            debugging and provider-specific metadata extraction.
        finish_reason: Reason the generation stopped (e.g. ``"stop"``,
            ``"length"``, ``"error"``).  Provider-specific string.

    Example:
        >>> resp = GenerationResponse(
        ...     text="Paris is the capital of France.",
        ...     model_name="llama3",
        ...     duration_seconds=0.82,
        ...     finish_reason="stop",
        ... )
    """

    text: str
    model_name: str
    duration_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tool_calls: list[dict[str, Any]] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"

    @property
    def tokens_per_second(self) -> float | None:
        """Compute throughput as completion tokens per second.

        Returns:
            Tokens-per-second float, or ``None`` if token count or duration
            are unavailable / zero.
        """
        if self.completion_tokens and self.duration_seconds > 0:
            return self.completion_tokens / self.duration_seconds
        return None


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """Abstract contract that every LLM provider client must satisfy.

    All higher-level orchestrator components depend on this interface, never
    on concrete implementations.  This enforces the Dependency Inversion
    Principle and makes the system trivially testable via mock clients.

    Subclass responsibilities
    -------------------------
    Concrete subclasses **must** implement:

    * :meth:`generate`          — blocking single-shot completion.
    * :meth:`stream_generate`   — token-streaming completion as an iterator.
    * :meth:`health_check`      — probe the provider's reachability.

    Subclasses **may** override (but are not required to):

    * :meth:`count_tokens`      — provider-specific token counting.
    * :meth:`get_model_info`    — provider-specific model metadata.

    Args:
        model_name: Canonical model identifier used by default for all
            requests that do not supply their own ``model_name``.
        base_url: Base HTTP URL of the provider server.
        timeout: Default request timeout in seconds.  Individual calls may
            override this per-request.

    Attributes:
        model_name: The default model identifier for this client instance.
        base_url: The provider's base HTTP URL (normalised, no trailing slash).
        timeout: Default request timeout in seconds.

    Example:
        Implement a minimal mock client for unit testing::

            class MockLLMClient(BaseLLMClient):
                def generate(self, request):
                    return GenerationResponse(
                        text="mocked", model_name=self.model_name,
                        duration_seconds=0.0,
                    )
                def stream_generate(self, request):
                    yield "mocked"
                def health_check(self):
                    return True
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout: int = 120,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError(
                f"{self.__class__.__name__}: model_name must be a non-empty string."
            )
        if not base_url or not base_url.strip():
            raise ValueError(
                f"{self.__class__.__name__}: base_url must be a non-empty string."
            )

        self.model_name: str = model_name.strip()
        self.base_url: str = base_url.rstrip("/")
        self.timeout: int = timeout

    # ------------------------------------------------------------------
    # Abstract interface — MUST be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Execute a blocking, non-streaming LLM completion.

        The entire response is buffered before returning.  Use
        :meth:`stream_generate` for real-time token delivery.

        Args:
            request: A fully populated :class:`GenerationRequest` describing
                the prompt and inference parameters.

        Returns:
            A :class:`GenerationResponse` containing the generated text and
            normalised metadata.

        Raises:
            OllamaConnectionError: (or the provider-specific subclass) if the
                server is unreachable or returns a non-2xx status.
            LLMInferenceError: If the server responds but generation fails
                (e.g. context-length exceeded, OOM on the server).
            OrchestratorValidationError: If ``request`` fails provider-side
                validation (e.g. unknown model name).
        """

    @abstractmethod
    def stream_generate(
        self, request: GenerationRequest
    ) -> Iterator[str]:
        """Execute a streaming LLM completion, yielding tokens as they arrive.

        Callers iterate over the returned generator to receive text chunks.
        The generator raises provider-specific exceptions on failure, just
        like :meth:`generate`.

        Args:
            request: A fully populated :class:`GenerationRequest`.

        Yields:
            str: Successive text chunks (tokens or token groups) as they are
                received from the provider.

        Raises:
            OllamaConnectionError: If the server is unreachable.
            LLMInferenceError: If generation fails mid-stream.

        Example:
            >>> for chunk in client.stream_generate(req):
            ...     print(chunk, end="", flush=True)
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Probe the provider to verify it is reachable and operational.

        Implementations should issue a lightweight HTTP request (e.g. a
        ``GET /`` or a minimal ``POST /api/generate``) and return ``True``
        on success.  The method must **never raise**; catch all exceptions
        internally and return ``False``.

        Returns:
            ``True`` if the provider responded successfully; ``False``
            otherwise.
        """

    # ------------------------------------------------------------------
    # Optional overridable hooks — concrete implementations are provided
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int | None:
        """Estimate the number of tokens in *text*.

        The default implementation returns ``None`` (unknown).  Subclasses
        that have access to a tokeniser (e.g. via ``tiktoken`` or the
        provider's own API) should override this to return a real count.

        Args:
            text: The string whose token count is requested.

        Returns:
            Integer token count, or ``None`` if not implemented by this
            client.
        """
        return None

    def get_model_info(self) -> dict[str, Any]:
        """Return provider-specific metadata about the configured model.

        The default implementation returns a minimal dict with the client
        class name and model name.  Subclasses should override to include
        richer provider metadata (context window, quantisation, etc.).

        Returns:
            A dictionary of model metadata.  The only guaranteed key is
            ``"model_name"``.
        """
        return {
            "model_name": self.model_name,
            "provider": self.__class__.__name__,
            "base_url": self.base_url,
        }

    # ------------------------------------------------------------------
    # Concrete helpers — used internally and by subclasses
    # ------------------------------------------------------------------

    def _timed_call(
        self, fn: Any, *args: Any, **kwargs: Any
    ) -> tuple[Any, float]:
        """Execute ``fn(*args, **kwargs)`` and return (result, elapsed_seconds).

        A lightweight utility that wraps any callable with wall-clock timing.
        Used by concrete ``generate()`` implementations to populate
        :attr:`GenerationResponse.duration_seconds` without duplicating
        timing boilerplate.

        Args:
            fn: Any callable to time.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            A 2-tuple ``(return_value, elapsed_seconds)`` where
            ``elapsed_seconds`` is a ``float`` measured with
            :func:`time.perf_counter`.
        """
        start: float = time.perf_counter()
        result: Any = fn(*args, **kwargs)
        elapsed: float = time.perf_counter() - start
        return result, elapsed

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"base_url={self.base_url!r}, "
            f"timeout={self.timeout}s)"
        )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "UNSET_MAX_TOKENS",
    "GenerationRequest",
    "GenerationResponse",
    "BaseLLMClient",
]
