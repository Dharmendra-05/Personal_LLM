# ==== core/exceptions.py ====
"""
Custom exception hierarchy for the Personal LLM Orchestrator.

Design philosophy
-----------------
* **Never raise built-in exceptions directly** — Every error surface in this
  codebase raises a subclass of :class:`PersonalLLMException`.  This gives
  callers (including the top-level error handler and test suites) a single
  ``except`` clause that catches *all* domain errors without accidentally
  swallowing unrelated runtime errors.

* **Structured metadata** — Every exception carries a human-readable
  ``message``, an optional machine-readable ``error_code`` (useful for API
  responses), and an optional ``details`` dict for arbitrary context.

* **Cause chaining** — Raise with ``raise FooError("...") from original_exc``
  to preserve the original traceback.  The hierarchy does not swallow causes.

* **Single import surface** — Import everything from ``core.exceptions``; do
  not reach into sub-modules.

Hierarchy overview
------------------
::

    PersonalLLMException                    ← base for all domain errors
    ├── ConfigurationError                  ← bad/missing config at startup
    │   └── ModelConfigurationError         ← bad LLM model specification
    ├── ConnectionError                     ← network / transport failures
    │   └── OllamaConnectionError           ← Ollama server unreachable / timeout
    ├── VectorStoreError                    ← ChromaDB operation failures
    │   ├── CollectionNotFoundError         ← named collection absent
    │   └── EmbeddingError                  ← embedding generation failed
    ├── PipelineError                       ← RAG / orchestration failures
    │   ├── ContextRetrievalError           ← retrieval step failed
    │   └── LLMInferenceError               ← model inference step failed
    └── ValidationError                     ← input / output schema violation
"""

from __future__ import annotations

from typing import Any


# ===========================================================================
# Base exception
# ===========================================================================


class PersonalLLMException(Exception):
    """Base class for all Personal LLM Orchestrator domain exceptions.

    All concrete exceptions in this module inherit from this class.
    Catching ``PersonalLLMException`` guarantees you handle every error
    originating from within the orchestrator.

    Attributes:
        message: Human-readable description of the error condition.
        error_code: Optional machine-readable identifier (e.g. "OLLAMA_001").
            Intended for use in structured API error responses.
        details: Optional dictionary of additional structured context.

    Args:
        message: Human-readable error description.
        error_code: Optional machine-readable error identifier.
        details: Optional mapping of supplementary context key/value pairs.

    Example:
        >>> raise PersonalLLMException("Something went wrong", error_code="GEN_001")
        Traceback (most recent call last):
            ...
        core.exceptions.PersonalLLMException: [GEN_001] Something went wrong
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message: str = message
        self.error_code: str | None = error_code
        self.details: dict[str, Any] = details or {}
        formatted: str = f"[{error_code}] {message}" if error_code else message
        super().__init__(formatted)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={self.details!r})"
        )


# ===========================================================================
# Configuration errors
# ===========================================================================


class ConfigurationError(PersonalLLMException):
    """Raised when application configuration is invalid or incomplete.

    Typically raised during startup when ``core.config.AppSettings``
    validation fails or a required environment variable is absent.

    Args:
        message: Description of the configuration problem.
        error_code: Optional code; defaults to "CFG_001" if not provided.
        details: Optional mapping of context (e.g. ``{"field": "OLLAMA_BASE_URL"}``).

    Example:
        >>> raise ConfigurationError(
        ...     "OLLAMA_BASE_URL must be a valid HTTP URL",
        ...     error_code="CFG_001",
        ...     details={"field": "OLLAMA_BASE_URL", "received": "not-a-url"},
        ... )
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = "CFG_001",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class ModelConfigurationError(ConfigurationError):
    """Raised when an LLM model specification is invalid or unsupported.

    Examples include referencing an Ollama model tag that has not been
    pulled, or specifying incompatible model parameters.

    Args:
        message: Description of the model configuration problem.
        model_name: The offending model identifier, if known.
        error_code: Optional code; defaults to "CFG_002".
        details: Optional additional context.

    Attributes:
        model_name: The problematic model identifier.

    Example:
        >>> raise ModelConfigurationError(
        ...     "Model 'gpt-999' is not available in Ollama",
        ...     model_name="gpt-999",
        ... )
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        error_code: str | None = "CFG_002",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.model_name: str | None = model_name
        enriched_details: dict[str, Any] = details or {}
        if model_name is not None:
            enriched_details["model_name"] = model_name
        super().__init__(message, error_code=error_code, details=enriched_details)


# ===========================================================================
# Connection / network errors
# ===========================================================================


class OrchestratorConnectionError(PersonalLLMException):
    """Raised when a network or transport-level failure occurs.

    Use this class (or its subclasses) for any situation where a remote
    service is unreachable, times out, or returns an unexpected HTTP status.

    Args:
        message: Description of the connection failure.
        url: The target URL that was being contacted.
        error_code: Optional code; defaults to "CONN_001".
        details: Optional additional context.

    Attributes:
        url: Target URL of the failed connection attempt.

    Note:
        Named ``OrchestratorConnectionError`` to avoid shadowing the
        built-in ``ConnectionError``.
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        error_code: str | None = "CONN_001",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.url: str | None = url
        enriched_details: dict[str, Any] = details or {}
        if url is not None:
            enriched_details["url"] = url
        super().__init__(message, error_code=error_code, details=enriched_details)


class OllamaConnectionError(OrchestratorConnectionError):
    """Raised when the Ollama server is unreachable or returns an error.

    Specific sub-scenarios include:
    * Connection refused (server not running).
    * Request timeout exceeded (model loading / slow inference).
    * Non-2xx HTTP status from the Ollama API.

    Args:
        message: Description of the Ollama connectivity problem.
        url: The Ollama endpoint URL that was targeted.
        status_code: HTTP status code returned by Ollama, if available.
        error_code: Optional code; defaults to "OLLAMA_001".
        details: Optional additional context.

    Attributes:
        status_code: HTTP status code from the failed Ollama response, or
            ``None`` if the server was not reached at all.

    Example:
        >>> raise OllamaConnectionError(
        ...     "Ollama server refused connection",
        ...     url="http://localhost:11434/api/generate",
        ...     error_code="OLLAMA_001",
        ... )
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        status_code: int | None = None,
        error_code: str | None = "OLLAMA_001",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.status_code: int | None = status_code
        enriched_details: dict[str, Any] = details or {}
        if status_code is not None:
            enriched_details["status_code"] = status_code
        super().__init__(
            message, url=url, error_code=error_code, details=enriched_details
        )


# ===========================================================================
# Vector store errors
# ===========================================================================


class VectorStoreError(PersonalLLMException):
    """Raised when a ChromaDB operation fails.

    Serves as the base class for all vector store related errors.

    Args:
        message: Description of the vector store failure.
        collection: Name of the ChromaDB collection involved, if applicable.
        error_code: Optional code; defaults to "VS_001".
        details: Optional additional context.

    Attributes:
        collection: The ChromaDB collection name associated with the failure.
    """

    def __init__(
        self,
        message: str,
        collection: str | None = None,
        error_code: str | None = "VS_001",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.collection: str | None = collection
        enriched_details: dict[str, Any] = details or {}
        if collection is not None:
            enriched_details["collection"] = collection
        super().__init__(message, error_code=error_code, details=enriched_details)


class CollectionNotFoundError(VectorStoreError):
    """Raised when a requested ChromaDB collection does not exist.

    Args:
        collection: The name of the missing collection.
        error_code: Optional code; defaults to "VS_002".
        details: Optional additional context.

    Example:
        >>> raise CollectionNotFoundError(collection="my_docs")
    """

    def __init__(
        self,
        collection: str,
        error_code: str | None = "VS_002",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=f"ChromaDB collection '{collection}' does not exist.",
            collection=collection,
            error_code=error_code,
            details=details,
        )


class EmbeddingError(VectorStoreError):
    """Raised when the sentence-transformer embedding step fails.

    Causes include: model not loaded, device OOM, or malformed input text.

    Args:
        message: Description of the embedding failure.
        model_name: Embedding model that was being used.
        error_code: Optional code; defaults to "VS_003".
        details: Optional additional context.

    Attributes:
        model_name: The embedding model name associated with the failure.

    Example:
        >>> raise EmbeddingError(
        ...     "CUDA out of memory during embedding",
        ...     model_name="all-MiniLM-L6-v2",
        ... )
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        error_code: str | None = "VS_003",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.model_name: str | None = model_name
        enriched_details: dict[str, Any] = details or {}
        if model_name is not None:
            enriched_details["model_name"] = model_name
        super().__init__(
            message=message,
            error_code=error_code,
            details=enriched_details,
        )


# ===========================================================================
# Pipeline / orchestration errors
# ===========================================================================


class PipelineError(PersonalLLMException):
    """Raised when an orchestration or RAG pipeline step fails.

    This is the base class for errors that occur during the multi-step
    retrieval-augmented generation process.

    Args:
        message: Description of the pipeline failure.
        stage: Name of the pipeline stage where the failure occurred
            (e.g. ``"retrieval"``, ``"augmentation"``, ``"generation"``).
        error_code: Optional code; defaults to "PIPE_001".
        details: Optional additional context.

    Attributes:
        stage: The pipeline stage where the failure was detected.
    """

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        error_code: str | None = "PIPE_001",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.stage: str | None = stage
        enriched_details: dict[str, Any] = details or {}
        if stage is not None:
            enriched_details["stage"] = stage
        super().__init__(message, error_code=error_code, details=enriched_details)


class ContextRetrievalError(PipelineError):
    """Raised when the retrieval stage of the RAG pipeline fails.

    Typical causes: ChromaDB query error, embedding failure, or zero
    relevant documents found above the similarity threshold.

    Args:
        message: Description of the retrieval failure.
        query: The query string that triggered the failure.
        error_code: Optional code; defaults to "PIPE_002".
        details: Optional additional context.

    Attributes:
        query: The retrieval query that caused the failure.

    Example:
        >>> raise ContextRetrievalError(
        ...     "No documents found above similarity threshold 0.75",
        ...     query="What is RAG?",
        ... )
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        error_code: str | None = "PIPE_002",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.query: str | None = query
        enriched_details: dict[str, Any] = details or {}
        if query is not None:
            enriched_details["query"] = query
        super().__init__(
            message=message,
            stage="retrieval",
            error_code=error_code,
            details=enriched_details,
        )


class LLMInferenceError(PipelineError):
    """Raised when the LLM generation stage of the pipeline fails.

    Typical causes: Ollama returned a non-200 response during streaming,
    token limit exceeded, or the model produced an unparseable response.

    Args:
        message: Description of the inference failure.
        model_name: Model that was being used for inference.
        prompt_tokens: Approximate token count of the prompt, if known.
        error_code: Optional code; defaults to "PIPE_003".
        details: Optional additional context.

    Attributes:
        model_name: The model being used when the failure occurred.
        prompt_tokens: Estimated number of prompt tokens.

    Example:
        >>> raise LLMInferenceError(
        ...     "Ollama stream interrupted after 512 tokens",
        ...     model_name="mistral",
        ...     prompt_tokens=1024,
        ... )
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        prompt_tokens: int | None = None,
        error_code: str | None = "PIPE_003",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.model_name: str | None = model_name
        self.prompt_tokens: int | None = prompt_tokens
        enriched_details: dict[str, Any] = details or {}
        if model_name is not None:
            enriched_details["model_name"] = model_name
        if prompt_tokens is not None:
            enriched_details["prompt_tokens"] = prompt_tokens
        super().__init__(
            message=message,
            stage="generation",
            error_code=error_code,
            details=enriched_details,
        )


# ===========================================================================
# Validation errors
# ===========================================================================


class OrchestratorValidationError(PersonalLLMException):
    """Raised when input or output data fails schema validation.

    Distinct from ``pydantic.ValidationError`` — this exception is raised
    by *application-layer* validation logic (e.g. checking that a user
    query is non-empty or that a document chunk does not exceed the
    context window).

    Args:
        message: Description of the validation failure.
        field: The name of the field that failed validation, if applicable.
        received: The value that was received (avoid logging secrets here).
        error_code: Optional code; defaults to "VAL_001".
        details: Optional additional context.

    Attributes:
        field: The validated field name.
        received: The offending value.

    Note:
        Named ``OrchestratorValidationError`` to avoid shadowing
        ``pydantic.ValidationError``.

    Example:
        >>> raise OrchestratorValidationError(
        ...     "Query must not be empty",
        ...     field="query",
        ...     received="",
        ... )
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        received: Any = None,
        error_code: str | None = "VAL_001",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.field: str | None = field
        self.received: Any = received
        enriched_details: dict[str, Any] = details or {}
        if field is not None:
            enriched_details["field"] = field
        if received is not None:
            enriched_details["received"] = repr(received)
        super().__init__(message, error_code=error_code, details=enriched_details)


# ===========================================================================
# Public re-export surface
# ===========================================================================

__all__: list[str] = [
    # Base
    "PersonalLLMException",
    # Configuration
    "ConfigurationError",
    "ModelConfigurationError",
    # Connection
    "OrchestratorConnectionError",
    "OllamaConnectionError",
    # Vector store
    "VectorStoreError",
    "CollectionNotFoundError",
    "EmbeddingError",
    # Pipeline
    "PipelineError",
    "ContextRetrievalError",
    "LLMInferenceError",
    # Validation
    "OrchestratorValidationError",
]
