# ==== models/ollama_client.py ====
"""
Concrete Ollama REST API client for the Personal LLM Orchestrator.

This module implements :class:`BaseLLMClient` against the Ollama HTTP API
(``/api/generate``, ``/api/tags``, ``/api/show``).  It is the only module in
the codebase that may import ``requests`` directly.

Ollama API reference
--------------------
* ``POST /api/generate``  — single-shot or streaming text generation.
* ``GET  /api/tags``      — list all locally-pulled models.
* ``POST /api/show``      — retrieve metadata for a specific model.

Error mapping
-------------
Every ``requests`` exception is caught here and re-raised as a domain
exception from ``core.exceptions``.  Nothing from ``requests`` ever leaks
out of this module.

+---------------------------------+------------------------------------------+
| ``requests`` exception          | Domain exception raised                  |
+=================================+==========================================+
| ``ConnectionError``             | ``OllamaConnectionError`` (OLLAMA_001)   |
| ``Timeout``                     | ``OllamaConnectionError`` (OLLAMA_002)   |
| ``HTTPError`` (non-2xx)         | ``OllamaConnectionError`` (OLLAMA_003)   |
| ``JSONDecodeError``             | ``LLMInferenceError``   (PIPE_003)       |
| ``KeyError`` in response parse  | ``LLMInferenceError``   (PIPE_003)       |
+---------------------------------+------------------------------------------+

Thread safety
-------------
A single :class:`requests.Session` is created per client instance.  The
session manages connection pooling for keep-alive reuse.  The session itself
is **not** thread-safe for concurrent writes; if concurrent generation calls
are needed, instantiate one client per thread or use a thread-local pool.
"""

from __future__ import annotations

import json
from typing import Any, Final, Iterator

import requests
import requests.exceptions

from core.exceptions import LLMInferenceError, OllamaConnectionError
from core.utils.logger import get_logger
from models.base import (
    UNSET_MAX_TOKENS,
    BaseLLMClient,
    GenerationRequest,
    GenerationResponse,
)

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ollama endpoint paths (relative to ``base_url``).
_ENDPOINT_CHAT: Final[str] = "/api/chat"
_ENDPOINT_TAGS: Final[str] = "/api/tags"
_ENDPOINT_SHOW: Final[str] = "/api/show"
_ENDPOINT_HEALTH: Final[str] = "/"

#: ``Content-Type`` header value for all JSON payloads.
_JSON_CONTENT_TYPE: Final[str] = "application/json"

#: Ollama's ``finish_reason`` equivalent key in the response JSON.
_OLLAMA_DONE_KEY: Final[str] = "done"
_OLLAMA_RESPONSE_KEY: Final[str] = "response"


# ---------------------------------------------------------------------------
# Pydantic model for the per-model YAML configuration
# (used by the registry to pass validated params into the constructor)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Concrete client
# ---------------------------------------------------------------------------


class OllamaClient(BaseLLMClient):
    """Concrete LLM client that communicates with a local Ollama server.

    This class is the sole owner of all Ollama-specific HTTP logic.  It
    translates :class:`~models.base.GenerationRequest` objects into Ollama
    JSON payloads, handles every failure mode, and returns normalised
    :class:`~models.base.GenerationResponse` objects.

    Args:
        model_name: The Ollama model tag to use by default
            (e.g. ``"llama3"``, ``"gemma:7b"``).  Must already be pulled
            into the local Ollama registry.
        base_url: Base URL of the Ollama server.
            Defaults to ``"http://localhost:11434"``.
        timeout: Default per-request timeout in seconds.  Defaults to 120.
        default_temperature: Fallback temperature when the request does not
            specify one.  Overridden by ``GenerationRequest.temperature``.
        default_max_tokens: Fallback ``num_predict`` (Ollama's token cap)
            when the request sets ``max_tokens = UNSET_MAX_TOKENS``.
            ``0`` means Ollama uses its own default.
        keep_alive: Value forwarded to Ollama's ``keep_alive`` field,
            controlling how long the model stays loaded in memory after
            the request.  Accepts Ollama duration strings (e.g. ``"5m"``,
            ``"0"``).  Defaults to ``"5m"``.

    Attributes:
        model_name: Default model identifier for this client.
        base_url: The normalised (no trailing slash) Ollama base URL.
        timeout: Default HTTP request timeout in seconds.
        default_temperature: Fallback inference temperature.
        default_max_tokens: Fallback maximum token count.
        keep_alive: Ollama model keep-alive duration string.

    Raises:
        ValueError: If ``model_name`` or ``base_url`` are empty strings.

    Example:
        >>> client = OllamaClient(model_name="llama3")
        >>> resp = client.generate(
        ...     GenerationRequest(prompt="Hello!", temperature=0.5)
        ... )
        >>> print(resp.text)
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
        default_temperature: float = 0.7,
        default_max_tokens: int = 0,
        keep_alive: str = "5m",
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
        )
        self.default_temperature: float = default_temperature
        self.default_max_tokens: int = default_max_tokens
        self.keep_alive: str = keep_alive

        # One session per client instance — enables HTTP keep-alive reuse.
        self._session: requests.Session = self._build_session()

        logger.debug(
            "OllamaClient initialised: model=%s base_url=%s timeout=%ds",
            self.model_name,
            self.base_url,
            self.timeout,
        )

    # ------------------------------------------------------------------
    # BaseLLMClient abstract method implementations
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Execute a blocking, non-streaming completion against Ollama.

        Sends a ``POST /api/generate`` request with ``stream=false``.
        The full JSON response is received in a single HTTP response body
        and parsed into a :class:`GenerationResponse`.

        Args:
            request: A :class:`~models.base.GenerationRequest` with all
                inference parameters.

        Returns:
            A :class:`~models.base.GenerationResponse` containing the
            generated text, token counts (if reported by Ollama), and
            wall-clock duration.

        Raises:
            OllamaConnectionError: If the server is unreachable, the
                connection is refused, or the request times out.
            LLMInferenceError: If Ollama returns a non-2xx status, the
                response body is not valid JSON, or expected response keys
                are missing.

        Example:
            >>> req = GenerationRequest(prompt="What is 2+2?", max_tokens=32)
            >>> resp = client.generate(req)
            >>> resp.text
            '4'
        """
        payload: dict[str, Any] = self._build_payload(request, stream=False)
        url: str = f"{self.base_url}{_ENDPOINT_CHAT}"

        logger.debug(
            "generate() → POST %s | model=%s temp=%.2f max_tokens=%s",
            url,
            payload.get("model"),
            payload.get("options", {}).get("temperature", self.default_temperature),
            payload.get("options", {}).get("num_predict", "default"),
        )

        start_time: float
        raw_response: requests.Response
        raw_response, elapsed = self._timed_call(
            self._post_json, url, payload, stream=False
        )

        return self._parse_generate_response(
            raw=raw_response,
            duration_seconds=elapsed,
            requested_model=request.model_name or self.model_name,
        )

    def stream_generate(
        self, request: GenerationRequest
    ) -> Iterator[str]:
        """Execute a streaming completion, yielding text chunks as they arrive.

        Sends a ``POST /api/generate`` request with ``stream=true``.
        Each newline-delimited JSON object is decoded and its ``"response"``
        field yielded to the caller immediately, enabling real-time display.

        Args:
            request: A :class:`~models.base.GenerationRequest`.

        Yields:
            str: Successive text chunks produced by Ollama.  The final
                chunk (where ``"done": true``) is **not** yielded as text;
                it is used only to log completion statistics.

        Raises:
            OllamaConnectionError: If the server is unreachable.
            LLMInferenceError: If a streamed chunk cannot be parsed or the
                stream terminates unexpectedly.

        Example:
            >>> req = GenerationRequest(prompt="Count to five.", max_tokens=64)
            >>> for chunk in client.stream_generate(req):
            ...     print(chunk, end="", flush=True)
        """
        payload: dict[str, Any] = self._build_payload(request, stream=True)
        url: str = f"{self.base_url}{_ENDPOINT_CHAT}"

        logger.debug(
            "stream_generate() → POST %s | model=%s",
            url,
            payload.get("model"),
        )

        raw_response: requests.Response = self._post_json(url, payload, stream=True)

        try:
            for line in raw_response.iter_lines():
                if not line:
                    continue
                chunk: dict[str, Any] = self._decode_json_line(
                    line=line,
                    url=url,
                    model_name=request.model_name or self.model_name,
                )
                text_fragment: str = chunk.get("message", {}).get("content", "")
                if text_fragment:
                    yield text_fragment

                if chunk.get(_OLLAMA_DONE_KEY, False):
                    logger.debug(
                        "stream_generate() complete | model=%s eval_count=%s",
                        chunk.get("model", "unknown"),
                        chunk.get("eval_count", "?"),
                    )
                    return

        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                message=f"Stream interrupted: {exc}",
                url=url,
                error_code="OLLAMA_004",
                details={"model": request.model_name or self.model_name},
            ) from exc

    def health_check(self) -> bool:
        """Probe the Ollama server to verify it is reachable.

        Issues a lightweight ``GET /`` request.  Ollama returns HTTP 200
        with the plain-text body ``"Ollama is running"`` when healthy.

        Returns:
            ``True`` if Ollama responded with HTTP 200; ``False`` for any
            connection failure, timeout, or non-200 status.  **Never
            raises** — all exceptions are caught internally.

        Example:
            >>> client.health_check()
            True
        """
        url: str = f"{self.base_url}{_ENDPOINT_HEALTH}"
        try:
            resp = self._session.get(url, timeout=5)
            is_ok: bool = resp.status_code == 200
            logger.debug(
                "health_check() → %s (%d)", "OK" if is_ok else "FAIL",
                resp.status_code,
            )
            return is_ok
        except requests.exceptions.RequestException as exc:
            logger.warning("health_check() failed due to request error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public Ollama-specific extensions
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[dict[str, Any]]:
        """Return metadata for all models currently in the Ollama registry.

        Issues a ``GET /api/tags`` request and returns the ``"models"``
        array from the response.

        Returns:
            A list of dicts, each describing one model:
            ``{"name": str, "size": int, "digest": str, ...}``.
            Returns an empty list if the server is unreachable.

        Raises:
            OllamaConnectionError: If the HTTP request fails.
            LLMInferenceError: If the response body cannot be parsed.

        Example:
            >>> models = client.list_local_models()
            >>> [m["name"] for m in models]
            ['llama3:latest', 'gemma:7b']
        """
        url: str = f"{self.base_url}{_ENDPOINT_TAGS}"
        logger.debug("list_local_models() → GET %s", url)

        try:
            resp: requests.Response = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                message=f"Cannot reach Ollama at {url}: {exc}",
                url=url,
                error_code="OLLAMA_001",
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                message=f"Timeout fetching model list from {url}.",
                url=url,
                error_code="OLLAMA_002",
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise OllamaConnectionError(
                message=f"Ollama returned HTTP {resp.status_code} for GET {url}.",
                url=url,
                status_code=resp.status_code,
                error_code="OLLAMA_003",
            ) from exc

        try:
            data: dict[str, Any] = resp.json()
            models: list[dict[str, Any]] = data.get("models", [])
            logger.debug("list_local_models() → found %d models", len(models))
            return models
        except (json.JSONDecodeError, KeyError) as exc:
            raise LLMInferenceError(
                message=f"Failed to parse /api/tags response: {exc}",
                model_name=self.model_name,
                error_code="PIPE_003",
            ) from exc

    def show_model_info(self, model_name: str | None = None) -> dict[str, Any]:
        """Fetch detailed metadata for a specific model from Ollama.

        Issues a ``POST /api/show`` request.  Returns the raw JSON dict
        which may include ``modelfile``, ``parameters``, ``template``,
        ``details``, etc.

        Args:
            model_name: Model to inspect.  Defaults to
                :attr:`self.model_name` when ``None``.

        Returns:
            Raw metadata dict as returned by the Ollama ``/api/show``
            endpoint.

        Raises:
            OllamaConnectionError: If the HTTP request fails.
            LLMInferenceError: If the response body cannot be parsed.
        """
        target: str = model_name or self.model_name
        url: str = f"{self.base_url}{_ENDPOINT_SHOW}"
        logger.debug("show_model_info() → POST %s | model=%s", url, target)

        try:
            resp: requests.Response = self._session.post(
                url,
                json={"name": target},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            info: dict[str, Any] = resp.json()
            logger.debug("show_model_info() → received %d keys", len(info))
            return info
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                message=f"Cannot reach Ollama at {url}: {exc}",
                url=url,
                error_code="OLLAMA_001",
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                message=f"Timeout fetching model info for '{target}'.",
                url=url,
                error_code="OLLAMA_002",
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise OllamaConnectionError(
                message=(
                    f"Ollama returned HTTP {resp.status_code} for "
                    f"POST /api/show (model={target!r})."
                ),
                url=url,
                status_code=resp.status_code,
                error_code="OLLAMA_003",
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMInferenceError(
                message=f"Failed to parse /api/show response: {exc}",
                model_name=target,
                error_code="PIPE_003",
            ) from exc

    def get_model_info(self) -> dict[str, Any]:
        """Return provider metadata for the configured model.

        Overrides :meth:`~models.base.BaseLLMClient.get_model_info` to
        include Ollama-specific fields.  Falls back to the base
        implementation if ``/api/show`` is unreachable.

        Returns:
            Dict containing at minimum ``"model_name"``, ``"provider"``,
            ``"base_url"``, and ``"healthy"``.  If ``/api/show`` succeeds,
            includes Ollama-reported ``"details"`` and ``"parameters"``.
        """
        base_info: dict[str, Any] = {
            "model_name": self.model_name,
            "provider": "ollama",
            "base_url": self.base_url,
            "healthy": self.health_check(),
        }
        try:
            ollama_info = self.show_model_info()
            base_info["details"] = ollama_info.get("details", {})
            base_info["parameters"] = ollama_info.get("parameters", "")
        except Exception:  # noqa: BLE001
            logger.debug("get_model_info(): /api/show unavailable; using base info.")
        return base_info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Construct and configure the shared ``requests.Session``.

        Sets default headers and a reasonable retry-less adapter (retries are
        handled at the orchestration layer, not here).

        Returns:
            A configured :class:`requests.Session` instance.
        """
        session = requests.Session()
        session.headers.update(
            {
                "Content-Type": _JSON_CONTENT_TYPE,
                "Accept": _JSON_CONTENT_TYPE,
                "User-Agent": "PersonalLLMOrchestrator/0.1.0",
            }
        )
        return session

    def _build_payload(
        self,
        request: GenerationRequest,
        stream: bool,
    ) -> dict[str, Any]:
        """Translate a :class:`GenerationRequest` into an Ollama JSON payload.

        Args:
            request: The generation request to translate.
            stream: Whether to request a streaming (``true``) or
                non-streaming (``false``) response from Ollama.

        Returns:
            A dict ready to be serialised as the POST body.
        """
        # Resolve model name: request overrides instance default.
        model: str = request.model_name or self.model_name

        # Build the Ollama "options" sub-object from request parameters.
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repeat_penalty": request.repeat_penalty,
        }

        # Only include top_k if explicitly set (0 = disabled).
        if request.top_k > 0:
            options["top_k"] = request.top_k

        # Resolve num_predict (Ollama's max_tokens equivalent).
        if request.max_tokens != UNSET_MAX_TOKENS:
            options["num_predict"] = request.max_tokens
        elif self.default_max_tokens > 0:
            options["num_predict"] = self.default_max_tokens

        # Stop sequences.
        if request.stop_sequences:
            options["stop"] = request.stop_sequences

        if request.messages is not None:
            messages = request.messages
        else:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            if request.prompt:
                messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "keep_alive": self.keep_alive,
        }

        # Tools are deliberately omitted for Ollama as it is not consistently supported.
        # if request.tools:
        #     payload["tools"] = request.tools

        # Provider-specific pass-through parameters.
        if request.extra_params:
            payload.update(request.extra_params)

        return payload

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        stream: bool,
    ) -> requests.Response:
        """Send a ``POST`` request with a JSON body.

        All ``requests`` exceptions are caught and re-raised as domain
        exceptions.  Callers receive either a :class:`requests.Response` or
        a domain exception — never a raw ``requests`` exception.

        Args:
            url: The full target URL.
            payload: JSON-serialisable request body dict.
            stream: If ``True``, the response body is not downloaded
                immediately (enables ``iter_lines()``).

        Returns:
            The :class:`requests.Response` object.  HTTP error status codes
            cause an ``OllamaConnectionError`` to be raised before returning.

        Raises:
            OllamaConnectionError: On ``ConnectionError``, ``Timeout``, or
                non-2xx HTTP status.
        """
        try:
            resp: requests.Response = self._session.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                message=(
                    f"Cannot connect to Ollama at {url}. "
                    f"Is the server running? Detail: {exc}"
                ),
                url=url,
                error_code="OLLAMA_001",
                details={"model": payload.get("model")},
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                message=(
                    f"Ollama request timed out after {self.timeout}s "
                    f"(model={payload.get('model')!r}). "
                    f"Consider increasing OLLAMA_REQUEST_TIMEOUT."
                ),
                url=url,
                error_code="OLLAMA_002",
                details={
                    "model": payload.get("model"),
                    "timeout_seconds": self.timeout,
                },
            ) from exc
        except requests.exceptions.RequestException as exc:
            # Catch-all for TooManyRedirects, SSLError, etc.
            raise OllamaConnectionError(
                message=f"Unexpected request error contacting Ollama: {exc}",
                url=url,
                error_code="OLLAMA_005",
            ) from exc

        # Raise on 4xx / 5xx.
        if not resp.ok:
            # Attempt to extract Ollama's error message from the body.
            error_body: str = ""
            try:
                error_body = resp.json().get("error", resp.text)
            except Exception:  # noqa: BLE001
                error_body = resp.text[:500]

            raise OllamaConnectionError(
                message=(
                    f"Ollama returned HTTP {resp.status_code} for "
                    f"POST {url}: {error_body}"
                ),
                url=url,
                status_code=resp.status_code,
                error_code="OLLAMA_003",
                details={
                    "model": payload.get("model"),
                    "error_body": error_body,
                },
            )

        return resp

    def _parse_generate_response(
        self,
        raw: requests.Response,
        duration_seconds: float,
        requested_model: str,
    ) -> GenerationResponse:
        """Parse a non-streaming Ollama ``/api/generate`` HTTP response.

        Args:
            raw: The completed :class:`requests.Response` from Ollama.
            duration_seconds: Wall-clock elapsed seconds for this request.
            requested_model: The model name from the original request, used
                as a fallback if the response body omits ``"model"``.

        Returns:
            A populated :class:`GenerationResponse`.

        Raises:
            LLMInferenceError: If the response body is not valid JSON or is
                missing required keys.
        """
        try:
            data: dict[str, Any] = raw.json()
        except json.JSONDecodeError as exc:
            raise LLMInferenceError(
                message=f"Ollama returned non-JSON body: {exc}",
                model_name=requested_model,
                error_code="PIPE_003",
                details={"raw_body_preview": raw.text[:200]},
            ) from exc

        # Guard: Ollama surfaces generation-level errors in the "error" key
        # even on HTTP 200 in some versions.
        if "error" in data:
            raise LLMInferenceError(
                message=f"Ollama generation error: {data['error']}",
                model_name=requested_model,
                error_code="PIPE_003",
                details={"ollama_error": data["error"]},
            )

        try:
            message: dict[str, Any] = data.get("message", {})
            generated_text: str = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
        except KeyError as exc:
            raise LLMInferenceError(
                message=(
                    f"Ollama response missing expected 'message' structure. "
                    f"Keys present: {list(data.keys())}"
                ),
                model_name=requested_model,
                error_code="PIPE_003",
                details={"response_keys": list(data.keys())},
            ) from exc

        # Token counts — reported only when eval_count is present.
        prompt_tokens: int | None = data.get("prompt_eval_count")
        completion_tokens: int | None = data.get("eval_count")
        total_tokens: int | None = (
            (prompt_tokens + completion_tokens)
            if (prompt_tokens is not None and completion_tokens is not None)
            else None
        )

        finish_reason: str = "stop" if data.get(_OLLAMA_DONE_KEY, False) else "length"
        if tool_calls:
            finish_reason = "tool_calls"

        response = GenerationResponse(
            text=generated_text.strip(),
            model_name=data.get("model", requested_model),
            duration_seconds=duration_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            raw_response=data,
            finish_reason=finish_reason,
        )

        logger.debug(
            "generate() ← model=%s tokens=%s/%s dur=%.2fs finish=%s",
            response.model_name,
            prompt_tokens,
            completion_tokens,
            duration_seconds,
            finish_reason,
        )
        return response

    def _decode_json_line(
        self,
        line: bytes | str,
        url: str,
        model_name: str,
    ) -> dict[str, Any]:
        """Decode one NDJSON line from a streaming Ollama response.

        Args:
            line: Raw bytes or string line from ``iter_lines()``.
            url: Source URL (for error context).
            model_name: Model being used (for error context).

        Returns:
            Parsed JSON dict.

        Raises:
            LLMInferenceError: If the line is not valid JSON.
        """
        text: str = line.decode("utf-8") if isinstance(line, bytes) else line
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMInferenceError(
                message=f"Failed to decode streaming chunk: {exc}",
                model_name=model_name,
                error_code="PIPE_003",
                details={"raw_line": text[:200], "url": url},
            ) from exc

    def close(self) -> None:
        """Release the underlying ``requests.Session`` and its connections.

        Call this when the client is no longer needed to free socket
        resources.  Safe to call multiple times.

        Example:
            >>> client = OllamaClient(model_name="llama3")
            >>> # ... use client ...
            >>> client.close()
        """
        self._session.close()
        logger.debug("OllamaClient.close() — session closed for model=%s", self.model_name)

    def __enter__(self) -> "OllamaClient":
        """Support use as a context manager.

        Returns:
            ``self``
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close the session on context manager exit.

        Args:
            exc_type: Exception type if one was raised, else ``None``.
            exc_val: Exception instance if one was raised, else ``None``.
            exc_tb: Traceback object if an exception was raised, else ``None``.
        """
        self.close()


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__: list[str] = ["OllamaClient"]
