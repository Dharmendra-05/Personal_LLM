# ==== models/openai_compatible_client.py ====
"""
Client for REST APIs adhering to the OpenAI Chat Completions specification.

This generic client handles OpenAI, Groq, and Hugging Face serverless 
inference endpoints since they all follow the same standard API format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests

from core.exceptions import LLMInferenceError
from core.utils.logger import get_logger
from models.base import BaseLLMClient, GenerationRequest, GenerationResponse

logger = get_logger(__name__)


class OpenAICompatibleClient(BaseLLMClient):
    """Client for LLM APIs that implement the standard OpenAI `/chat/completions` spec.

    Args:
        model_name: The target model string.
        base_url: The API base URL (e.g., "https://api.groq.com/openai/v1").
        api_key: The authentication key for the provider.
        timeout: Request timeout in seconds.
        **kwargs: Additional configuration (e.g. system_prompt, default_temp).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        timeout: int = 120,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # Ensure URL points to the unified chat completions endpoint
        if not self.base_url.endswith("/chat/completions"):
            self.endpoint = f"{self.base_url}/chat/completions"
        else:
            self.endpoint = self.base_url

        # Shared HTTP session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _build_payload(
        self, request: GenerationRequest, stream: bool = False
    ) -> dict[str, Any]:
        """Convert a GenerationRequest into an OpenAI-compatible JSON payload."""
        messages = request.messages if request.messages is not None else []
        
        if not messages:
            sys_prompt = request.system_prompt or self.system_prompt
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            if request.prompt:
                messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": request.model_name or self.model_name,
            "messages": messages,
            "stream": stream,
        }

        # Apply GenerationRequest overrides or fall back to client defaults
        temp = request.temperature if request.temperature != 0.7 else self.default_temperature
        if temp is not None:
            payload["temperature"] = temp

        if request.max_tokens > 0:
            payload["max_tokens"] = request.max_tokens

        if request.top_p != 1.0:
            payload["top_p"] = request.top_p

        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
            
        if request.tools:
            payload["tools"] = request.tools

        # Pass any extra provider-specific parameters
        if request.extra_params:
            payload.update(request.extra_params)

        return payload

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Execute a synchronous generation request.
        
        Raises:
            LLMInferenceError: On HTTP or parsing failure.
        """
        payload = self._build_payload(request, stream=False)
        
        try:
            logger.debug("POST %s (model=%s)", self.endpoint, payload["model"])
            resp = self._session.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            # Log exact response text on error for debugging API limits/auth
            if not resp.ok:
                logger.error("API Error %d: %s", resp.status_code, resp.text)
                resp.raise_for_status()
                
            data = resp.json()
            return self._parse_json_response(data, payload["model"])

        except requests.exceptions.RequestException as e:
            raise LLMInferenceError(
                message=f"API request failed: {e}",
                provider="openai_compatible",
                error_code="REQ_ERR",
                raw_response=getattr(e.response, "text", ""),
            ) from e
        except json.JSONDecodeError as e:
            raise LLMInferenceError(
                message=f"Invalid JSON response: {e}",
                provider="openai_compatible",
                error_code="PARSE_ERR",
            ) from e

    def stream_generate(
        self, request: GenerationRequest
    ) -> Iterator[GenerationResponse]:
        """Execute a streaming generation request returning chunk updates.
        
        Note: Tool calling is typically not supported well via standard SSE streams,
        so standard generation is preferred for agentic features.
        """
        payload = self._build_payload(request, stream=True)
        # Not heavily used natively since tool loops run synchronously
        yield self.generate(request)

    def _parse_json_response(
        self, data: dict[str, Any], model_name: str
    ) -> GenerationResponse:
        """Parse the OpenAI-format JSON response into a GenerationResponse object."""
        choices = data.get("choices", [])
        if not choices:
            raise LLMInferenceError(
                message="API returned empty choices list.",
                provider="openai_compatible",
                error_code="EMPTY_CHOICES",
            )
            
        message = choices[0].get("message", {})
        content = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason", "stop")
        
        # Parse OpenAI tool call format into internal generic format
        tool_calls = None
        if "tool_calls" in message:
            tool_calls = []
            for t in message["tool_calls"]:
                try:
                    args = json.loads(t["function"]["arguments"])
                except Exception:
                    args = t["function"]["arguments"]  # fallback raw string
                    
                tool_calls.append({
                    "id": t.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": t["function"]["name"],
                        "arguments": args
                    }
                })

        usage = data.get("usage", {})

        return GenerationResponse(
            text=content,
            model_name=model_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def health_check(self) -> bool:
        """Ping the API models list endpoint to verify the key is active."""
        try:
            url = self.base_url.replace("/chat/completions", "/models")
            resp = self._session.get(url, timeout=10)
            return resp.ok
        except requests.exceptions.RequestException:
            return False
