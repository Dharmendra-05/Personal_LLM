# ==== models/registry.py ====
"""
YAML-driven model registry and factory for the Personal LLM Orchestrator.

The :class:`ModelRegistry` is the single entry point for obtaining an LLM
client anywhere in the application.  Callers never construct
:class:`~models.ollama_client.OllamaClient` (or future provider clients)
directly — they always go through the registry.  This enforces:

* **Open/Closed Principle** — new provider support requires only a new
  YAML ``provider`` value and a new client class; the registry dispatch
  table is the only change point.
* **Single Responsibility** — the registry owns model discovery,
  validation, and instantiation.  Individual clients own nothing about
  configuration loading.
* **Dependency Inversion** — the rest of the application depends on
  :class:`~models.base.BaseLLMClient`, not on ``OllamaClient``.

YAML schema
-----------
Every ``*.yaml`` file in the ``model_configs/`` directory must conform to
the :class:`ModelConfigSchema` Pydantic model.  Invalid files raise
:class:`~core.exceptions.ModelConfigurationError` at load time.

Required fields::

    name: str          # Unique registry key
    provider: str      # "ollama" | future providers
    model_tag: str     # Provider-specific model identifier
    base_url: str      # Provider server base URL

Optional fields::

    timeout: int                 # Request timeout seconds (default: 120)
    temperature: float           # Default sampling temperature (default: 0.7)
    max_tokens: int              # Hard token cap (default: 0 = provider default)
    top_p: float                 # Nucleus sampling (default: 1.0)
    top_k: int                   # Top-k sampling (default: 0 = disabled)
    repeat_penalty: float        # Repetition penalty (default: 1.1)
    keep_alive: str              # Ollama model keep-alive (default: "5m")
    system_prompt: str           # Default system prompt for this model
    description: str             # Human-readable description
    tags: list[str]              # Arbitrary searchable labels

Example YAML::

    name: local-llama3
    provider: ollama
    model_tag: llama3
    base_url: http://localhost:11434
    temperature: 0.7
    max_tokens: 2048
    description: "Meta LLaMA-3 8B via Ollama"
    tags: [chat, general]
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from core.exceptions import ModelConfigurationError
from core.utils.logger import get_logger
from models.base import BaseLLMClient
from models.ollama_client import OllamaClient
from models.openai_compatible_client import OpenAICompatibleClient

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default directory (relative to this file) to scan for model YAML configs.
_DEFAULT_CONFIGS_DIR: Final[Path] = Path(__file__).parent / "model_configs"

#: Map of provider names (lowercase) to their client factory callables.
#: Extend this dict when adding new provider support.
_PROVIDER_REGISTRY: Final[dict[str, type[BaseLLMClient]]] = {
    "ollama": OllamaClient,
    "openai_compatible": OpenAICompatibleClient,
}


# ---------------------------------------------------------------------------
# YAML schema validation model
# ---------------------------------------------------------------------------


class ModelConfigSchema(BaseModel):
    """Pydantic model that validates and coerces a single model YAML config.

    Every field here maps 1-to-1 with a YAML key.  Unknown YAML keys are
    silently ignored (``extra="ignore"``), making configs forward-compatible.

    Attributes:
        name: Unique human-readable registry key.  Used as the argument to
            :meth:`~ModelRegistry.get_model`.
        provider: Case-insensitive provider identifier.  Must be a key in
            :data:`_PROVIDER_REGISTRY`.
        model_tag: The provider's own model identifier (e.g. ``"llama3"``,
            ``"gemma:7b"``).
        base_url: Base URL of the provider server.
        timeout: Request timeout in seconds.
        temperature: Default sampling temperature.
        max_tokens: Hard cap on generated tokens.  ``0`` defers to the
            provider's default.
        top_p: Nucleus sampling probability mass.
        top_k: Top-k sampling pool size.  ``0`` disables it.
        repeat_penalty: Repetition penalty factor.
        keep_alive: Ollama model keep-alive duration string.
        system_prompt: Default system prompt injected into every request.
        description: Free-text description for tooling / UIs.
        tags: Arbitrary labels for filtering / discovery.

    Example:
        >>> cfg = ModelConfigSchema(
        ...     name="local-llama3",
        ...     provider="ollama",
        ...     model_tag="llama3",
        ...     base_url="http://localhost:11434",
        ... )
    """

    name: str = Field(..., min_length=1, description="Unique registry key.")
    provider: str = Field(..., min_length=1, description="Provider identifier.")
    model_tag: str = Field(..., min_length=1, description="Provider model identifier.")

    model_config = {
        "extra": "ignore",
        "protected_namespaces": ()
    }

    @model_validator(mode="after")
    def _ensure_latest_tag(self) -> "ModelConfigSchema":
        """Force use of :latest if specifically llama3 is mentioned without a tag."""
        if self.provider == "ollama" and self.model_tag == "llama3":
            object.__setattr__(self, "model_tag", "llama3:latest")
        return self

    base_url: str = Field(
        default="http://localhost:11434",
        description="Provider server base URL.",
    )
    timeout: int = Field(
        default=120, ge=1, le=600, description="Request timeout in seconds."
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default sampling temperature."
    )
    max_tokens: int = Field(
        default=0, ge=0, description="Max tokens (0 = provider default)."
    )
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=0.0)
    keep_alive: str = Field(default="5m")
    system_prompt: str | None = Field(default=None)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, v: object) -> str:
        """Coerce provider to lowercase for case-insensitive matching.

        Args:
            v: Raw value from YAML (should be a string).

        Returns:
            Lowercase provider string.

        Raises:
            ValueError: If ``v`` is not a string.
        """
        if not isinstance(v, str):
            raise ValueError(f"provider must be a string, got {type(v).__name__}")
        return v.lower().strip()

    @model_validator(mode="after")
    def _validate_provider_known(self) -> "ModelConfigSchema":
        """Ensure the provider maps to a registered client class.

        Returns:
            The validated schema instance.

        Raises:
            ValueError: If ``provider`` is not in :data:`_PROVIDER_REGISTRY`.
        """
        if self.provider not in _PROVIDER_REGISTRY:
            supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Supported providers: {supported}."
            )
        return self


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Discovers, validates, and instantiates LLM clients from YAML configs.

    The registry is the single authoritative source of model instances in the
    orchestrator.  It supports:

    * **Lazy instantiation** — Clients are constructed on first access and
      cached; models that are never used are never initialised.
    * **Thread-safe caching** — A ``threading.Lock`` serialises concurrent
      first-access client construction.
    * **Hot-reload** — Call :meth:`reload` to re-scan the config directory
      and update the registry without restarting the process.
    * **Extensible dispatch** — Add a new entry to :data:`_PROVIDER_REGISTRY`
      to support a new provider without touching this class.

    Args:
        configs_dir: Path to the directory containing ``*.yaml`` model config
            files.  Defaults to ``models/model_configs/`` relative to the
            package root.
        auto_load: If ``True`` (default), the registry scans and validates all
            YAML files during ``__init__``.  Set to ``False`` to defer loading
            until the first :meth:`get_model` call.

    Attributes:
        configs_dir: Resolved path to the model configs directory.

    Raises:
        ModelConfigurationError: If ``configs_dir`` does not exist or any
            YAML file fails schema validation (when ``auto_load=True``).

    Example:
        >>> registry = ModelRegistry()
        >>> registry.list_models()
        ['local-llama3', 'local-gemma']
        >>> client = registry.get_model("local-llama3")
        >>> client.health_check()
        True
    """

    def __init__(
        self,
        configs_dir: Path | str = _DEFAULT_CONFIGS_DIR,
        auto_load: bool = True,
    ) -> None:
        self.configs_dir: Path = Path(configs_dir).resolve()

        # Internal state
        # _configs: validated config schemas, keyed by model name
        self._configs: dict[str, ModelConfigSchema] = {}
        # _clients: lazily-instantiated client instances, keyed by model name
        self._clients: dict[str, BaseLLMClient] = {}
        # _lock: serialises lazy client construction under concurrent access
        self._lock: threading.Lock = threading.Lock()

        if auto_load:
            self._load_all_configs()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_model(self, name: str) -> BaseLLMClient:
        """Return a ready-to-use LLM client for the named model.

        The client is constructed on first call and cached for all subsequent
        calls.  Construction is thread-safe.

        Args:
            name: Registry name of the model as specified in its YAML
                ``name`` field.

        Returns:
            A concrete :class:`~models.base.BaseLLMClient` subclass
            appropriate for the model's provider.

        Raises:
            ModelConfigurationError: If *name* is not in the registry or the
                config directory has not been loaded.

        Example:
            >>> client = registry.get_model("local-llama3")
            >>> resp = client.generate(GenerationRequest(prompt="Hi!"))
        """
        if not self._configs:
            # Lazily load if auto_load was False.
            self._load_all_configs()

        if name == "auto-advanced":
            # --- Smart Priority Fallback Chain ---
            # Tier 1 (Expensive): Anthropic/OpenAI
            # Tier 2 (Free/Fast): Groq
            # Tier 3 (Free/Backup): Hugging Face
            # Tier 4 (Offline Fallback): Local Ollama Model
            from core.config import get_settings
            keys = get_settings().api_keys
            log = get_logger(__name__)
            
            resolved_client = None
            if keys.anthropic:
                pass # TODO Anthropic client mapping if needed in the future
            if keys.openai and not resolved_client:
                resolved_client = self._build_dynamic_openai_client("gpt-4o", "https://api.openai.com/v1", keys.openai)
            if keys.groq and not resolved_client:
                resolved_client = self._build_dynamic_openai_client("llama3-70b-8192", "https://api.groq.com/openai/v1", keys.groq)
            if keys.hf and not resolved_client:
                # Meta-Llama-3.1-70B-Instruct is a solid HF serverless model
                resolved_client = self._build_dynamic_openai_client("meta-llama/Meta-Llama-3.1-70B-Instruct", "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-70B-Instruct/v1", keys.hf)
                
            if resolved_client:
                # Cache the resolved client under 'auto-advanced' so it's reused
                self._clients["auto-advanced"] = resolved_client
                return resolved_client
                
            # If all external API checks fail, resolve to local Ollama
            log.warning("Advanced routing requested but no API keys configured. Falling back to offline defaults.")
            name = get_settings().ollama.default_model

        if name not in self._configs and name != "auto-advanced":
            # Standard error if model flat out doesn't exist
            available: list[str] = sorted(self._configs.keys())
            raise ModelConfigurationError(
                message=(
                    f"Model '{name}' is not registered. "
                    f"Available models: {available}"
                ),
                model_name=name,
                error_code="CFG_002",
                details={"available_models": available},
            )

        # Double-checked locking: fast path (no lock) for already-cached.
        if name in self._clients:
            return self._clients[name]

        with self._lock:
            # Re-check inside the lock to avoid double-construction.
            if name not in self._clients:
                self._clients[name] = self._instantiate_client(
                    self._configs[name]
                )
                logger.info(
                    "ModelRegistry: instantiated '%s' (%s / %s)",
                    name,
                    self._configs[name].provider,
                    self._configs[name].model_tag,
                )

        return self._clients[name]

    def _build_dynamic_openai_client(self, model_tag: str, base_url: str, api_key: str) -> BaseLLMClient:
        """Dynamically build an OpenAICompatibleClient without needing a YAML config."""
        return _PROVIDER_REGISTRY["openai_compatible"](
            model_name=model_tag,
            base_url=base_url,
            api_key=api_key,
        )

    def list_models(self) -> list[str]:
        """Return a sorted list of all registered model names.

        Returns:
            Alphabetically sorted list of model name strings from loaded
            YAML configs. Includes 'auto-advanced'.
        """
        models = sorted(self._configs.keys())
        if "auto-advanced" not in models:
            models.append("auto-advanced")
        return sorted(models)

    def list_model_details(self) -> list[dict[str, Any]]:
        """Return summary metadata for every registered model.

        Does **not** instantiate clients; reads only the validated config
        schemas.

        Returns:
            A list of dicts, one per model, containing:
            ``name``, ``provider``, ``model_tag``, ``description``, ``tags``,
            and ``base_url``.

        Example:
            >>> for m in registry.list_model_details():
            ...     print(m["name"], "-", m["description"])
        """
        return [
            {
                "name": cfg.name,
                "provider": cfg.provider,
                "model_tag": cfg.model_tag,
                "base_url": cfg.base_url,
                "description": cfg.description,
                "tags": cfg.tags,
                "timeout": cfg.timeout,
                "temperature": cfg.temperature,
            }
            for cfg in self._configs.values()
        ]

    def get_config(self, name: str) -> ModelConfigSchema:
        """Return the validated :class:`ModelConfigSchema` for a named model.

        Useful for inspecting parameters without triggering client
        instantiation.

        Args:
            name: Registry name of the model.

        Returns:
            The :class:`ModelConfigSchema` instance for the requested model.

        Raises:
            ModelConfigurationError: If *name* is not registered.
        """
        if name not in self._configs:
            available = sorted(self._configs.keys())
            raise ModelConfigurationError(
                message=f"Config for '{name}' not found. Available: {available}",
                model_name=name,
                error_code="CFG_002",
            )
        return self._configs[name]

    def reload(self) -> None:
        """Re-scan the config directory and refresh the registry.

        Discards all cached client instances and config schemas, then
        reloads from disk.  Thread-safe.

        This is useful for hot-reloading configuration in long-running
        processes without a full restart.

        Raises:
            ModelConfigurationError: If any YAML file fails validation after
                reload.
        """
        with self._lock:
            logger.info(
                "ModelRegistry: reloading configs from %s", self.configs_dir
            )
            # Close existing clients that support it.
            for client in self._clients.values():
                if hasattr(client, "close"):
                    try:
                        client.close()  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
            self._clients.clear()
            self._configs.clear()
        self._load_all_configs()
        logger.info(
            "ModelRegistry: reload complete — %d models registered.",
            len(self._configs),
        )

    def is_registered(self, name: str) -> bool:
        """Check whether a model name exists in the registry.

        Args:
            name: Model registry name to check.

        Returns:
            ``True`` if the model is registered, ``False`` otherwise.
        """
        return name in self._configs or name == "auto-advanced"

    def __len__(self) -> int:
        """Return the number of registered model configurations.

        Returns:
            Integer count of loaded model configs.
        """
        return len(self._configs)

    def __repr__(self) -> str:
        return (
            f"ModelRegistry("
            f"configs_dir={str(self.configs_dir)!r}, "
            f"models={self.list_models()})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_configs(self) -> None:
        """Scan ``configs_dir`` for ``*.yaml`` files and load each one.

        Raises:
            ModelConfigurationError: If ``configs_dir`` does not exist or
                is not a directory, or if any YAML file fails validation.
        """
        if not self.configs_dir.exists():
            raise ModelConfigurationError(
                message=(
                    f"Model configs directory does not exist: {self.configs_dir}. "
                    f"Create it and add at least one *.yaml model config file."
                ),
                error_code="CFG_003",
                details={"path": str(self.configs_dir)},
            )

        if not self.configs_dir.is_dir():
            raise ModelConfigurationError(
                message=(
                    f"Model configs path is not a directory: {self.configs_dir}"
                ),
                error_code="CFG_003",
                details={"path": str(self.configs_dir)},
            )

        yaml_files: list[Path] = sorted(self.configs_dir.glob("*.yaml"))
        # Also match .yml
        yaml_files += sorted(self.configs_dir.glob("*.yml"))

        if not yaml_files:
            logger.warning(
                "ModelRegistry: no *.yaml files found in %s. "
                "The registry is empty.",
                self.configs_dir,
            )
            return

        loaded_count: int = 0
        error_count: int = 0

        for yaml_path in yaml_files:
            try:
                config = self._load_single_config(yaml_path)
            except ModelConfigurationError as exc:
                logger.error(
                    "ModelRegistry: skipping %s — %s",
                    yaml_path.name,
                    exc.message,
                )
                error_count += 1
                continue

            if config.name in self._configs:
                logger.warning(
                    "ModelRegistry: duplicate model name '%s' in %s — "
                    "overwriting previous entry.",
                    config.name,
                    yaml_path.name,
                )

            self._configs[config.name] = config
            loaded_count += 1
            logger.debug(
                "ModelRegistry: loaded '%s' (provider=%s, tag=%s) from %s",
                config.name,
                config.provider,
                config.model_tag,
                yaml_path.name,
            )

        logger.info(
            "ModelRegistry: loaded %d model config(s), %d error(s). "
            "Registered models: %s",
            loaded_count,
            error_count,
            self.list_models(),
        )

        if error_count > 0 and loaded_count == 0:
            raise ModelConfigurationError(
                message=(
                    f"All {error_count} YAML file(s) in {self.configs_dir} "
                    f"failed validation. No models registered."
                ),
                error_code="CFG_003",
            )

    def _load_single_config(self, yaml_path: Path) -> ModelConfigSchema:
        """Parse and validate one YAML config file.

        Args:
            yaml_path: Absolute path to the YAML file.

        Returns:
            A validated :class:`ModelConfigSchema` instance.

        Raises:
            ModelConfigurationError: If the file cannot be read, is invalid
                YAML, or fails Pydantic schema validation.
        """
        # --- File read ---
        try:
            raw_text: str = yaml_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ModelConfigurationError(
                message=f"Cannot read config file '{yaml_path.name}': {exc}",
                error_code="CFG_004",
                details={"path": str(yaml_path)},
            ) from exc

        # --- YAML parse ---
        try:
            data: Any = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ModelConfigurationError(
                message=f"YAML parse error in '{yaml_path.name}': {exc}",
                error_code="CFG_004",
                details={"path": str(yaml_path)},
            ) from exc

        if not isinstance(data, dict):
            raise ModelConfigurationError(
                message=(
                    f"Config file '{yaml_path.name}' must contain a YAML mapping "
                    f"(dict) at the top level, got {type(data).__name__}."
                ),
                error_code="CFG_004",
                details={"path": str(yaml_path)},
            )

        # --- Pydantic validation ---
        try:
            config = ModelConfigSchema.model_validate(data)
        except Exception as exc:  # pydantic.ValidationError
            raise ModelConfigurationError(
                message=(
                    f"Schema validation failed for '{yaml_path.name}': {exc}"
                ),
                error_code="CFG_004",
                details={
                    "path": str(yaml_path),
                    "validation_errors": str(exc),
                },
            ) from exc

        return config

    def _instantiate_client(self, config: ModelConfigSchema) -> BaseLLMClient:
        """Construct the appropriate :class:`BaseLLMClient` from a config.

        Args:
            config: A validated :class:`ModelConfigSchema`.

        Returns:
            A concrete :class:`BaseLLMClient` subclass instance.

        Raises:
            ModelConfigurationError: If the provider has no registered factory
                or client construction fails.
        """
        client_class: type[BaseLLMClient] | None = _PROVIDER_REGISTRY.get(
            config.provider
        )
        if client_class is None:
            # Should not reach here if _validate_provider_known ran, but guard
            # against dynamic modifications to _PROVIDER_REGISTRY.
            raise ModelConfigurationError(
                message=f"No factory registered for provider '{config.provider}'.",
                model_name=config.name,
                error_code="CFG_002",
            )

        # Build provider-specific constructor kwargs.
        # OllamaClient accepts all these parameters; future clients will
        # have their own mapping branches here.
        kwargs: dict[str, Any] = {
            "model_name": config.model_tag,
            "base_url": config.base_url,
            "timeout": config.timeout,
        }

        if config.provider == "ollama":
            kwargs.update(
                {
                    "default_temperature": config.temperature,
                    "default_max_tokens": config.max_tokens,
                    "keep_alive": config.keep_alive,
                }
            )

        try:
            client: BaseLLMClient = client_class(**kwargs)
        except Exception as exc:
            raise ModelConfigurationError(
                message=(
                    f"Failed to instantiate {client_class.__name__} "
                    f"for model '{config.name}': {exc}"
                ),
                model_name=config.name,
                error_code="CFG_002",
                details={"constructor_kwargs": list(kwargs.keys())},
            ) from exc

        return client


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "ModelConfigSchema",
    "ModelRegistry",
]
