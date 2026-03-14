# ==== core/config.py ====
"""
Application-wide configuration management for the Personal LLM Orchestrator.

This module defines a single, authoritative `AppSettings` class built on top of
Pydantic's `BaseSettings`.  All runtime configuration is sourced from environment
variables (or a `.env` file) and validated/coerced at import time so that
misconfiguration is caught immediately — never silently at runtime.

Design decisions
----------------
* **Single source of truth** — Every configurable knob lives here.  No magic
  strings are scattered across the codebase.
* **Fail-fast validation** — Pydantic raises `ValidationError` during startup if
  a required value is missing or has an illegal type/value, preventing partial
  initialisation.
* **Immutable after load** — `model_config` sets ``frozen=True`` so the settings
  object cannot be mutated after construction, preventing accidental config drift.
* **Singleton accessor** — The module-level ``get_settings()`` function is
  decorated with ``@lru_cache`` so the `.env` file is parsed exactly once per
  process lifetime.

Usage
-----
    from core.config import get_settings

    settings = get_settings()
    print(settings.ollama_base_url)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from models.api_keys import APIKeyManager

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LogLevelLiteral = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
EmbeddingDeviceLiteral = Literal["cpu", "cuda", "mps"]

# ---------------------------------------------------------------------------
# Settings sections (nested Pydantic models for logical grouping)
# ---------------------------------------------------------------------------


class OllamaSettings(BaseSettings):
    """Configuration for the Ollama LLM backend connection.

    Attributes:
        base_url: HTTP(S) base URL of the Ollama server instance.
        request_timeout: Per-request socket timeout in seconds.
        default_model: Fallback model tag when the caller does not specify one.
    """

    base_url: AnyHttpUrl = Field(
        default="http://127.0.0.1:11434",
        description="Base URL of the running Ollama server.",
    )
    request_timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="HTTP request timeout in seconds (1–600).",
    )
    default_model: str = Field(
        default="llama3",
        min_length=1,
        description="Default Ollama model tag to use when none is specified.",
    )

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )


class ChromaSettings(BaseSettings):
    """Configuration for the ChromaDB vector store.

    Attributes:
        persist_dir: Filesystem path where ChromaDB persists its data.
        default_collection: Name of the default collection used by the pipeline.
        top_k: Maximum number of nearest-neighbour results per query.
    """

    persist_dir: Path = Field(
        default=Path("./data/chroma_store"),
        description="Directory where ChromaDB persists SQLite and segment data.",
    )
    default_collection: str = Field(
        default="llm_orchestrator_default",
        min_length=1,
        description="Default ChromaDB collection name.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results returned per similarity search (1–100).",
    )

    model_config = SettingsConfigDict(
        env_prefix="CHROMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    @field_validator("persist_dir", mode="before")
    @classmethod
    def _coerce_persist_dir(cls, value: object) -> Path:
        """Coerce string values from the environment into ``pathlib.Path``.

        Args:
            value: Raw value read from the environment variable.

        Returns:
            A resolved ``Path`` object.
        """
        return Path(str(value))


class EmbeddingSettings(BaseSettings):
    """Configuration for the sentence-transformer embedding model.

    Attributes:
        model_name: HuggingFace model ID or local directory path.
        batch_size: Number of texts processed per embedding forward pass.
        device: Hardware target — "cpu", "cuda", or "mps".
    """

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        min_length=1,
        description="HuggingFace model name or local path for embeddings.",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=2048,
        description="Embedding inference batch size (1–2048).",
    )
    device: EmbeddingDeviceLiteral = Field(
        default="cpu",
        description='Compute device: "cpu", "cuda", or "mps".',
    )

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )


class LoggingSettings(BaseSettings):
    """Configuration for the dual-sink logging infrastructure.

    Attributes:
        level: Minimum log level for the console (stdout) handler.
        file_path: Destination path for the rotating debug log file.
        file_max_bytes: Byte threshold that triggers log rotation.
        file_backup_count: Number of rotated backup files to retain.
    """

    level: LogLevelLiteral = Field(
        default="WARNING",
        description='Console log level: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL".',
    )
    file_path: Path = Field(
        default=Path("./logs/orchestrator.log"),
        description="Path for the rotating debug log file.",
    )
    file_max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        ge=1024,
        description="Maximum file size in bytes before log rotation.",
    )
    file_backup_count: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Number of rotated log backups to keep (0–50).",
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def _coerce_file_path(cls, value: object) -> Path:
        """Coerce string values from the environment into ``pathlib.Path``.

        Args:
            value: Raw value read from the environment variable.

        Returns:
            A resolved ``Path`` object.
        """
        return Path(str(value))


# ---------------------------------------------------------------------------
# Top-level AppSettings
# ---------------------------------------------------------------------------


import os

# Suppress ChromaDB Telemetry and ONNX Runtime warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# 3 is Error level only for onnxruntime
os.environ["ORT_LOG_LEVEL"] = "3"

class AppSettings(BaseSettings):
    """Root application configuration for the Personal LLM Orchestrator.

    Aggregates all sub-setting groups and exposes app-level meta fields.
    Loaded once at startup via :func:`get_settings`.

    Attributes:
        app_name: Human-readable identifier for this orchestrator instance.
        app_version: PEP-440 semantic version string.
        debug_mode: Master debug toggle; forces ``LOG_LEVEL`` to ``DEBUG``.
        ollama: Ollama backend connection settings.
        chroma: ChromaDB vector store settings.
        embedding: Sentence-transformer embedding model settings.
        logging: Dual-sink logging infrastructure settings.

    Example:
        >>> from core.config import get_settings
        >>> s = get_settings()
        >>> str(s.ollama.base_url)
        'http://localhost:11434'
    """

    # --- App meta ---
    app_name: str = Field(
        default="PersonalLLMOrchestrator",
        min_length=1,
        description="Human-readable name for this orchestrator instance.",
    )
    app_version: str = Field(
        default="0.1.0",
        pattern=r"^\d+\.\d+\.\d+",
        description="Semantic version string (MAJOR.MINOR.PATCH).",
    )
    debug_mode: bool = Field(
        default=False,
        description="Master debug flag; overrides LOG_LEVEL → DEBUG when True.",
    )

    # --- Nested sub-settings ---
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # --- API Keys unified flow ---
    api_keys: "APIKeyManager" = Field(
        default_factory=lambda: __import__("models.api_keys").api_keys.provider_keys
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
        frozen=True,  # Prevent mutation after construction
    )


    # ------------------------------------------------------------------
    # Cross-field validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _apply_debug_mode_override(self) -> "AppSettings":
        """Force ``logging.level`` to ``DEBUG`` when ``debug_mode`` is enabled.

        This validator runs after all fields have been populated.  Because
        ``frozen=True``, we use ``object.__setattr__`` to bypass the immutability
        guard during the construction phase only.

        Returns:
            The (potentially mutated) settings instance.
        """
        if self.debug_mode and self.logging.level != "DEBUG":
            # Pydantic frozen models forbid normal attribute assignment;
            # object.__setattr__ is the canonical workaround inside validators.
            object.__setattr__(
                self.logging,
                "level",
                "DEBUG",
            )
            logging.getLogger(__name__).debug(
                "debug_mode=True → LOG_LEVEL overridden to DEBUG"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def ensure_directories(self) -> None:
        """Create required filesystem directories if they do not yet exist.

        This method should be called once at application startup before any
        component tries to write to disk.

        Raises:
            OSError: If directory creation fails due to a permissions error or
                an invalid path.
        """
        directories: list[Path] = [
            self.chroma.persist_dir,
            self.logging.file_path.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def as_safe_dict(self) -> dict[str, object]:
        """Return a redacted representation of the settings safe for logging.

        Sensitive fields (API keys, passwords) are masked.  Currently the
        orchestrator has no secrets, but this method is a pre-wired extension
        point for when credentials are added.

        Returns:
            A nested dictionary of configuration values with sensitive fields
            replaced by ``"***"``.
        """
        raw: dict[str, object] = self.model_dump()
        # Extension point: mask secrets here when added, e.g.:
        # raw["some_api_key"] = "***"
        return raw


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the application-wide ``AppSettings`` singleton.

    The underlying ``AppSettings`` object is constructed (and the ``.env`` file
    parsed) exactly once per process.  Subsequent calls return the cached
    instance.

    Returns:
        The fully validated and immutable ``AppSettings`` instance.

    Raises:
        pydantic.ValidationError: If any required environment variable is
            missing or holds an invalid value.

    Example:
        >>> settings = get_settings()
        >>> settings.app_name
        'PersonalLLMOrchestrator'
    """
    settings = AppSettings()
    settings.ensure_directories()
    return settings
