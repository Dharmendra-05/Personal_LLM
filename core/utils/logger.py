# ==== core/utils/logger.py ====
"""
Logging infrastructure for the Personal LLM Orchestrator.

Architecture
------------
This module implements a **dual-sink** logging strategy:

* **Console handler** — Streams ``INFO`` and above to ``sys.stderr``.
  Uses a concise, human-friendly format ideal for interactive terminal
  sessions and container log aggregators (e.g. Docker / Kubernetes stdout).

* **Rotating file handler** — Writes ``DEBUG`` and above to a configurable
  log file.  The file rotates when it reaches ``LOG_FILE_MAX_BYTES`` and
  retains ``LOG_FILE_BACKUP_COUNT`` archives, preventing unbounded disk use.

Design decisions
----------------
* **Named loggers** — Every module obtains its logger via
  ``get_logger(__name__)``  rather than using the root logger.  This isolates
  per-module verbosity and avoids polluting third-party library log output.
* **Idempotent setup** — ``setup_logging()`` is guarded by a module-level
  flag so re-importing or calling it twice does not duplicate handlers.
* **No root-logger mutation** — We configure only the ``"orchestrator"``
  hierarchy, leaving the root logger (and therefore ``httpx``, ``chromadb``,
  etc.) at their default levels.
* **Thread safety** — ``logging.handlers.RotatingFileHandler`` uses an
  internal ``threading.Lock``; this module adds no additional locking beyond
  what the stdlib provides.

Usage
-----
Recommended pattern for every module in the codebase:

.. code-block:: python

    from core.utils.logger import get_logger

    logger = get_logger(__name__)

    def my_function() -> None:
        logger.debug("Entering my_function")
        logger.info("my_function completed successfully")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Root name for the entire orchestrator logger hierarchy.
#: All module loggers are children: e.g. ``orchestrator.core.config``.
LOGGER_ROOT_NAME: Final[str] = "orchestrator"

#: Format string used by the console (INFO) handler.
#: Deliberately compact — timestamp, level, name, message.
CONSOLE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

#: Format string used by the rotating file (DEBUG) handler.
#: More verbose — includes module path, line number, and thread name.
FILE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(module)s:%(lineno)d | %(threadName)s | %(message)s"
)

#: ISO-8601-like timestamp format shared by both handlers.
DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

#: Guard flag — True once ``setup_logging`` has been called successfully.
_logging_configured: bool = False


# ---------------------------------------------------------------------------
# Core setup function
# ---------------------------------------------------------------------------


def setup_logging(
    *,
    console_level: int | str = logging.INFO,
    file_level: int | str = logging.DEBUG,
    log_file_path: Path | str = Path("./logs/orchestrator.log"),
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure the orchestrator logging hierarchy.

    This function should be called **once** at application startup, typically
    from ``main.py`` or the application factory.  Subsequent calls are
    no-ops (idempotent).

    The function:

    1. Resolves and creates the log file's parent directory if necessary.
    2. Attaches a :class:`logging.StreamHandler` to ``sys.stderr`` at
       *console_level*.
    3. Attaches a :class:`logging.handlers.RotatingFileHandler` at
       *file_level*.
    4. Sets the root ``"orchestrator"`` logger to ``DEBUG`` (the lowest of
       the two handler thresholds) so that neither handler inadvertently
       filters records before they reach the handler's own level check.

    Args:
        console_level: Minimum log level for the console handler.  Accepts
            either an integer constant (e.g. ``logging.INFO``) or a string
            (e.g. ``"INFO"``).  Defaults to ``logging.INFO``.
        file_level: Minimum log level for the rotating file handler.
            Defaults to ``logging.DEBUG``.
        log_file_path: Destination path for the rotating log file.  The
            parent directory is created automatically.  Defaults to
            ``./logs/orchestrator.log``.
        max_bytes: Maximum size (in bytes) of the active log file before it
            is rotated.  Defaults to 10 MB.
        backup_count: Number of rotated backup files to retain.  Defaults to 5.

    Raises:
        OSError: If the log directory cannot be created (e.g. permission
            denied).
        ValueError: If an invalid log level string is supplied.

    Example:
        >>> from core.utils.logger import setup_logging
        >>> setup_logging(console_level="WARNING", log_file_path="/tmp/app.log")
    """
    global _logging_configured  # noqa: PLW0603

    if _logging_configured:
        return

    # ------------------------------------------------------------------
    # 1. Resolve date-based log file path
    # ------------------------------------------------------------------
    # If the provided path is a directory or the default, we use date-based names
    base_log_dir = Path(log_file_path).parent if not Path(log_file_path).is_dir() else Path(log_file_path)
    sessions_dir = base_log_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    resolved_path = sessions_dir / f"{today}.log"

    # ------------------------------------------------------------------
    # 2. Normalise level arguments to integers
    # ------------------------------------------------------------------
    console_level_int: int = _resolve_level(console_level)
    file_level_int: int = _resolve_level(file_level)

    # ------------------------------------------------------------------
    # 3. Build shared formatters
    # ------------------------------------------------------------------
    console_formatter = logging.Formatter(
        fmt=CONSOLE_FORMAT,
        datefmt=DATE_FORMAT,
    )
    file_formatter = logging.Formatter(
        fmt=FILE_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # ------------------------------------------------------------------
    # 4. Console handler (stderr, WARNING+ by default for clean CLI)
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(console_level_int)
    console_handler.setFormatter(console_formatter)
    console_handler.set_name("orchestrator.console")

    # ------------------------------------------------------------------
    # 5. Rotating file handler (debug log file, DEBUG+)
    # ------------------------------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(resolved_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=False,  # Open the file immediately on handler construction
    )
    file_handler.setLevel(file_level_int)
    file_handler.setFormatter(file_formatter)
    file_handler.set_name("orchestrator.file")

    # ------------------------------------------------------------------
    # 6. Configure the root orchestrator logger
    # ------------------------------------------------------------------
    root_logger: logging.Logger = logging.getLogger(LOGGER_ROOT_NAME)

    # Set the logger to the *lowest* threshold so handlers can filter
    # independently.  Without this, the logger itself would silently
    # discard records before they reach either handler.
    effective_root_level: int = min(console_level_int, file_level_int)
    root_logger.setLevel(effective_root_level)

    # Avoid duplicate handlers on repeated imports in interactive shells
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Prevent log records from bubbling up to the root (``""`` logger),
    # which avoids duplicate output if the root logger has its own handlers.
    root_logger.propagate = False

    _logging_configured = True

    # Capture python warnings into the logging system
    logging.captureWarnings(True)
    py_warnings_logger = logging.getLogger("py.warnings")
    py_warnings_logger.setLevel(logging.WARNING)
    # They will now propagate to our 'orchestrator' handlers if we don't stop them,
    # or we can attach the file handler directly to them.
    py_warnings_logger.addHandler(file_handler)
    py_warnings_logger.propagate = False

    # Emit a session startup banner to the file log
    root_logger.info("")
    root_logger.info("=" * 80)
    root_logger.info(f" SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    root_logger.info("=" * 80)
    root_logger.debug(
        "Logging initialised. "
        "console_level=%s | file_level=%s | log_file=%s | "
        "max_bytes=%d | backup_count=%d",
        logging.getLevelName(console_level_int),
        logging.getLevelName(file_level_int),
        resolved_path,
        max_bytes,
        backup_count,
    )


# ---------------------------------------------------------------------------
# Public accessor — used by every module
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that is a child of the orchestrator hierarchy.

    All loggers returned by this function sit under the ``"orchestrator"``
    root logger, inheriting its handlers and effective level.

    Args:
        name: Typically ``__name__`` of the calling module.  If the name
            does not already start with ``"orchestrator."``, the prefix is
            prepended automatically so the logger is always part of the
            correct hierarchy.

    Returns:
        A :class:`logging.Logger` instance scoped to the orchestrator
        hierarchy.

    Raises:
        TypeError: If *name* is not a non-empty string.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialised.")
    """
    if not isinstance(name, str) or not name.strip():
        raise TypeError(
            f"Logger name must be a non-empty string, got {name!r} ({type(name).__name__})."
        )

    # Namespace the logger under the orchestrator hierarchy.
    if not name.startswith(f"{LOGGER_ROOT_NAME}.") and name != LOGGER_ROOT_NAME:
        qualified_name = f"{LOGGER_ROOT_NAME}.{name}"
    else:
        qualified_name = name

    return logging.getLogger(qualified_name)


# ---------------------------------------------------------------------------
# Helper to configure logging from AppSettings (convenience wrapper)
# ---------------------------------------------------------------------------


def configure_from_settings() -> None:
    """Bootstrap logging using values from ``core.config.AppSettings``.

    This is a convenience wrapper that reads the validated ``AppSettings``
    singleton and calls :func:`setup_logging` with the appropriate
    parameters.  Import-order safe: ``core.config`` is only imported when
    this function is called, not at module import time.

    Raises:
        pydantic.ValidationError: If ``AppSettings`` fails validation.
        OSError: If the log directory cannot be created.

    Example:
        >>> from core.utils.logger import configure_from_settings
        >>> configure_from_settings()
    """
    # Deferred import to avoid circular dependency at module load time.
    from core.config import get_settings  # noqa: PLC0415

    settings = get_settings()

    setup_logging(
        console_level=settings.logging.level,
        file_level="DEBUG",
        log_file_path=settings.logging.file_path,
        max_bytes=settings.logging.file_max_bytes,
        backup_count=settings.logging.file_backup_count,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_level(level: int | str) -> int:
    """Normalise a log level to its integer representation.

    Args:
        level: An integer level constant (e.g. ``logging.DEBUG``) or a
            case-insensitive string (e.g. ``"debug"``).

    Returns:
        The integer log level.

    Raises:
        ValueError: If *level* is a string that does not map to a known
            log level.
        TypeError: If *level* is neither an ``int`` nor a ``str``.
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        numeric: int = logging.getLevelName(level.upper())
        if not isinstance(numeric, int):
            raise ValueError(
                f"Unknown log level string: {level!r}.  "
                f"Valid options are: DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )
        return numeric
    raise TypeError(
        f"Log level must be an int or str, got {type(level).__name__!r}."
    )


# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "LOGGER_ROOT_NAME",
    "setup_logging",
    "get_logger",
    "configure_from_settings",
]
