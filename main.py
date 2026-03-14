#!/usr/bin/env python3
# ==== main.py ====
"""
Personal LLM Orchestrator — CLI Entry Point
============================================
This module is the **only** file that interacts with the terminal.
All business logic, model dispatch, and RAG retrieval live in
:class:`~core.orchestrator.SystemOrchestrator`.  This module handles:

* Bootstrapping: logging, settings, ASCII banner.
* The REPL loop: read → dispatch → print.
* Built-in slash commands (``/help``, ``/models``, ``/reload``, etc.).
* Graceful shutdown on ``KeyboardInterrupt`` and ``EOF``.

Architecture principle
----------------------
Terminal I/O is centralised here and **nowhere else**.  The orchestrator
returns strings; ``main.py`` prints them.  This keeps the core testable
without a terminal.

Usage
-----
.. code-block:: bash

    # From the project root:
    python main.py

    # With a specific model override:
    python main.py --model local-gemma

    # Debug mode (verbose logging to console):
    python main.py --debug

    # Point at a custom data directory:
    python main.py --data-dir /path/to/documents
"""

from __future__ import annotations

import os
import sys
import warnings

# --- Early Environment Suppressions & CLI Optimization ---
# Suppress Pydantic and other library warnings from CLI, but they still go to logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
# Divert ChromaDB telemetry to a no-op implementation
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ORT_LOG_LEVEL"] = "3"
os.environ["CHROMA_TELEMETRY_IMPL"] = "chromadb.telemetry.product.NoopTelemetry"

# Robust patch for ChromaDB telemetry to prevent background thread startup and noise
try:
    import chromadb.telemetry.product
    import chromadb.telemetry.posthog
    import chromadb.telemetry.segment

    def _noop(*args, **kwargs):
        return None

    for mod in [chromadb.telemetry.product, chromadb.telemetry.posthog, chromadb.telemetry.segment]:
        for cls_name in ["Posthog", "AnonymousTelemetry", "Sentry"]:
            if hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                cls.capture = _noop
                cls.send = _noop
                cls.__init__ = _noop
except Exception:
    pass

# ---------------------------------------------------------------------------

import argparse
import shutil
import textwrap
import time
from pathlib import Path
from typing import Final, TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import AppSettings

# ---------------------------------------------------------------------------
# Project root on sys.path (allows running `python main.py` from any CWD)
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Bootstrap logging before any other project imports
# ---------------------------------------------------------------------------
from core.utils.logger import configure_from_settings, get_logger, setup_logging  # noqa: E402

# Minimal pre-settings console logging so startup errors are visible
setup_logging(console_level="WARNING", log_file_path=_PROJECT_ROOT / "logs" / "orchestrator.log")
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Now import project modules (logging is ready)
# ---------------------------------------------------------------------------
from core.exceptions import PersonalLLMException  # noqa: E402
from core.orchestrator import OrchestratorResponse, SystemOrchestrator  # noqa: E402
from core.router import RouteMode  # noqa: E402

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

# Detect true terminal width for wrapping
_TERM_WIDTH: Final[int] = shutil.get_terminal_size(fallback=(100, 24)).columns
_WRAP_WIDTH: Final[int] = min(_TERM_WIDTH - 4, 100)

# ANSI colour codes (gracefully disabled when not in a TTY)
_IS_TTY: Final[bool] = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI colour code if stdout is a TTY."""
    if not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _green(t: str) -> str:
    return _c("32", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _red(t: str) -> str:
    return _c("31", t)


def _magenta(t: str) -> str:
    return _c("35", t)


def _purple_gradient(text: str) -> str:
    """Apply a true-color purple/magenta gradient to a block of text."""
    if not _IS_TTY:
        return text
    lines = text.strip("\n").split("\n")
    if not lines:
        return text

    result = []
    for i, line in enumerate(lines):
        ratio = i / max(1, len(lines) - 1)
        r = int(255 - (170 * ratio))
        g = 0
        b = 255
        result.append(f"\033[38;2;{r};{g};{b}m{line}\033[0m")

    return "\n" + "\n".join(result) + "\n"

# ---------------------------------------------------------------------------
# ASCII Banner
# ---------------------------------------------------------------------------

_BANNER: Final[str] = r"""
   ███████╗██╗   ██╗███████╗██╗  ██╗   ██╗███╗   ██╗███╗   ██╗
   ██╔════╝██║   ██║██╔════╝██║  ╚██╗ ██╔╝████╗  ██║████╗  ██║
   █████╗  ██║   ██║█████╗  ██║   ╚████╔╝ ██╔██╗ ██║██╔██╗ ██║
   ██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║    ╚██╔╝  ██║╚██╗██║██║╚██╗██║
   ███████╗ ╚████╔╝ ███████╗███████╗██║   ██║ ╚████║██║ ╚████║
   ╚══════╝  ╚═══╝  ╚══════╝╚══════╝╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═══╝
"""

_TAGLINE: Final[str] = (
    "  Lifelong Multi-Modal Agent · Autonomous Evelynn · System Operations"
)
_VERSION: Final[str] = "v1.1.0"

# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

_COMMANDS: Final[list[tuple[str, str, str]]] = [
    ("/help",    "",           "Show this help message"),
    ("/models",  "",           "List all registered LLM models"),
    ("/reload",  "",           "Re-index documents from the data directory"),
    ("/stats",   "",           "Show vector store collection statistics"),
    ("/history", "",           "Display recent conversation history"),
    ("/clear",   "",           "Clear conversation history"),
    ("/health",  "",           "Check subsystem connectivity"),
    ("/route",   "<query>",    "Show routing decision for a query (dry run)"),
    ("/model",   "<n>",        "Switch default model for this session"),
    ("/debug",   "<query>",    "Process query and show full metadata"),
    ("/exit",    "",           "Exit the orchestrator"),
    ("/quit",    "",           "Exit the orchestrator"),
]


# ---------------------------------------------------------------------------
# Printer helpers (all terminal output goes through these)
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    """Print the ASCII banner and version info."""
    print(_purple_gradient(_BANNER), end="")
    print(_bold(_magenta(_TAGLINE)))
    width = min(_TERM_WIDTH, 80)
    print(_dim("  " + "─" * (width - 2)))
    print(f"  {_bold('Version:')} {_VERSION}   "
          f"{_bold('Python:')} {sys.version.split()[0]}")
    print(_dim("  " + "─" * (width - 2)))
    print()


def _print_help() -> None:
    """Print the slash command reference table."""
    print()
    print(_bold(_magenta("  ── Available Commands ───────────────────────────────────")))
    for cmd, arg, desc in _COMMANDS:
        cmd_fmt = _yellow(f"{cmd:<10}")
        arg_fmt = _dim(f"{arg:<14}") if arg else " " * 14
        print(f"    {cmd_fmt} {arg_fmt}  {desc}")
    print()
    print(_dim("  Anything else is sent to the LLM."))
    print(_dim("  Press Ctrl+C or type /exit to quit."))
    print()


def _print_response(resp: OrchestratorResponse, show_metadata: bool = False) -> None:
    """Render an :class:`~core.orchestrator.OrchestratorResponse` to the terminal.

    Args:
        resp: The orchestrator response to display.
        show_metadata: If ``True``, print timing, token counts, and routing
            info below the response text.
    """
    if not resp.success:
        print()
        print(_red(f"  ✖  Error: {resp.error}"))
        print()
        return

    # Mode badge
    mode_colours: dict[RouteMode, str] = {
        RouteMode.PERSONAL_MEMORY: _cyan("[PERSONAL_MEMORY]"),
        RouteMode.ADVANCED_KNOWLEDGE: _yellow("[ADVANCED_KNOWLEDGE]"),
    }
    badge = mode_colours.get(resp.route_mode, f"[{resp.route_mode.name}]")

    print()
    print(f"  {badge}  {_bold(_magenta('Evelynn:'))}")
    print()

    # Word-wrap the response text
    for paragraph in resp.text.split("\n"):
        if paragraph.strip():
            wrapped = textwrap.fill(
                paragraph,
                width=_WRAP_WIDTH,
                initial_indent="    ",
                subsequent_indent="    ",
            )
            print(wrapped)
        else:
            print()

    if show_metadata:
        print()
        print(_dim("  ── Metadata " + "─" * 40))
        print(_dim(f"  Model      : {resp.model_name}"))
        print(_dim(f"  Route      : {resp.route_mode.name} "
                   f"(confidence={resp.decision.confidence:.2f})"))
        print(_dim(f"  RAG chunks : {resp.rag_context_used}"))
        dur = f"{resp.duration_seconds:.2f}s"
        tok = ""
        if resp.prompt_tokens and resp.completion_tokens:
            tok = (f" | tokens: {resp.prompt_tokens}→{resp.completion_tokens} "
                   f"({resp.completion_tokens / resp.duration_seconds:.0f} tok/s)"
                   if resp.duration_seconds > 0 else "")
        print(_dim(f"  Duration   : {dur}{tok}"))
        sigs = ", ".join(resp.decision.matched_signals[:5]) or "none"
        print(_dim(f"  Signals    : {sigs}"))
        print()


def _print_section(title: str) -> None:
    print()
    print(_bold(_cyan(f"  ── {title} " + "─" * max(0, 46 - len(title)))))


def _print_error(msg: str) -> None:
    print(_red(f"\n  ✖  {msg}\n"))


def _print_ok(msg: str) -> None:
    print(_green(f"\n  ✔  {msg}\n"))


def _print_info(msg: str) -> None:
    print(_dim(f"  ℹ  {msg}"))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _handle_models(orch: SystemOrchestrator) -> None:
    """List all registered models."""
    _print_section("Registered Models")
    models = orch.list_models()
    if not models:
        _print_error("No models registered. Add YAML configs to models/model_configs/.")
        return
    for m in models:
        tag = m.get("model_tag", "?")
        provider = m.get("provider", "?")
        desc = m.get("description", "")
        temp = m.get("temperature", "?")
        print(f"    {_bold(_yellow(m['name']))}")
        print(f"      {_dim('Provider:')} {provider}  "
              f"{_dim('Tag:')} {tag}  "
              f"{_dim('Temp:')} {temp}")
        if desc:
            print(f"      {_dim('Desc:')} {desc[:80]}")
        print()


def _handle_reload(orch: SystemOrchestrator) -> None:
    """Re-index the knowledge base."""
    _print_section("Re-indexing Knowledge Base")
    _print_info("Scanning documents and updating vector store …")
    try:
        start = time.perf_counter()
        result = orch.reload_knowledge_base(confirm_reindex=True)
        elapsed = time.perf_counter() - start
        print()
        print(f"    {_bold('Files found  :')} {result['files_found']}")
        print(f"    {_bold('Chunks found :')} {result['chunks_found']}")
        print(f"    {_bold('Added        :')} {_green(str(result['added']))}")
        print(f"    {_bold('Skipped      :')} {_dim(str(result['skipped']))}")
        if result["failed"]:
            print(f"    {_bold('Failed       :')} {_red(str(result['failed']))}")
        print(f"    {_bold('Duration     :')} {elapsed:.2f}s")
        print()
    except PersonalLLMException as exc:
        _print_error(f"Reload failed: {exc.message}")


def _handle_stats(orch: SystemOrchestrator) -> None:
    """Display vector store statistics."""
    _print_section("Vector Store Stats")
    stats = orch.get_collection_stats()
    print(f"    {_bold('Collection :')} {stats['collection_name']}")
    print(f"    {_bold('Documents  :')} {stats['document_count']}")
    print()


def _handle_history(orch: SystemOrchestrator) -> None:
    """Display conversation history."""
    _print_section("Conversation History")
    history = orch.get_history()
    if not history:
        _print_info("No conversation history yet.")
        return
    for turn in history:
        role_fmt = _bold(_cyan("[You]")) if turn["role"] == "user" else _bold("[AI]")
        mode_tag = _dim(f"({turn['mode']})")
        content = turn["content"]
        print(f"    {role_fmt} {mode_tag}")
        for line in content.split("\n"):
            print(f"      {line}")
        print()


def _handle_clear(orch: SystemOrchestrator) -> None:
    """Clear conversation history."""
    orch.clear_history()
    _print_ok("Conversation history cleared.")


def _handle_health(orch: SystemOrchestrator) -> None:
    """Check subsystem health."""
    _print_section("Health Check")
    statuses = orch.health_check()
    for name, ok in statuses.items():
        icon = _green("✔") if ok else _red("✖")
        status_text = _green("OK") if ok else _red("UNREACHABLE")
        print(f"    {icon}  {name:<20} {status_text}")
    print()
    if not statuses.get("default_model", False):
        _print_info("Tip: make sure `ollama serve` is running.")


def _handle_route(orch: SystemOrchestrator, args_str: str) -> None:
    """Show routing decision for a dry-run query."""
    query = args_str.strip()
    if not query:
        _print_error("Usage: /route <query>")
        return
    _print_section("Routing Decision (dry run)")
    # Re-use the orchestrator's own router so results are consistent
    explanation = orch._router.explain(query)
    for line in explanation.split("\n"):
        print(f"    {line}")
    print()


def _handle_model_switch(
    orch: SystemOrchestrator,
    args_str: str,
    session_model: list[str | None],
) -> None:
    """Switch the default model for this session."""
    name = args_str.strip()
    if not name:
        current = session_model[0] or orch._default_model_name or "(auto)"
        _print_info(f"Current model: {_bold(current)}")
        _print_info("Usage: /model <n>   (see /models for available names)")
        return
    if not orch._registry.is_registered(name):
        _print_error(f"Model '{name}' not found. Use /models to list available models.")
        return
    orch._default_model_name = name
    session_model[0] = name
    _print_ok(f"Default model switched to '{_bold(name)}'.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_arg_parser(settings: "AppSettings") -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="personal-llm-orchestrator",
        description="Personal LLM Orchestrator — local AI with RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --model local-gemma
              python main.py --debug --data-dir ./my_docs
        """),
    )
    parser.add_argument(
        "--model", "-m",
        default=settings.ollama.default_model,
        metavar="NAME",
        help=f"Default model registry name to use (default: {settings.ollama.default_model}).",
    )
    parser.add_argument(
        "--data-dir",
        default="assets/",
        metavar="PATH",
        help="Path to the personal assets/ documents directory (default: assets/).",
    )
    parser.add_argument(
        "--chroma-dir",
        default=str(settings.chroma.persist_dir),
        metavar="PATH",
        help=f"Path for ChromaDB persistence (default: {settings.chroma.persist_dir}).",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding.model_name,
        metavar="MODEL",
        help=f"Sentence-transformer embedding model name (default: {settings.embedding.model_name}).",
    )
    parser.add_argument(
        "--device",
        default=settings.embedding.device,
        choices=["cpu", "cuda", "mps"],
        help=f"Compute device for embeddings (default: {settings.embedding.device}).",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=settings.chroma.top_k,
        metavar="K",
        help=f"Number of RAG context chunks to retrieve per query (default: {settings.chroma.top_k}).",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=5,
        metavar="TURNS",
        help="Number of conversation turns to keep in memory (default: 5).",
    )
    parser.add_argument(
        "--no-auto-index",
        action="store_true",
        help="Skip automatic indexing on first launch.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output (overrides LOG_LEVEL).",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Always print routing/timing metadata after each response.",
    )
    return parser


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def _run_repl(
    orch: SystemOrchestrator,
    show_metadata: bool = False,
) -> None:
    """Run the Read-Eval-Print Loop until the user exits.

    Args:
        orch: Fully initialised :class:`~core.orchestrator.SystemOrchestrator`.
        show_metadata: If ``True``, print routing/timing metadata after each
            LLM response.
    """
    # Mutable cell for per-session model override (list so inner functions can write)
    session_model: list[str | None] = [None]

    print(_dim(f"  Type your message and press Enter.  "
               f"Use {_yellow('/help')} for commands, "
               f"{_yellow('Ctrl+C')} to quit."))
    print()

    while True:
        # ── Prompt ──────────────────────────────────────────────────────
        try:
            raw = input(_bold(_cyan("  You › "))).strip()
        except EOFError:
            print()
            _print_info("EOF detected — exiting.")
            break
        except KeyboardInterrupt:
            print()
            _handle_exit()
            break

        if not raw:
            continue

        # ── Slash command dispatch ───────────────────────────────────────
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            args_str = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                _handle_exit()
                break
            elif cmd == "/help":
                _print_help()
            elif cmd == "/models":
                _handle_models(orch)
            elif cmd == "/reload":
                _handle_reload(orch)
            elif cmd == "/stats":
                _handle_stats(orch)
            elif cmd == "/history":
                _handle_history(orch)
            elif cmd == "/clear":
                _handle_clear(orch)
            elif cmd == "/health":
                _handle_health(orch)
            elif cmd == "/route":
                _handle_route(orch, args_str)
            elif cmd == "/model":
                _handle_model_switch(orch, args_str, session_model)
            elif cmd == "/debug":
                query = args_str.strip()
                if not query:
                    _print_error("Usage: /debug <query>")
                else:
                    resp = orch.process_query(
                        query,
                        model_name=session_model[0],
                    )
                    _print_response(resp, show_metadata=True)
            else:
                _print_error(f"Unknown command '{cmd}'. Type /help for commands.")
            continue

        # ── Regular query → orchestrator ─────────────────────────────────
        # Route ONCE here — the decision is passed directly into process_query
        # so the router is not called a second time inside the orchestrator.
        decision = orch.route_query(raw)

        if decision.mode != RouteMode.GENERAL_CHAT:
            _print_info("Thinking …")

        resp = orch.process_query(
            raw,
            model_name=session_model[0],
            decision=decision,          # ← pre-computed, no double-route
        )
        _print_response(resp, show_metadata=show_metadata)


def _handle_exit() -> None:
    """Print a farewell message and exit cleanly."""
    print()
    print(_magenta("  ╔══════════════════════════════════════╗"))
    print(_magenta("  ║") + f"   Evelynn is shutting down...        " + _magenta("║"))
    print(_magenta("  ║") + "   Goodbye!  🤖                         " + _magenta("║"))
    print(_magenta("  ╚══════════════════════════════════════╝"))
    print()
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# Startup sequence
# ---------------------------------------------------------------------------


def _print_startup_status(orch: SystemOrchestrator) -> None:
    """Print a brief status block after the orchestrator initialises."""
    models = orch.list_models()
    stats = orch.get_collection_stats()
    health = orch.health_check()

    def _status_icon(ok: bool) -> str:
        return _green("✔") if ok else _red("✖")

    print(_bold("  System Status"))
    print(_dim("  " + "─" * 46))

    default_model_name = orch._default_model_name or "(none)"
    if default_model_name != "(none)":
        try:
            client = orch._registry.get_model(default_model_name)
            url = getattr(client, "base_url", "N/A")
        except Exception:
            url = "Error"
    else:
        url = "N/A"

    print(f"  {_status_icon(health.get('registry', False))}  "
          f"Model Registry    {len(models)} model(s)")

    vs_health = health.get('vector_store', False)
    count = stats.get('document_count', -1)
    count_str = f"{count} chunks" if count >= 0 else _dim("Deferred")
    print(f"  {_status_icon(vs_health)}  "
          f"Vector Store      {count_str}")

    dm_ok = health.get('default_model', False)
    print(f"  {_status_icon(dm_ok)}  "
          f"Default Model     {_bold(default_model_name)} @ {url}")

    if not dm_ok:
        print()
        print(_yellow("  ⚠  Ollama is not reachable at ") + _bold(url))
        print(_dim("     Ensure `ollama serve` is running. Type /health to retry connection."))

    print(_dim("  " + "─" * 46))
    print()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> int:
    """Application entry point.

    Returns:
        Exit code (``0`` on clean exit, ``1`` on fatal error).
    """
    from core.config import get_settings
    from core.utils.logger import configure_from_settings

    configure_from_settings()
    settings = get_settings()

    parser = _build_arg_parser(settings)
    args = parser.parse_args()

    if args.debug:
        setup_logging(
            console_level="DEBUG",
            log_file_path=_PROJECT_ROOT / "logs" / "orchestrator.log",
        )
        logger.debug("Debug mode enabled via --debug flag.")

    _print_banner()

    print(_bold("  Initialising …"), end="", flush=True)
    try:
        orch = SystemOrchestrator(
            data_dir=args.data_dir,
            chroma_persist_dir=args.chroma_dir,
            embedding_model=args.embedding_model,
            embedding_device=args.device,
            default_model_name=args.model,
            rag_top_k=args.rag_top_k,
            memory_turns=args.memory,
            auto_index=not args.no_auto_index,
        )
        print(_green("  done."))
        print()
    except PersonalLLMException as exc:
        print(_red("  FAILED."))
        print()
        _print_error(f"Startup error [{exc.error_code}]: {exc.message}")
        logger.critical("Fatal startup error: %s", exc)
        return 1
    except Exception as exc:
        print(_red("  FAILED."))
        print()
        _print_error(f"Unexpected startup error: {exc}")
        logger.critical("Unexpected startup error", exc_info=True)
        return 1

    _print_startup_status(orch)

    try:
        _run_repl(orch, show_metadata=args.show_metadata)
    except KeyboardInterrupt:
        print()
        _handle_exit()

    return 0


# ---------------------------------------------------------------------------
# Entry point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
