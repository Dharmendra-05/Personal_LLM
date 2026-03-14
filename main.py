#!/usr/bin/env python3
# ==== main.py ====
"""
Personal LLM Orchestrator — CLI Entry Point
============================================

Performance routing
-------------------
* GENERAL_CHAT   → stream_query() with optional dedicated fast-chat model
                   (--chat-model flag, or OLLAMA_CHAT_MODEL env var).
                   Configure tinyllama / phi3:mini for near-instant greetings.
* ADVANCED_KNOWLEDGE → stream_query(); auto-falls back to cloud API when
                   Ollama is unreachable.
* PERSONAL_MEMORY → process_query() (full blocking RAG + tool-call pipeline).

Startup optimisations
---------------------
* ONNX GPU-discovery warning suppressed via OS-level fd-2 redirect during init
  (the only reliable method — native C++ bypasses Python's sys.stderr).
* Embedding model pre-warmed in a background daemon thread after startup so the
  first PERSONAL_MEMORY query doesn't pay the ~58 s cold-start.
* max_tokens capped per mode: GENERAL_CHAT=512, ADVANCED=1024, MEMORY=2048.
  On 8 GB RAM this alone cuts generation time by ~4x.
"""

from __future__ import annotations

import os
import sys
import warnings

# ---------------------------------------------------------------------------
# ① Env-var suppressions — must precede any C-extension import.
# ---------------------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("ORT_LOG_LEVEL", "3")
os.environ.setdefault("ONNXRUNTIME_LOGGING_SEVERITY_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=UserWarning,  module="pydantic.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# ---------------------------------------------------------------------------
# ② OS-level stderr redirect helper.
#
#    Native C++ (onnxruntime, chromadb) writes to file descriptor 2 directly,
#    bypassing Python's sys.stderr object entirely.  The only reliable fix is
#    os.dup2 to redirect fd 2 at the OS level.  Used only around the
#    SystemOrchestrator init call so normal stderr is unaffected.
# ---------------------------------------------------------------------------
import contextlib


@contextlib.contextmanager
def _suppress_native_stderr():
    """Redirect OS-level stderr (fd 2) to /dev/null temporarily."""
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield          # no real fd (pytest capture etc.) — skip
        return
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)


# ---------------------------------------------------------------------------
# ③ Lazy ChromaDB telemetry patch.
#    Intercepts chromadb.telemetry.product at import time and replaces
#    ProductTelemetry.capture() with a no-op.  Avoids eager chromadb import
#    at startup (which would trigger ONNX before our fd redirect is active).
# ---------------------------------------------------------------------------


class _LazyChromaTelemPatch:
    def find_module(self, name, path=None):
        return self if name == "chromadb.telemetry.product" else None

    def load_module(self, name):
        import importlib
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        mod = importlib.import_module(name)
        cls = getattr(mod, "ProductTelemetry", None)
        if cls is not None:
            cls.capture = lambda self, *a, **kw: None  # type: ignore[method-assign]
        return mod


sys.meta_path.insert(0, _LazyChromaTelemPatch())  # type: ignore[arg-type]

# ---------------------------------------------------------------------------

import argparse
import shutil
import textwrap
import threading
import time
from pathlib import Path
from typing import Final, TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import AppSettings

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Bootstrap logging before any other project imports
# ---------------------------------------------------------------------------
from core.utils.logger import configure_from_settings, get_logger, setup_logging  # noqa: E402

setup_logging(
    console_level="WARNING",
    log_file_path=_PROJECT_ROOT / "logs" / "orchestrator.log",
)
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from core.exceptions import PersonalLLMException  # noqa: E402
from core.orchestrator import OrchestratorResponse, SystemOrchestrator  # noqa: E402
from core.router import RouteMode  # noqa: E402

# ---------------------------------------------------------------------------
# Per-mode token caps.
#
# At 8 GB RAM with llama3:8b on CPU each 512-token block takes ~20 s.
# Capping GENERAL_CHAT at 512 tokens alone cuts response time from ~40 s to
# ~5–8 s.  Users can override with --max-tokens-chat etc. in a future flag.
# ---------------------------------------------------------------------------
_MAX_TOKENS: Final[dict[RouteMode, int]] = {
    RouteMode.GENERAL_CHAT:       512,
    RouteMode.ADVANCED_KNOWLEDGE: 1024,
    RouteMode.PERSONAL_MEMORY:    2048,
}

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_TERM_WIDTH: Final[int] = shutil.get_terminal_size(fallback=(100, 24)).columns
_WRAP_WIDTH: Final[int] = min(_TERM_WIDTH - 4, 100)
_IS_TTY: Final[bool] = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def _bold(t: str) -> str:    return _c("1", t)
def _dim(t: str) -> str:     return _c("2", t)
def _green(t: str) -> str:   return _c("32", t)
def _yellow(t: str) -> str:  return _c("33", t)
def _cyan(t: str) -> str:    return _c("36", t)
def _red(t: str) -> str:     return _c("31", t)
def _magenta(t: str) -> str: return _c("35", t)


def _purple_gradient(text: str) -> str:
    if not _IS_TTY:
        return text
    lines = text.strip("\n").split("\n")
    result = []
    for i, line in enumerate(lines):
        ratio = i / max(1, len(lines) - 1)
        r = int(255 - (170 * ratio))
        result.append(f"\033[38;2;{r};0;255m{line}\033[0m")
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
_TAGLINE: Final[str] = "  Lifelong Multi-Modal Agent · Autonomous Evelynn · System Operations"
_VERSION: Final[str] = "v1.2.0"

# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

_COMMANDS: Final[list[tuple[str, str, str]]] = [
    ("/help",    "",        "Show this help message"),
    ("/models",  "",        "List all registered LLM models"),
    ("/reload",  "",        "Re-index documents from the data directory"),
    ("/stats",   "",        "Show vector store collection statistics"),
    ("/history", "",        "Display recent conversation history"),
    ("/clear",   "",        "Clear conversation history"),
    ("/health",  "",        "Check subsystem connectivity"),
    ("/route",   "<query>", "Show routing decision for a query (dry run)"),
    ("/model",   "<n>",     "Switch default model for this session"),
    ("/debug",   "<query>", "Process query and show full metadata"),
    ("/exit",    "",        "Exit the orchestrator"),
    ("/quit",    "",        "Exit the orchestrator"),
]

# ---------------------------------------------------------------------------
# Printer helpers
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    print(_purple_gradient(_BANNER), end="")
    print(_bold(_magenta(_TAGLINE)))
    width = min(_TERM_WIDTH, 80)
    print(_dim("  " + "─" * (width - 2)))
    print(f"  {_bold('Version:')} {_VERSION}   {_bold('Python:')} {sys.version.split()[0]}")
    print(_dim("  " + "─" * (width - 2)))
    print()


def _print_help() -> None:
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


def _mode_badge(mode: RouteMode) -> str:
    colours = {
        RouteMode.PERSONAL_MEMORY:    _cyan("[PERSONAL_MEMORY]"),
        RouteMode.ADVANCED_KNOWLEDGE: _yellow("[ADVANCED_KNOWLEDGE]"),
        RouteMode.GENERAL_CHAT:       _dim("[GENERAL_CHAT]"),
    }
    return colours.get(mode, f"[{mode.name}]")


def _print_response_header(mode: RouteMode) -> None:
    print()
    print(f"  {_mode_badge(mode)}  {_bold(_magenta('Evelynn:'))}")
    print()


def _stream_print(chunks) -> tuple[str, int]:
    """Print streaming chunks with clean line-buffered indentation.

    The previous chunk-at-a-time approach had a formatting bug: when Ollama's
    first chunk started with or was a newline, the subsequent text appeared
    without indentation.

    This version buffers text until a natural word/sentence boundary, then
    wraps and prints the complete line with consistent 4-space indentation.
    Each physical newline from the model also flushes the buffer and resets
    the indent.

    Returns:
        (full_text, char_count)
    """
    _INDENT = "    "
    line_buf: list[str] = []   # chars accumulated for the current visual line
    full_parts: list[str] = [] # everything, for history
    char_count = 0
    line_started = False        # True once we've printed something on this line

    def _emit_line(text: str) -> None:
        """Wrap *text* and print with indentation."""
        nonlocal line_started
        if not text:
            return
        if not line_started:
            # First text on this line — wrap the whole thing with indent.
            wrapped = textwrap.fill(
                text, width=_WRAP_WIDTH,
                initial_indent=_INDENT, subsequent_indent=_INDENT,
            )
            print(wrapped, end="", flush=True)
        else:
            print(text, end="", flush=True)
        line_started = True

    def _flush_buf() -> None:
        _emit_line("".join(line_buf))
        line_buf.clear()

    for raw_chunk in chunks:
        if not raw_chunk:
            continue
        full_parts.append(raw_chunk)
        char_count += len(raw_chunk)

        # Split on newlines so each model paragraph gets its own indent.
        parts = raw_chunk.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                # A newline boundary — flush current buffer and start fresh.
                _flush_buf()
                print()               # actual newline to terminal
                line_started = False  # next text on this line gets indent

            if part:
                line_buf.append(part)

            # Flush on space (word boundary) or when buffer is long enough to
            # avoid holding too many chars before they appear on screen.
            joined = "".join(line_buf)
            if part.endswith(" ") or len(joined) >= 60:
                _flush_buf()

    # Flush any remaining content.
    _flush_buf()

    return "".join(full_parts), char_count


def _print_response(resp: OrchestratorResponse, show_metadata: bool = False) -> None:
    """Render a blocking OrchestratorResponse (PERSONAL_MEMORY) to the terminal."""
    if not resp.success:
        print()
        print(_red(f"  ✖  Error: {resp.error}"))
        print()
        return

    _print_response_header(resp.route_mode)

    for paragraph in resp.text.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(
                paragraph, width=_WRAP_WIDTH,
                initial_indent="    ", subsequent_indent="    ",
            ))
        else:
            print()

    if show_metadata:
        _print_metadata(
            model=resp.model_name,
            mode=resp.route_mode,
            confidence=resp.decision.confidence,
            rag_chunks=resp.rag_context_used,
            duration=resp.duration_seconds,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            signals=resp.decision.matched_signals,
        )


def _print_metadata(
    *,
    model: str,
    mode: RouteMode,
    confidence: float,
    rag_chunks: int,
    duration: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    signals: list[str] | None = None,
) -> None:
    print()
    print(_dim("  ── Metadata " + "─" * 40))
    print(_dim(f"  Model      : {model}"))
    print(_dim(f"  Route      : {mode.name} (confidence={confidence:.2f})"))
    print(_dim(f"  RAG chunks : {rag_chunks}"))
    tok = ""
    if prompt_tokens and completion_tokens and duration > 0:
        tok = (
            f" | tokens: {prompt_tokens}→{completion_tokens} "
            f"({completion_tokens / duration:.0f} tok/s)"
        )
    print(_dim(f"  Duration   : {duration:.2f}s{tok}"))
    sigs_str = ", ".join((signals or [])[:5]) or "none"
    print(_dim(f"  Signals    : {sigs_str}"))
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
# Stdin flush helper
# ---------------------------------------------------------------------------


def _flush_stdin() -> None:
    """Discard keystrokes buffered in stdin during a long LLM response.

    When generation takes 5–40 s, users often press Enter multiple times
    while waiting.  Those newlines accumulate in stdin and get consumed as
    empty queries on the next loop iteration, producing several blank
    ``You ›`` prompts.
    """
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass  # Non-TTY (pipe, Windows, pytest) — skip silently.


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _handle_models(orch: SystemOrchestrator) -> None:
    _print_section("Registered Models")
    models = orch.list_models()
    if not models:
        _print_error("No models registered. Add YAML configs to models/model_configs/.")
        return
    for m in models:
        print(f"    {_bold(_yellow(m['name']))}")
        print(f"      {_dim('Provider:')} {m.get('provider','?')}  "
              f"{_dim('Tag:')} {m.get('model_tag','?')}  "
              f"{_dim('Temp:')} {m.get('temperature','?')}")
        if desc := m.get("description", ""):
            print(f"      {_dim('Desc:')} {desc[:80]}")
        print()


def _handle_reload(orch: SystemOrchestrator) -> None:
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
    _print_section("Vector Store Stats")
    stats = orch.get_collection_stats()
    print(f"    {_bold('Collection :')} {stats['collection_name']}")
    print(f"    {_bold('Documents  :')} {stats['document_count']}")
    print()


def _handle_history(orch: SystemOrchestrator) -> None:
    _print_section("Conversation History")
    history = orch.get_history()
    if not history:
        _print_info("No conversation history yet.")
        return
    for turn in history:
        role_fmt = _bold(_cyan("[You]")) if turn["role"] == "user" else _bold("[AI]")
        mode_tag = _dim(f"({turn['mode']})")
        print(f"    {role_fmt} {mode_tag}")
        for line in turn["content"].split("\n"):
            print(f"      {line}")
        print()


def _handle_clear(orch: SystemOrchestrator) -> None:
    orch.clear_history()
    _print_ok("Conversation history cleared.")


def _handle_health(orch: SystemOrchestrator) -> None:
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
    query = args_str.strip()
    if not query:
        _print_error("Usage: /route <query>")
        return
    _print_section("Routing Decision (dry run)")
    explanation = orch._router.explain(query)
    for line in explanation.split("\n"):
        print(f"    {line}")
    print()


def _handle_model_switch(
    orch: SystemOrchestrator,
    args_str: str,
    session_model: list[str | None],
) -> None:
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
    parser = argparse.ArgumentParser(
        prog="personal-llm-orchestrator",
        description="Personal LLM Orchestrator — local AI with RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --model local-llama3
              python main.py --chat-model tinyllama --model local-llama3
              python main.py --debug --data-dir ./my_docs
        """),
    )
    parser.add_argument("--model", "-m", default=settings.ollama.default_model,
                        metavar="NAME",
                        help=f"Default model (default: {settings.ollama.default_model}).")
    parser.add_argument(
        "--chat-model",
        default=os.environ.get("OLLAMA_CHAT_MODEL", ""),
        metavar="NAME",
        help=(
            "Dedicated model for GENERAL_CHAT queries (e.g. 'tinyllama', 'phi3:mini'). "
            "Defaults to --model. On 8 GB RAM: run llama3 for advanced, "
            "tinyllama for instant greetings."
        ),
    )
    parser.add_argument("--data-dir", default="assets/", metavar="PATH",
                        help="Path to the assets/ directory (default: assets/).")
    parser.add_argument("--chroma-dir", default=str(settings.chroma.persist_dir),
                        metavar="PATH",
                        help=f"ChromaDB persistence path.")
    parser.add_argument("--embedding-model", default=settings.embedding.model_name,
                        metavar="MODEL", help=f"Embedding model.")
    parser.add_argument("--device", default=settings.embedding.device,
                        choices=["cpu", "cuda", "mps"], help="Compute device.")
    parser.add_argument("--rag-top-k", type=int, default=settings.chroma.top_k,
                        metavar="K", help="RAG context chunks per query.")
    parser.add_argument("--memory", type=int, default=5, metavar="TURNS",
                        help="Conversation turns to keep in memory (default: 5).")
    parser.add_argument("--no-auto-index", action="store_true",
                        help="Skip automatic indexing on first launch.")
    parser.add_argument("--no-prewarm", action="store_true",
                        help="Disable background embedding model pre-warm.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output.")
    parser.add_argument("--show-metadata", action="store_true",
                        help="Always print routing/timing metadata after each response.")
    return parser


# ---------------------------------------------------------------------------
# Background embedding pre-warm
# ---------------------------------------------------------------------------


def _start_prewarm_thread(orch: SystemOrchestrator) -> threading.Thread:
    """Pre-warm the embedding model in the background after startup.

    On 8 GB RAM, loading torch + sentence-transformers takes ~58 s on first
    use.  By doing it in a daemon thread immediately after init, the model is
    ready long before the user types their first PERSONAL_MEMORY query.
    Calling collection_count() triggers the lazy ChromaDB + embedding init.
    """
    def _warm() -> None:
        try:
            logger.debug("prewarm: triggering VectorStore lazy load …")
            with _suppress_native_stderr():
                _ = orch._vector_store.collection_count()
            logger.debug("prewarm: VectorStore and embedding model ready.")
        except Exception as exc:
            logger.debug("prewarm failed (non-fatal — first RAG query will re-try): %s", exc)

    t = threading.Thread(target=_warm, name="embedding-prewarm", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def _run_repl(
    orch: SystemOrchestrator,
    show_metadata: bool = False,
    chat_model: str | None = None,
) -> None:
    """Run the Read-Eval-Print Loop until the user exits.

    Routing strategy
    ----------------
    Route once via route_query() — decision forwarded; router never called
    twice for the same input.

    * GENERAL_CHAT / ADVANCED_KNOWLEDGE → stream_query():
      - GENERAL_CHAT uses ``chat_model`` if configured (smaller = faster).
      - Tokens stream as they arrive; first word in ~1–2 s.
      - max_tokens capped per _MAX_TOKENS to limit generation time on slow HW.
      - Ollama errors auto-fall back to cloud inside stream_query().
    * PERSONAL_MEMORY → process_query():
      - Full RAG retrieval + tool-call pipeline.
      - Embedding model pre-warmed in background so cold-start is avoided.

    Stdin is flushed after each response to prevent buffered Enter presses
    from appearing as empty queries on the next turn.
    """
    session_model: list[str | None] = [None]

    print(_dim(f"  Type your message and press Enter.  "
               f"Use {_yellow('/help')} for commands, {_yellow('Ctrl+C')} to quit."))
    if chat_model:
        print(_dim(f"  Fast-chat model active: {_bold(chat_model)} (for GENERAL_CHAT)"))
    print()

    while True:
        # ── Read ────────────────────────────────────────────────────────
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
            elif cmd == "/help":      _print_help()
            elif cmd == "/models":    _handle_models(orch)
            elif cmd == "/reload":    _handle_reload(orch)
            elif cmd == "/stats":     _handle_stats(orch)
            elif cmd == "/history":   _handle_history(orch)
            elif cmd == "/clear":     _handle_clear(orch)
            elif cmd == "/health":    _handle_health(orch)
            elif cmd == "/route":     _handle_route(orch, args_str)
            elif cmd == "/model":     _handle_model_switch(orch, args_str, session_model)
            elif cmd == "/debug":
                query = args_str.strip()
                if not query:
                    _print_error("Usage: /debug <query>")
                else:
                    resp = orch.process_query(query, model_name=session_model[0])
                    _print_response(resp, show_metadata=True)
            else:
                _print_error(f"Unknown command '{cmd}'. Type /help for commands.")
            continue

        # ── Regular query ────────────────────────────────────────────────
        # Route ONCE — decision is forwarded; no double-route.
        decision = orch.route_query(raw)

        if decision.mode != RouteMode.GENERAL_CHAT:
            _print_info("Thinking …")

        if decision.mode in (RouteMode.GENERAL_CHAT, RouteMode.ADVANCED_KNOWLEDGE):
            # ── Streaming path ─────────────────────────────────────────
            effective_model = (
                chat_model
                if (chat_model and decision.mode == RouteMode.GENERAL_CHAT)
                else session_model[0]
            )

            _print_response_header(decision.mode)
            stream_start = time.perf_counter()

            try:
                chunks = orch.stream_query(
                    raw,
                    decision=decision,
                    model_name=effective_model,
                    max_tokens=_MAX_TOKENS[decision.mode],
                )
                full_text, char_count = _stream_print(chunks)
            except Exception as exc:
                full_text, char_count = "", 0
                _print_error(f"Streaming error: {exc}")

            print("\n")  # blank line after response

            if show_metadata:
                _print_metadata(
                    model=effective_model or orch._default_model_name or "unknown",
                    mode=decision.mode,
                    confidence=decision.confidence,
                    rag_chunks=0,
                    duration=time.perf_counter() - stream_start,
                    signals=decision.matched_signals,
                )

        else:
            # ── Blocking RAG path (PERSONAL_MEMORY) ────────────────────
            resp = orch.process_query(
                raw,
                model_name=session_model[0],
                decision=decision,
                max_tokens=_MAX_TOKENS[RouteMode.PERSONAL_MEMORY],
            )
            _print_response(resp, show_metadata=show_metadata)

        # Flush stdin to discard Enter presses buffered during generation.
        _flush_stdin()


def _handle_exit() -> None:
    print()
    print(_magenta("  ╔══════════════════════════════════════╗"))
    print(_magenta("  ║") + "   Evelynn is shutting down...        " + _magenta("║"))
    print(_magenta("  ║") + "   Goodbye!  🤖                         " + _magenta("║"))
    print(_magenta("  ╚══════════════════════════════════════╝"))
    print()
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# Startup status block
# ---------------------------------------------------------------------------


def _print_startup_status(orch: SystemOrchestrator, chat_model: str | None = None) -> None:
    models = orch.list_models()
    stats = orch.get_collection_stats()
    health = orch.health_check()

    def _icon(ok: bool) -> str:
        return _green("✔") if ok else _red("✖")

    default_model_name = orch._default_model_name or "(none)"
    url = "N/A"
    if default_model_name != "(none)":
        try:
            client = orch._registry.get_model(default_model_name)
            url = getattr(client, "base_url", "N/A")
        except Exception:
            url = "Error"

    print(_bold("  System Status"))
    print(_dim("  " + "─" * 46))
    print(f"  {_icon(health.get('registry', False))}  "
          f"Model Registry    {len(models)} model(s)")

    vs_health = health.get("vector_store", False)
    count = stats.get("document_count", -1)
    count_str = (
        f"{count} chunks" if count >= 0
        else _dim("Deferred (pre-warming in background)")
    )
    print(f"  {_icon(vs_health)}  Vector Store      {count_str}")

    dm_ok = health.get("default_model", False)
    chat_note = f"  {_dim('(chat: ' + chat_model + ')')}" if chat_model else ""
    print(f"  {_icon(dm_ok)}  Default Model     "
          f"{_bold(default_model_name)} @ {url}{chat_note}")

    if not dm_ok:
        print()
        print(_yellow("  ⚠  Ollama is not reachable at ") + _bold(url))
        print(_dim("     Ensure `ollama serve` is running. Type /health to retry."))

    print(_dim("  " + "─" * 46))
    print()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> int:
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

    chat_model: str | None = args.chat_model.strip() or None

    print(_bold("  Initialising …"), end="", flush=True)
    try:
        # _suppress_native_stderr() redirects fd 2 at the OS level so
        # onnxruntime's GPU-discovery warning never reaches the terminal.
        with _suppress_native_stderr():
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

    # Pre-warm embedding model in the background.
    if not args.no_prewarm:
        _start_prewarm_thread(orch)

    _print_startup_status(orch, chat_model=chat_model)

    try:
        _run_repl(orch, show_metadata=args.show_metadata, chat_model=chat_model)
    except KeyboardInterrupt:
        print()
        _handle_exit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
