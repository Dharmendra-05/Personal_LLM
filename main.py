#!/usr/bin/env python3
# ==== main.py ====
"""
Personal LLM Orchestrator — CLI Entry Point
============================================
Handles: bootstrapping, the REPL loop, slash commands, and graceful shutdown.

Performance notes
-----------------
* GENERAL_CHAT and ADVANCED_KNOWLEDGE queries are served through
  stream_query() — first token appears in ~1–2 s instead of ~40 s.
* PERSONAL_MEMORY queries use the full blocking RAG pipeline via
  process_query() because they may involve tool-call loops.
* ChromaDB and the embedding model are never loaded for GENERAL_CHAT queries.
"""

from __future__ import annotations

import os
import sys
import warnings

# ---------------------------------------------------------------------------
# ① Silence noisy native-library output as early as possible.
#    These must be set before ANY import that might load chromadb/onnxruntime.
# ---------------------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")   # chromadb 0.4+
os.environ.setdefault("ORT_LOG_LEVEL", "3")                     # onnxruntime errors only
os.environ.setdefault("ONNXRUNTIME_LOGGING_SEVERITY_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")        # suppress HuggingFace fork warning

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

# ---------------------------------------------------------------------------
# ② Targeted stderr filter for noise that bypasses env-var suppression.
#
#    Some messages (chromadb telemetry errors, ONNX GPU-discovery warnings)
#    are printed directly to stderr by native/C code *after* Python starts.
#    We intercept them here rather than letting them pollute the terminal.
#    All other stderr output passes through unchanged.
# ---------------------------------------------------------------------------

_SUPPRESS_STDERR = (
    "Failed to send telemetry event",   # chromadb telemetry capture() errors
    "DiscoverDevicesForPlatform",        # onnxruntime GPU probe on CPU-only machines
    "device_discovery.cc",              # onnxruntime GPU probe detail
    "ReadFileContents Failed",          # onnxruntime /sys/class/drm not found
    "ONNX Runtime",                     # general onnxruntime noise
)


class _StderrFilter:
    """Transparent stderr wrapper that drops known-noisy lines."""

    def write(self, text: str) -> int:
        if any(pat in text for pat in _SUPPRESS_STDERR):
            return len(text)
        return sys.__stderr__.write(text)

    def flush(self) -> None:
        sys.__stderr__.flush()

    def isatty(self) -> bool:
        return sys.__stderr__.isatty()

    # Delegate everything else (fileno, encoding, …) to the real stderr.
    def __getattr__(self, name: str):
        return getattr(sys.__stderr__, name)


sys.stderr = _StderrFilter()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# ③ Patch ChromaDB's ProductTelemetry at the class level so capture() calls
#    become no-ops.  This is more reliable than patching Posthog/Sentry
#    because ProductTelemetry is the actual dispatcher used in 0.4+.
#
#    NOTE: We do NOT import chromadb.telemetry.* here — that would eagerly
#    load onnxruntime and trigger the GPU-discovery warning before our filter
#    is active.  Instead we register a lazy patch via sys.modules hooks.
# ---------------------------------------------------------------------------

class _LazyChromaTelemPatch:
    """Patches ProductTelemetry.capture() the first time chromadb is imported."""

    def find_module(self, name, path=None):
        # We only care about the product telemetry module.
        if name == "chromadb.telemetry.product":
            return self
        return None

    def load_module(self, name):
        import importlib
        # Remove ourselves from meta_path to avoid infinite recursion.
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        mod = importlib.import_module(name)
        # Patch the class immediately after load.
        cls = getattr(mod, "ProductTelemetry", None)
        if cls is not None:
            cls.capture = lambda self, *args, **kwargs: None  # type: ignore[method-assign]
        return mod


sys.meta_path.insert(0, _LazyChromaTelemPatch())  # type: ignore[arg-type]

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
# Project imports (logging is now active)
# ---------------------------------------------------------------------------
from core.exceptions import PersonalLLMException  # noqa: E402
from core.orchestrator import OrchestratorResponse, SystemOrchestrator  # noqa: E402
from core.router import RouteMode  # noqa: E402

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_TERM_WIDTH: Final[int] = shutil.get_terminal_size(fallback=(100, 24)).columns
_WRAP_WIDTH: Final[int] = min(_TERM_WIDTH - 4, 100)
_IS_TTY: Final[bool] = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str) -> str: return _c("1", t)
def _dim(t: str) -> str:  return _c("2", t)
def _green(t: str) -> str: return _c("32", t)
def _yellow(t: str) -> str: return _c("33", t)
def _cyan(t: str) -> str:  return _c("36", t)
def _red(t: str) -> str:   return _c("31", t)
def _magenta(t: str) -> str: return _c("35", t)


def _purple_gradient(text: str) -> str:
    if not _IS_TTY:
        return text
    lines = text.strip("\n").split("\n")
    if not lines:
        return text
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

_TAGLINE: Final[str] = (
    "  Lifelong Multi-Modal Agent · Autonomous Evelynn · System Operations"
)
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
    ("/model",   "<name>",  "Switch default model for this session"),
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
    """Return a coloured mode badge string for the given route mode."""
    colours = {
        RouteMode.PERSONAL_MEMORY:   _cyan("[PERSONAL_MEMORY]"),
        RouteMode.ADVANCED_KNOWLEDGE: _yellow("[ADVANCED_KNOWLEDGE]"),
        RouteMode.GENERAL_CHAT:      _dim("[GENERAL_CHAT]"),
    }
    return colours.get(mode, f"[{mode.name}]")


def _print_response_header(mode: RouteMode) -> None:
    """Print the mode badge + speaker label that precedes every response."""
    print()
    print(f"  {_mode_badge(mode)}  {_bold(_magenta('Evelynn:'))}")
    print()


def _print_response(resp: OrchestratorResponse, show_metadata: bool = False) -> None:
    """Render a blocking OrchestratorResponse to the terminal (used for PERSONAL_MEMORY)."""
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
        _print_info("Usage: /model <name>   (see /models for available names)")
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
              python main.py --model local-gemma
              python main.py --debug --data-dir ./my_docs
        """),
    )
    parser.add_argument("--model", "-m", default=settings.ollama.default_model,
                        metavar="NAME",
                        help=f"Default model (default: {settings.ollama.default_model}).")
    parser.add_argument("--data-dir", default="assets/", metavar="PATH",
                        help="Path to the personal assets/ directory (default: assets/).")
    parser.add_argument("--chroma-dir", default=str(settings.chroma.persist_dir),
                        metavar="PATH",
                        help=f"ChromaDB persistence path (default: {settings.chroma.persist_dir}).")
    parser.add_argument("--embedding-model", default=settings.embedding.model_name,
                        metavar="MODEL",
                        help=f"Embedding model (default: {settings.embedding.model_name}).")
    parser.add_argument("--device", default=settings.embedding.device,
                        choices=["cpu", "cuda", "mps"],
                        help=f"Compute device (default: {settings.embedding.device}).")
    parser.add_argument("--rag-top-k", type=int, default=settings.chroma.top_k,
                        metavar="K",
                        help=f"RAG context chunks per query (default: {settings.chroma.top_k}).")
    parser.add_argument("--memory", type=int, default=5, metavar="TURNS",
                        help="Conversation turns to keep in memory (default: 5).")
    parser.add_argument("--no-auto-index", action="store_true",
                        help="Skip automatic indexing on first launch.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output.")
    parser.add_argument("--show-metadata", action="store_true",
                        help="Always print routing/timing metadata after each response.")
    return parser


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def _run_repl(orch: SystemOrchestrator, show_metadata: bool = False) -> None:
    """Run the Read-Eval-Print Loop until the user exits.

    Routing strategy
    ----------------
    * Route once via ``orch.route_query()`` — the decision is passed to
      ``stream_query`` or ``process_query`` so the router is **never called
      twice** for the same input.
    * ``GENERAL_CHAT`` and ``ADVANCED_KNOWLEDGE`` → ``stream_query()``:
      tokens printed as they arrive; first word in ~1–2 s.
    * ``PERSONAL_MEMORY`` → ``process_query()``:
      full RAG pipeline including tool-call loop.
    """
    session_model: list[str | None] = [None]

    print(_dim(f"  Type your message and press Enter.  "
               f"Use {_yellow('/help')} for commands, {_yellow('Ctrl+C')} to quit."))
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
                    resp = orch.process_query(query, model_name=session_model[0])
                    _print_response(resp, show_metadata=True)
            else:
                _print_error(f"Unknown command '{cmd}'. Type /help for commands.")
            continue

        # ── Regular query ────────────────────────────────────────────────
        # Route ONCE — decision is forwarded to avoid double-routing.
        decision = orch.route_query(raw)

        if decision.mode != RouteMode.GENERAL_CHAT:
            _print_info("Thinking …")

        if decision.mode in (RouteMode.GENERAL_CHAT, RouteMode.ADVANCED_KNOWLEDGE):
            # ── Streaming path ───────────────────────────────────────────
            # Tokens arrive in ~1–2 s instead of buffering for 40+ s.
            _print_response_header(decision.mode)

            stream_start = time.perf_counter()
            char_count = 0
            print("    ", end="", flush=True)  # initial indent

            try:
                for chunk in orch.stream_query(
                    raw,
                    decision=decision,
                    model_name=session_model[0],
                ):
                    # Preserve newlines with proper indentation.
                    if "\n" in chunk:
                        segments = chunk.split("\n")
                        for i, seg in enumerate(segments):
                            if i > 0:
                                print()  # real newline
                                if seg:  # indent continuation line
                                    print("    ", end="", flush=True)
                            print(seg, end="", flush=True)
                            char_count += len(seg)
                    else:
                        print(chunk, end="", flush=True)
                        char_count += len(chunk)

            except Exception as exc:
                print()
                _print_error(f"Streaming error: {exc}")

            print("\n")  # blank line after response

            if show_metadata:
                _print_metadata(
                    model=orch._default_model_name or "unknown",
                    mode=decision.mode,
                    confidence=decision.confidence,
                    rag_chunks=0,
                    duration=time.perf_counter() - stream_start,
                    signals=decision.matched_signals,
                )

        else:
            # ── Blocking RAG path (PERSONAL_MEMORY) ──────────────────────
            resp = orch.process_query(
                raw,
                model_name=session_model[0],
                decision=decision,
            )
            _print_response(resp, show_metadata=show_metadata)


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


def _print_startup_status(orch: SystemOrchestrator) -> None:
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
    count_str = f"{count} chunks" if count >= 0 else _dim("Deferred (loads on first RAG query)")
    print(f"  {_icon(vs_health)}  Vector Store      {count_str}")

    dm_ok = health.get("default_model", False)
    print(f"  {_icon(dm_ok)}  Default Model     {_bold(default_model_name)} @ {url}")

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


if __name__ == "__main__":
    sys.exit(main())