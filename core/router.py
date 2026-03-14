# ==== core/router.py ====
"""
Query routing and intent classification for the Personal LLM Orchestrator.

Overview
--------
The :class:`QueryRouter` classifies each incoming user query into one of
three processing modes before the orchestrator dispatches it to the LLM:

* **RAG** — The query appears to be asking about specific knowledge that
  should be grounded in indexed documents (factual questions, document
  references, "what is / explain / summarise" patterns).

* **CODE** — The query is a programming task: writing, reviewing, debugging,
  or explaining code.

* **CHAT** — General conversational exchange, opinion, reasoning, or
  creative tasks that need no external context retrieval.

Classification strategy
-----------------------
The router uses a **weighted multi-signal** approach rather than a single
regex so that borderline queries land in the right bucket:

1. **Keyword/phrase matching** — Each mode has a curated list of high-signal
   trigger phrases (e.g. ``"debug"`` → CODE, ``"according to"`` → RAG).
   Each match contributes a positive score to its mode.

2. **Regex pattern matching** — Structural patterns (e.g. code fence
   markers, import statements) add higher-weight signals.

3. **Heuristic tiebreakers** — Query length, question-word presence, and
   explicit document-reference markers break ties.

The mode with the highest cumulative score wins.  When all scores are zero
(no signals fired), the router falls back to **CHAT**.

Design notes
------------
* The router is **stateless** and **side-effect-free** — safe to call from
  any thread or async context without locking.
* Adding new intents requires only a new :class:`RouteMode` member and a
  corresponding entry in :data:`_SIGNAL_TABLE`.
* The :class:`RoutingDecision` return type gives the orchestrator (and tests)
  full transparency into *why* a routing decision was made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final, Sequence

from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Route mode enum
# ---------------------------------------------------------------------------


class RouteMode(Enum):
    """Enumeration of supported query routing destinations."""
    
    PERSONAL_MEMORY = auto()
    ADVANCED_KNOWLEDGE = auto()
    GENERAL_CHAT = auto()


# ---------------------------------------------------------------------------
# Routing decision value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Immutable result of a :meth:`QueryRouter.route` call.

    Attributes:
        mode: The selected :class:`RouteMode` destination.
        confidence: A normalised score in ``[0.0, 1.0]`` indicating how
            strongly the router commits to this decision.  ``1.0`` means
            every signal unanimously agreed; values below ``0.5`` indicate
            a low-confidence fallback.
        scores: Raw accumulated score for each mode before normalisation.
            Useful for debugging and unit testing.
        matched_signals: Human-readable list of signal descriptions that
            contributed to the final decision (e.g.
            ``["keyword:debug", "pattern:code_fence"]``).
        query_preview: First 80 characters of the classified query, stored
            for logging without retaining the full query string.

    Example:
        >>> decision = router.route("How do I fix a segfault in C?")
        >>> decision.mode
        <RouteMode.CODE: 2>
        >>> decision.confidence
        0.85
        >>> decision.matched_signals
        ['keyword:segfault', 'keyword:fix', 'pattern:language_mention']
    """

    mode: RouteMode
    confidence: float
    scores: dict[str, float]
    matched_signals: list[str]
    query_preview: str

    def __str__(self) -> str:
        return (
            f"RoutingDecision(mode={self.mode.name}, "
            f"confidence={self.confidence:.2f}, "
            f"signals={self.matched_signals})"
        )


# ---------------------------------------------------------------------------
# Signal table
# ---------------------------------------------------------------------------
# Each entry: (mode, weight, signal_description, compiled_pattern_or_None)
# For keyword entries pattern is None and matching is done via substring search.
# For regex entries the compiled pattern is used.

_SignalEntry = tuple[RouteMode, float, str, re.Pattern[str] | None]


def _kw(mode: RouteMode, weight: float, keyword: str) -> _SignalEntry:
    """Create a keyword signal entry (case-insensitive substring match)."""
    return (mode, weight, f"keyword:{keyword}", None)


def _rx(mode: RouteMode, weight: float, label: str, pattern: str) -> _SignalEntry:
    """Create a regex signal entry."""
    return (mode, weight, f"pattern:{label}", re.compile(pattern, re.IGNORECASE | re.DOTALL))


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------
# Weights are calibrated so that a single strong signal (weight ≥ 1.5)
# overrides weak noise, while multiple weak signals (0.5) can combine to
# shift the decision.

_SIGNAL_TABLE: Final[list[_SignalEntry]] = [

    # ── GENERAL_CHAT signals (Greetings, basic chit-chat) ──────────────────────
    _kw(RouteMode.GENERAL_CHAT, 3.0, "hi"),
    _kw(RouteMode.GENERAL_CHAT, 3.0, "hello"),
    _kw(RouteMode.GENERAL_CHAT, 3.0, "hey"),
    _kw(RouteMode.GENERAL_CHAT, 2.0, "how are you"),
    _kw(RouteMode.GENERAL_CHAT, 3.0, "good morning"),
    _kw(RouteMode.GENERAL_CHAT, 3.0, "good evening"),
    _kw(RouteMode.GENERAL_CHAT, 1.5, "greetings"),
    _rx(RouteMode.GENERAL_CHAT, 4.0, "explicit_greet", r"^(hi|hello|hey|hola)\b"),

    # ── PERSONAL_MEMORY signals (Retrieving from your Assets) ──────────────
    _kw(RouteMode.PERSONAL_MEMORY, 2.0, "according to"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.0, "based on the document"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.0, "in my files"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.0, "do i have"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.5, "list all the text files"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.5, "list my assets"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.5, "list all assets"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.2, "show my files"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.0, "personal memory"),
    _kw(RouteMode.PERSONAL_MEMORY, 2.0, "personal section"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.8, "find information about"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.8, "retrieve"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.5, "mine"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.2, "who is"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.2, "when did"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.2, "where is"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.0, "tell me about"),
    _kw(RouteMode.PERSONAL_MEMORY, 1.0, "my"),
    _kw(RouteMode.PERSONAL_MEMORY, 0.8, "history of"),
    _rx(RouteMode.PERSONAL_MEMORY, 0.5, "explicit_i", r"\b(i)\b"),
    # Typo command detection (route to ADVANCED or at least AWAY from memory if not semantic)
    _rx(RouteMode.ADVANCED_KNOWLEDGE, 5.0, "cmd_typo", r"^\\[a-z]+$"), 

    # ── ADVANCED_KNOWLEDGE signals (Coding, Mathematics, System Design, General Knowledge) ────
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "write a function"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "how does quantum"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "implement"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "refactor"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "debug"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "fix this code"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "explain the theory"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "history of the universe"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "who was"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.5, "search"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.5, "research"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "what are the differences"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 2.0, "compare"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.8, "code review"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.8, "write tests"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.8, "tell me a story"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.8, "what happened in"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.5, "traceback"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.5, "optimise"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.5, "algorithm"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.2, "function"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.2, "class"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.2, "async"),
    _kw(RouteMode.ADVANCED_KNOWLEDGE, 1.0, "explain the concept"),
    _rx(RouteMode.ADVANCED_KNOWLEDGE, 4.0, "explicit_search", r"^(search|research)\s+"),
    _rx(RouteMode.ADVANCED_KNOWLEDGE, 3.0, "code_fence", r"```[a-z]*\s"),
    _rx(RouteMode.ADVANCED_KNOWLEDGE, 2.5, "import_stmt", r"\b(import|from\s+\w+\s+import)\b"),
]


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class QueryRouter:
    """Stateless query classifier that maps user input to a :class:`RouteMode`.

    The router runs the full signal table against each query string and
    returns a :class:`RoutingDecision` with detailed scoring metadata.

    Args:
        advanced_threshold: Minimum total ADVANCED score before ADVANCED mode is eligible.
        default_mode: Mode returned when no signals fire above their
            thresholds.  Defaults to :attr:`RouteMode.CHAT`.

    Attributes:
        rag_threshold: Configured RAG eligibility threshold.
        code_threshold: Configured CODE eligibility threshold.
        default_mode: Fallback mode when no signals fire.

    Example:
        >>> router = QueryRouter()
        >>> d = router.route("Can you write a Python function to sort a list?")
        >>> d.mode
        <RouteMode.CODE: 2>
    """

    def __init__(
        self,
        advanced_threshold: float = 1.0,
        default_mode: RouteMode = RouteMode.GENERAL_CHAT,
    ) -> None:
        self.advanced_threshold: float = advanced_threshold
        self.default_mode: RouteMode = default_mode

        # Pre-compile keyword lower-case cache for fast matching
        self._keyword_signals: list[tuple[RouteMode, float, str, str]] = []
        self._regex_signals: list[tuple[RouteMode, float, str, re.Pattern[str]]] = []

        for mode, weight, label, pattern in _SIGNAL_TABLE:
            if pattern is None:
                # Keyword signal — extract the keyword from label "keyword:xxx"
                kw = label.split(":", 1)[1]
                self._keyword_signals.append((mode, weight, label, kw))
            else:
                self._regex_signals.append((mode, weight, label, pattern))

        logger.debug(
            "QueryRouter initialised: %d keyword signals, %d regex signals",
            len(self._keyword_signals),
            len(self._regex_signals),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def route(self, query: str) -> RoutingDecision:
        """Classify *query* and return a :class:`RoutingDecision`.

        Args:
            query: Raw user input string.  Must be non-empty.

        Returns:
            A :class:`RoutingDecision` with the selected mode, confidence
            score, raw per-mode scores, and matched signal descriptions.

        Raises:
            ValueError: If *query* is empty or whitespace-only.

        Example:
            >>> decision = router.route("Explain how RAG works")
            >>> decision.mode
            <RouteMode.PERSONAL_MEMORY: 1>
        """
        if not query or not query.strip():
            raise ValueError("QueryRouter.route(): query must not be empty.")

        query_lower: str = query.lower()
        scores: dict[RouteMode, float] = {
            RouteMode.PERSONAL_MEMORY: 0.0,
            RouteMode.ADVANCED_KNOWLEDGE: 0.0,
            RouteMode.GENERAL_CHAT: 0.0,
        }
        matched: list[str] = []

        # --- Keyword signals ---
        for mode, weight, label, keyword in self._keyword_signals:
            if keyword in query_lower:
                scores[mode] += weight
                matched.append(label)

        # --- Regex signals ---
        for mode, weight, label, pattern in self._regex_signals:
            if pattern.search(query):
                scores[mode] += weight
                matched.append(label)

        # --- Heuristic adjustments ---
        scores, extra_signals = self._apply_heuristics(query, query_lower, scores)
        matched.extend(extra_signals)

        # --- Determine winner ---
        mode, confidence = self._select_mode(scores)

        decision = RoutingDecision(
            mode=mode,
            confidence=round(confidence, 4),
            scores={m.name: round(s, 4) for m, s in scores.items()},
            matched_signals=matched,
            query_preview=query[:80],
        )

        logger.debug("QueryRouter.route(): %s", decision)
        return decision

    def explain(self, query: str) -> str:
        """Return a human-readable explanation of a routing decision.

        Useful for debugging and the ``/explain`` CLI command.

        Args:
            query: The query to classify and explain.

        Returns:
            A multi-line string describing the routing decision.
        """
        d = self.route(query)
        lines: list[str] = [
            f"Query    : {d.query_preview!r}",
            f"Decision : {d.mode.name} (confidence={d.confidence:.2f})",
            "Scores   :",
        ]
        for name, score in d.scores.items():
            lines.append(f"  {name:8s} = {score:.2f}")
        lines.append("Signals  :")
        for sig in d.matched_signals:
            lines.append(f"  • {sig}")
        if not d.matched_signals:
            lines.append("  (none — default fallback)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_heuristics(
        self,
        query: str,
        query_lower: str,
        scores: dict[RouteMode, float],
    ) -> tuple[dict[RouteMode, float], list[str]]:
        """Apply lightweight heuristic score adjustments.

        Heuristics are intentionally low-weight additive signals that break
        ties without overriding strong keyword/regex scores.

        Args:
            query: Original query string (for length checks).
            query_lower: Lowercased query string.
            scores: Mutable scores dict to adjust in-place.
        """
        extras: list[str] = []

        # Long queries (> 120 chars) lean toward ADVANCED_KNOWLEDGE
        if len(query) > 120 and scores[RouteMode.PERSONAL_MEMORY] < 1.0:
            scores[RouteMode.ADVANCED_KNOWLEDGE] += 0.5
            extras.append("heuristic:long_complex_query")

        # Short single-word queries are almost always PERSONAL_MEMORY (casual chat)
        # UNLESS GENERAL_CHAT already has a higher score (greeting handled by signal table)
        tokens: list[str] = query.strip().split()
        if len(tokens) <= 2 and scores[RouteMode.ADVANCED_KNOWLEDGE] < 1.0 and scores[RouteMode.GENERAL_CHAT] < 1.0:
            scores[RouteMode.PERSONAL_MEMORY] += 1.0
            extras.append("heuristic:very_short_query")
            
        # Presence of backtick inline code — medium ADVANCED signal
        if "`" in query and scores[RouteMode.ADVANCED_KNOWLEDGE] < 1.5:
            scores[RouteMode.ADVANCED_KNOWLEDGE] += 0.8
            extras.append("heuristic:inline_code_backtick")

        return scores, extras

    def _select_mode(
        self, scores: dict[RouteMode, float]
    ) -> tuple[RouteMode, float]:
        """Pick the winning mode and compute a normalised confidence.

        Args:
            scores: Per-mode accumulated scores.

        Returns:
            2-tuple ``(winning_mode, confidence)`` where confidence is in
            ``[0.0, 1.0]``.
        """
        personal_score: float = scores[RouteMode.PERSONAL_MEMORY]
        advanced_score: float = scores[RouteMode.ADVANCED_KNOWLEDGE]
        chat_score: float = scores[RouteMode.GENERAL_CHAT]

        # Apply eligibility thresholds — modes below threshold are zeroed out
        # for winner selection (but raw scores are preserved in RoutingDecision)
        effective: dict[RouteMode, float] = {
            RouteMode.ADVANCED_KNOWLEDGE: advanced_score if advanced_score >= self.advanced_threshold else 0.0,
            RouteMode.PERSONAL_MEMORY: personal_score,  # PERSONAL_MEMORY has no threshold
            RouteMode.GENERAL_CHAT: chat_score,        # GENERAL_CHAT has no threshold
        }

        total: float = sum(effective.values())

        if total == 0.0:
            # No signal above threshold — fall back to default
            return self.default_mode, 0.0

        # Winner is the mode with the highest effective score.
        # Tie-breaking priority: GENERAL_CHAT > PERSONAL_MEMORY > ADVANCED_KNOWLEDGE
        winner: RouteMode = max(
            effective,
            key=lambda m: (
                effective[m], 
                m == RouteMode.GENERAL_CHAT,
                m == RouteMode.PERSONAL_MEMORY
            ),
        )
        winner_score: float = effective[winner]

        # Confidence: winner's fraction of total effective score
        confidence: float = winner_score / total if total > 0 else 0.0

        return winner, min(confidence, 1.0)


# ---------------------------------------------------------------------------
# Public re-export surface
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "RouteMode",
    "RoutingDecision",
    "QueryRouter",
]
