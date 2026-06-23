"""The routing policy - the competitive heart of the agent.

Principle: **local is ~free, remote is the budget.** So we run a cost-ordered
cascade - try the cheap local model first, and escalate to the strong remote
model only when a cheap signal says the local answer is unreliable. On any
ambiguity we bias toward escalation (spend tokens for safety), so worst case the
agent degrades toward "all-remote" and stays above the accuracy floor.

v1 (default, guaranteed shippable): difficulty pre-screen -> local solve ->
confidence gate (self-rating / format check / optional agreement) -> escalate.
v2 (opt-in flags): remote *verify* instead of re-solve, and per-category
thresholds. Enabling v2 is a config change, not a rewrite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .accounting import Ledger
from .models.base import Model
from .prompts import (
    SOLVE_SYSTEM,
    parse_confidence,
    parse_verify,
    self_rate_prompt,
    verify_prompt,
)
from .tasks import Task

_REASON_MARKERS = (
    "why",
    "explain",
    "step by step",
    "prove",
    "derive",
    "calculate",
    "reason",
    "compare",
    "analyze",
)


def _norm(text: str) -> str:
    """Loose normalization for comparing two answers (case/space/punct-folded)."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


@dataclass
class RouteDecision:
    """The outcome of routing one task."""

    answer: str
    route: str            # local | remote(forced|hard|escalated|verified|verify)
    confidence: Optional[int] = None
    difficulty: str = "medium"
    trace: List[str] = field(default_factory=list)


@dataclass
class RouterConfig:
    """All routing knobs. Defaults = the shippable v1 policy."""

    local_keep_threshold: int = 70           # keep local answer if confidence >= this
    use_self_rate: bool = True               # one extra cheap local call for confidence
    use_agreement: bool = False              # second local sample; disagreement => escalate
    hard_difficulty_to_remote: bool = True   # skip a wasted local call on clearly-hard tasks
    safety_mode: bool = False                # force every task to remote (panic button)
    solve_max_tokens: int = 512
    rate_max_tokens: int = 8
    agreement_temperature: float = 0.6
    # v2 upgrades (off by default):
    use_verify: bool = False                 # remote verifies local answer instead of re-solving
    per_category: Optional[Dict[str, int]] = None  # category -> threshold override


class RouterPolicy:
    """Decides local vs remote per task and produces the answer + trace.

    ``route()`` performs the model calls and records every one into ``ledger``,
    so the cost numbers always reflect exactly what ran.
    """

    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        self.config = config or RouterConfig()

    # -- heuristics ------------------------------------------------------

    def _difficulty(self, text: str) -> str:
        n = len(text)
        low = text.lower()
        has_marker = any(m in low for m in _REASON_MARKERS)
        multi_q = low.count("?") >= 2
        if n > 1200 or (has_marker and n > 400) or (multi_q and n > 600):
            return "hard"
        if n < 200 and not has_marker:
            return "easy"
        return "medium"

    def _threshold(self, category: str) -> int:
        pc = self.config.per_category
        if pc and category in pc:
            return int(pc[category])
        return int(self.config.local_keep_threshold)

    @staticmethod
    def _format_ok(task: Task, answer: str) -> bool:
        """For label tasks, the answer must match an allowed label; else accept."""
        if not answer or not answer.strip():
            return False
        if task.allowed_labels:
            na = _norm(answer)
            return any(_norm(lbl) in na or na in _norm(lbl) for lbl in task.allowed_labels)
        return True

    # -- the cascade -----------------------------------------------------

    def route(self, task: Task, local: Model, remote: Model, ledger: Ledger) -> RouteDecision:
        cfg = self.config
        trace: List[str] = []

        if cfg.safety_mode:
            ans, u = remote.complete(task.input, system=SOLVE_SYSTEM, max_tokens=cfg.solve_max_tokens)
            ledger.record(task.id, "remote", u)
            trace.append("safety_mode -> remote")
            return RouteDecision(ans, "remote(forced)", None, "n/a", trace)

        difficulty = self._difficulty(task.input)
        trace.append(f"difficulty={difficulty}")

        if cfg.hard_difficulty_to_remote and difficulty == "hard":
            ans, u = remote.complete(task.input, system=SOLVE_SYSTEM, max_tokens=cfg.solve_max_tokens)
            ledger.record(task.id, "remote", u)
            trace.append("hard -> remote (skipped local)")
            return RouteDecision(ans, "remote(hard)", None, difficulty, trace)

        # Cheap path: solve locally.
        local_ans, lu = local.complete(task.input, system=SOLVE_SYSTEM, max_tokens=cfg.solve_max_tokens)
        ledger.record(task.id, "local", lu)

        escalate, confidence = self._should_escalate(task, local, local_ans, ledger, trace)

        if not escalate:
            trace.append("kept local")
            return RouteDecision(local_ans, "local", confidence, difficulty, trace)

        # Escalation path.
        if cfg.use_verify:
            vp = verify_prompt(task.input, local_ans)
            vtext, vu = remote.complete(vp, max_tokens=cfg.solve_max_tokens)
            ledger.record(task.id, "verify", vu)
            ok, corrected = parse_verify(vtext)
            if ok:
                trace.append("remote verify: OK (kept local answer)")
                return RouteDecision(local_ans, "remote(verified)", confidence, difficulty, trace)
            trace.append("remote verify: corrected")
            return RouteDecision(corrected or local_ans, "remote(verify)", confidence, difficulty, trace)

        ans, u = remote.complete(task.input, system=SOLVE_SYSTEM, max_tokens=cfg.solve_max_tokens)
        ledger.record(task.id, "remote", u)
        trace.append("escalated -> remote")
        return RouteDecision(ans, "remote(escalated)", confidence, difficulty, trace)

    def _should_escalate(
        self, task: Task, local: Model, local_ans: str, ledger: Ledger, trace: List[str]
    ) -> "tuple[bool, Optional[int]]":
        """Decide whether to escalate, returning ``(escalate, confidence)``.

        Bias to safety: a failed format check, an unparseable rating, or a
        disagreement all force escalation.
        """
        cfg = self.config

        if not self._format_ok(task, local_ans):
            trace.append("format check failed -> escalate")
            return True, None

        if cfg.use_agreement:
            alt, au = local.complete(
                task.input, system=SOLVE_SYSTEM,
                max_tokens=cfg.solve_max_tokens, temperature=cfg.agreement_temperature,
            )
            ledger.record(task.id, "local_agree", au)
            if _norm(alt) != _norm(local_ans):
                trace.append("local self-disagreement -> escalate")
                return True, None
            trace.append("local self-agreement")

        if not cfg.use_self_rate:
            trace.append("no self-rate; kept local by default")
            return False, None

        rate_text, ru = local.complete(
            self_rate_prompt(task.input, local_ans), max_tokens=cfg.rate_max_tokens
        )
        ledger.record(task.id, "local_rate", ru)
        confidence = parse_confidence(rate_text)
        threshold = self._threshold(task.category)
        if confidence is None:
            trace.append("confidence unparseable -> escalate")
            return True, None
        trace.append(f"confidence={confidence} threshold={threshold}")
        return confidence < threshold, confidence
