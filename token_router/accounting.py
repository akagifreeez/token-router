"""The cost ledger - the single source of truth that mirrors the leaderboard.

Every model call (local or remote, at any router stage) is recorded here, so the
totals can never drift from what actually happened. Rows are streamed to a JSONL
file as they happen, so a crash mid-run still leaves an inspectable partial
record. ``summary()`` produces exactly the numbers we optimize: total tokens,
estimated cost, and the local-vs-remote split.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .models.base import Usage
from .pricing import Pricing


@dataclass
class LedgerRow:
    task_id: str
    stage: str          # "local" | "local_rate" | "remote" | "verify" | ...
    backend: str        # "local" | "fireworks"
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    estimated: bool


class Ledger:
    """Records per-call usage and produces leaderboard-shaped summaries."""

    def __init__(self, pricing: Optional[Pricing] = None, jsonl_path: Optional[str] = None) -> None:
        self.pricing = pricing or Pricing()
        self.jsonl_path = jsonl_path
        self.rows: List[LedgerRow] = []
        self._fh = None
        if jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)
            self._fh = open(jsonl_path, "w", encoding="utf-8")

    def record(self, task_id: str, stage: str, usage: Usage) -> LedgerRow:
        cost = self.pricing.cost_usd(
            backend=usage.backend,
            model=usage.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        row = LedgerRow(
            task_id=task_id,
            stage=stage,
            backend=usage.backend,
            model=usage.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=round(cost, 8),
            latency_ms=usage.latency_ms,
            estimated=usage.estimated,
        )
        self.rows.append(row)
        if self._fh is not None:
            self._fh.write(json.dumps(asdict(row)) + "\n")
            self._fh.flush()
        return row

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- summaries -------------------------------------------------------

    def summary(self) -> dict:
        task_ids = {r.task_id for r in self.rows}
        backends = sorted({r.backend for r in self.rows})
        by_backend: Dict[str, dict] = {}
        for b in backends:
            rs = [r for r in self.rows if r.backend == b]
            by_backend[b] = {
                "calls": len(rs),
                "prompt_tokens": sum(r.prompt_tokens for r in rs),
                "completion_tokens": sum(r.completion_tokens for r in rs),
                "total_tokens": sum(r.total_tokens for r in rs),
                "cost_usd": round(sum(r.cost_usd for r in rs), 6),
            }

        # A task is "kept local" if it never hit the fireworks backend.
        remote_tasks = {r.task_id for r in self.rows if r.backend == "fireworks"}
        n_tasks = len(task_ids)
        kept_local = n_tasks - len(remote_tasks)
        latencies = [r.latency_ms for r in self.rows if r.latency_ms]

        return {
            "tasks": n_tasks,
            "calls": len(self.rows),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self.rows),
            "total_completion_tokens": sum(r.completion_tokens for r in self.rows),
            "total_tokens": sum(r.total_tokens for r in self.rows),
            "est_cost_usd": round(sum(r.cost_usd for r in self.rows), 6),
            "kept_local_tasks": kept_local,
            "escalated_tasks": len(remote_tasks),
            "local_keep_rate": round(kept_local / n_tasks, 4) if n_tasks else 0.0,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "by_backend": by_backend,
        }

    def format_summary(self) -> str:
        s = self.summary()
        lines = [
            "=== token-router run summary ===",
            f"tasks={s['tasks']}  calls={s['calls']}  "
            f"kept_local={s['kept_local_tasks']}  escalated={s['escalated_tasks']}  "
            f"local_keep_rate={s['local_keep_rate']:.1%}",
            f"total_tokens={s['total_tokens']:,}  "
            f"(prompt={s['total_prompt_tokens']:,}  completion={s['total_completion_tokens']:,})",
            f"est_cost_usd=${s['est_cost_usd']:.6f}  mean_latency_ms={s['mean_latency_ms']}",
        ]
        for b, d in s["by_backend"].items():
            lines.append(
                f"  [{b}] calls={d['calls']}  tokens={d['total_tokens']:,}  cost=${d['cost_usd']:.6f}"
            )
        return "\n".join(lines)
