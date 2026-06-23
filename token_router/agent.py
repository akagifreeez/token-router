"""The agent: run each task through the router, accounting as it goes.

Thin glue - the interesting logic lives in ``RouterPolicy``. The agent loops
tasks, asks the router for a decision (which records its own model calls into
the shared ``Ledger``), and assembles a ``Result`` with the per-task token/cost
totals pulled back out of the ledger.
"""
from __future__ import annotations

from typing import List, Optional

from .accounting import Ledger
from .models.base import Model
from .router import RouterPolicy
from .tasks import Result, Task


class Agent:
    def __init__(
        self,
        local: Model,
        remote: Model,
        *,
        router: Optional[RouterPolicy] = None,
        ledger: Optional[Ledger] = None,
        warmup: bool = True,
    ) -> None:
        self.local = local
        self.remote = remote
        self.router = router or RouterPolicy()
        self.ledger = ledger or Ledger()
        if warmup:
            # Fail fast on a misconfigured local backend rather than mid-run.
            self.local.warmup()

    def run_task(self, task: Task) -> Result:
        decision = self.router.route(task, self.local, self.remote, self.ledger)
        rows = [r for r in self.ledger.rows if r.task_id == task.id]
        return Result(
            id=task.id,
            answer=decision.answer,
            route=decision.route,
            category=task.category,
            confidence=decision.confidence,
            total_tokens=sum(r.total_tokens for r in rows),
            cost_usd=round(sum(r.cost_usd for r in rows), 8),
        )

    def run(self, tasks: List[Task]) -> List[Result]:
        return [self.run_task(t) for t in tasks]
