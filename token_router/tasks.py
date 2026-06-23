"""Task and Result models + JSONL/JSON loaders and writers.

This module is the **task-adapter seam**: the generic loaders here read a simple
``{id, input, ...}`` shape today, and at kickoff you only adjust the parsing in
``load_tasks`` (and the output shape in ``write_results``) to match the official
task format - the router/agent never change.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    """One unit of work. ``gold``/``allowed_labels`` are for offline eval only;
    real kickoff tasks won't carry them."""

    id: str
    input: str
    category: str = "general"
    gold: Optional[str] = None
    allowed_labels: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """The agent's answer for one task plus the routing trace."""

    id: str
    answer: str
    route: str            # "local" | "remote" | "remote(forced)" | ...
    category: str = "general"
    confidence: Optional[int] = None
    total_tokens: int = 0
    cost_usd: float = 0.0


def _coerce_task(obj: Dict[str, Any], idx: int) -> Task:
    # Tolerant of common field aliases so a new input format is a small tweak.
    tid = str(obj.get("id", obj.get("task_id", idx)))
    text = obj.get("input", obj.get("prompt", obj.get("question", obj.get("text", ""))))
    return Task(
        id=tid,
        input=str(text),
        category=str(obj.get("category", obj.get("type", "general"))),
        gold=obj.get("gold", obj.get("answer", obj.get("expected"))),
        allowed_labels=obj.get("allowed_labels") or obj.get("labels"),
        meta=obj.get("meta", {}) or {},
    )


def load_tasks(path: str) -> List[Task]:
    """Load tasks from JSONL (one object per line) or a JSON array."""
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()
    if not content:
        return []
    tasks: List[Task] = []
    if content[0] == "[":
        for i, obj in enumerate(json.loads(content)):
            tasks.append(_coerce_task(obj, i))
    else:
        for i, line in enumerate(content.splitlines()):
            line = line.strip()
            if line:
                tasks.append(_coerce_task(json.loads(line), i))
    return tasks


def write_results(path: str, results: List[Result]) -> None:
    """Write results as JSONL (one object per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
