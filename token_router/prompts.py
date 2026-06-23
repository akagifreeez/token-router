"""Prompt templates and their parsers.

Deliberately task-agnostic: the wording must work for whatever tasks drop at
kickoff. Two cheap signals power the router - a self-confidence rating (one
extra *local* call) and an optional remote verify (the v2 upgrade).
"""
from __future__ import annotations

import re
from typing import Optional

# A terse system prompt keeps completions short (fewer tokens) and on-task.
SOLVE_SYSTEM = (
    "You are a precise assistant. Answer the task directly and concisely. "
    "Give only the answer with no preamble, explanation, or restating the question."
)

_SELF_RATE_TEMPLATE = (
    "You just answered a task. Rate how confident you are that your answer is "
    "fully correct, as a single integer from 0 to 100.\n"
    "Reply with ONLY the integer.\n\n"
    "TASK:\n{task}\n\nYOUR ANSWER:\n{answer}\n\nCONFIDENCE (0-100):"
)

_VERIFY_TEMPLATE = (
    "Check whether the candidate answer to the task is correct.\n"
    "If it is correct, reply with exactly: OK\n"
    "If it is wrong or incomplete, reply with the corrected answer ONLY "
    "(no preamble, no explanation).\n\n"
    "TASK:\n{task}\n\nCANDIDATE ANSWER:\n{answer}\n\nVERDICT:"
)


def self_rate_prompt(task_input: str, answer: str) -> str:
    return _SELF_RATE_TEMPLATE.format(task=task_input, answer=answer)


def verify_prompt(task_input: str, answer: str) -> str:
    return _VERIFY_TEMPLATE.format(task=task_input, answer=answer)


def parse_confidence(text: str) -> Optional[int]:
    """Extract a 0-100 confidence integer from a self-rating reply.

    Returns None if no number is found (the router treats "no signal" as low
    confidence and escalates - bias to safety).
    """
    if not text:
        return None
    m = re.search(r"\d{1,3}", text)
    if not m:
        return None
    val = int(m.group(0))
    return max(0, min(100, val))


def parse_verify(text: str) -> "tuple[bool, str]":
    """Parse a verify reply into ``(was_ok, corrected_answer)``.

    ``was_ok`` True means the candidate was accepted; otherwise the returned
    string is the remote model's corrected answer.
    """
    stripped = (text or "").strip()
    if stripped.upper().startswith("OK") and len(stripped) <= 4:
        return True, ""
    return False, stripped
