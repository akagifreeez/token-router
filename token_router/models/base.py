"""The uniform model interface every backend honors.

The router and agent depend ONLY on ``Model`` and ``Usage`` - never on a
concrete backend - so a local model, a remote API, and a scripted test double
are interchangeable. Every ``complete()`` returns ``(text, Usage)`` and Usage is
always populated (token counts fall back to an estimate when a backend doesn't
report them), so cost accounting can never silently drift from reality.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


class ModelError(RuntimeError):
    """A model call failed after exhausting retries (transient errors)."""


@dataclass(frozen=True)
class Usage:
    """Token/latency accounting for a single ``complete()`` call.

    ``backend`` is ``"local"`` or ``"fireworks"`` and drives both pricing and the
    leaderboard's local-vs-remote split. ``estimated`` is True when the token
    counts came from the heuristic fallback rather than the backend itself.
    """

    prompt_tokens: int
    completion_tokens: int
    model: str
    backend: str
    latency_ms: float = 0.0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


def count_tokens(text: str, model: str = "") -> int:
    """Cheap, dependency-free token estimate (~4 chars/token).

    Only used as a *fallback* when a backend does not report real usage, so that
    ``Usage`` is never empty. Deliberately conservative and rounds up, so we
    never under-count tokens in the cost ledger.
    """
    if not text:
        return 0
    # ~4 chars per token is the standard rough rule for English-ish text; round
    # up so the estimate is never zero for non-empty text.
    return max(1, (len(text) + 3) // 4)


class Model(abc.ABC):
    """A text-completion backend.

    Implementations: ``LocalModel`` (Ollama), ``FireworksModel`` (remote API),
    and ``ScriptedModel`` (offline test double). ``name`` is the model id used
    for the pricing table; ``backend`` is ``"local"`` or ``"fireworks"``.
    """

    name: str
    backend: str

    @abc.abstractmethod
    def complete(self, prompt: str, **kw: object) -> "tuple[str, Usage]":
        """Return ``(text, Usage)`` for ``prompt``.

        Keyword args carry generation params (``max_tokens``, ``temperature``,
        ``stop``, ``system``). Implementations must always return a populated
        ``Usage`` and raise ``ModelError`` on unrecoverable failure.
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: make a tiny call so the first real task isn't slow.

        Default no-op; backends with cold-start cost (local models) override it.
        Must surface a misconfiguration by raising (fail fast).
        """
        return None
