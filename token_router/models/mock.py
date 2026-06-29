"""A deterministic ``Model`` for offline tests and local development.

``ScriptedModel`` lets a test fix the exact text a "model" returns for a given
prompt, with no network and no real model - the analog of how ``hl-read`` tests
patch the SDK ``Info`` class. It records every call so tests can assert *which*
backend was hit and how often (e.g. "the remote model was never called").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from .base import Model, Usage, count_tokens

# A response spec is either a callable(prompt, **kw) -> str, or a dict that maps
# an exact prompt (or a substring) to the answer text.
Responder = Union[Callable[..., str], Dict[str, str]]


@dataclass
class Call:
    """One recorded invocation, for test assertions."""

    prompt: str
    kwargs: dict


class ScriptedModel(Model):
    """A canned, deterministic model backend.

    ``responses`` may be:

    * a ``callable(prompt, **kw) -> str`` - full control, or
    * a ``dict`` - matched first by exact prompt, then by substring containment;
      falls back to ``default``.

    Token counts are derived deterministically from the text via ``count_tokens``
    (or you can pin them with ``prompt_tokens``/``completion_tokens``).
    """

    def __init__(
        self,
        responses: Optional[Responder] = None,
        *,
        name: str = "scripted",
        backend: str = "local",
        default: str = "",
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.name = name
        self.backend = backend
        self._responses = responses
        self._default = default
        self._pt = prompt_tokens
        self._ct = completion_tokens
        self._latency = latency_ms
        self.calls: List[Call] = []
        self.warmups = 0

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _answer(self, prompt: str, **kw: object) -> str:
        r = self._responses
        if r is None:
            return self._default
        if callable(r):
            return r(prompt, **kw)
        if prompt in r:
            return r[prompt]
        for key, val in r.items():
            if key in prompt:
                return val
        return self._default

    def complete(self, prompt: str, **kw: object) -> Tuple[str, Usage]:
        self.calls.append(Call(prompt=prompt, kwargs=dict(kw)))
        text = self._answer(prompt, **kw)
        usage = Usage(
            prompt_tokens=self._pt if self._pt is not None else count_tokens(prompt, self.name),
            completion_tokens=self._ct if self._ct is not None else count_tokens(text, self.name),
            model=self.name,
            backend=self.backend,
            latency_ms=self._latency,
        )
        return text, usage

    def warmup(self) -> None:
        self.warmups += 1
