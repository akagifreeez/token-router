"""Local model backend: an Ollama server.

Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so this
is the same thin ``requests`` + ``ResilientClient`` shape as the Fireworks
backend (the local server can be transiently busy or cold-starting, so it
benefits from the same retry layer). Local inference has no API cost, so the
router treats it as the cheap tier; we still measure and report local tokens
because the leaderboard may count total tokens, not just dollars.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, List, Optional, Tuple

import requests

from .base import Model, ModelError, Usage, count_tokens
from ._http import HttpError, ResilientClient

DEFAULT_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:3b-instruct")


def _default_base_url() -> str:
    # OLLAMA_BASE_URL wins; else derive from OLLAMA_HOST; else localhost.
    explicit = os.environ.get("OLLAMA_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/") + "/v1"


class LocalModel(Model, ResilientClient):
    """A ``Model`` backed by a local Ollama server."""

    backend = "local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: Optional[str] = None,
        max_retries: int = 2,
        backoff_base: float = 0.3,
        backoff_max: float = 4.0,
        rate_limit_per_min: Optional[float] = None,
        cache_ttl: float = 0.0,
        http_timeout: Optional[float] = 120.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        ResilientClient.__init__(
            self,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            rate_limit_per_min=rate_limit_per_min,
            cache_ttl=cache_ttl,
            http_timeout=http_timeout,
        )
        self.name = model
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self._session = session or requests.Session()

    def _post_chat(self, body: dict) -> dict:
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=self.http_timeout,
        )
        if resp.status_code >= 400:
            snippet = (resp.text or "")[:200]
            raise HttpError(f"HTTP {resp.status_code} from Ollama: {snippet}", resp.status_code)
        return resp.json()

    @staticmethod
    def _messages(prompt: str, system: Optional[str]) -> List[dict]:
        msgs: List[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def complete(self, prompt: str, **kw: Any) -> Tuple[str, Usage]:
        body = {
            "model": self.name,
            "messages": self._messages(prompt, kw.get("system")),
            "max_tokens": int(kw.get("max_tokens", 512)),
            "temperature": float(kw.get("temperature", 0.0)),
            "stream": False,
        }
        if kw.get("stop"):
            body["stop"] = kw["stop"]
        t0 = time.monotonic()
        data = self._call(self._post_chat, body)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"unexpected Ollama response shape: {e}") from e
        u = data.get("usage") or {}
        pt = u.get("prompt_tokens")
        ct = u.get("completion_tokens")
        estimated = pt is None or ct is None
        usage = Usage(
            prompt_tokens=int(pt) if pt is not None else count_tokens(prompt, self.name),
            completion_tokens=int(ct) if ct is not None else count_tokens(text, self.name),
            model=self.name,
            backend=self.backend,
            latency_ms=latency_ms,
            estimated=estimated,
        )
        return text, usage

    def warmup(self) -> None:
        """Load the model into memory and surface a misconfig immediately."""
        self.complete("ok", max_tokens=1)
