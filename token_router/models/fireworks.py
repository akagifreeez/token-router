"""Remote model backend: the Fireworks AI API.

Fireworks exposes an OpenAI-compatible chat-completions REST endpoint, so this
is a thin ``requests`` client - no vendor SDK - wrapped in the shared
``ResilientClient`` retry/backoff/rate-limit/cache layer (ported from hl-read).
The API key is read from the environment and never logged or committed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, List, Optional, Tuple

import requests

from .base import Model, ModelError, Usage, count_tokens
from ._http import HttpError, ResilientClient

# Fireworks serverless model ids drift; this is the verified-working default for
# the current account. Use evals/find_remote_model.py to rediscover if it 404s.
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-20b"
DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksModel(Model, ResilientClient):
    """A ``Model`` backed by the Fireworks AI chat-completions API.

    ``api_key`` falls back to ``$FIREWORKS_API_KEY``. ``fallback_urls`` are tried
    in order if the primary host fails after exhausting its retries. All the
    resilience knobs (``max_retries``/``backoff_*``/``rate_limit_per_min``/
    ``cache_ttl``/``http_timeout``) come from ``ResilientClient``.
    """

    backend = "fireworks"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        fallback_urls: Optional[List[str]] = None,
        max_retries: int = 4,
        backoff_base: float = 0.4,
        backoff_max: float = 8.0,
        rate_limit_per_min: Optional[float] = None,
        cache_ttl: float = 0.0,
        http_timeout: Optional[float] = 60.0,
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
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        primary = base_url.rstrip("/")
        self._urls = [primary] + [u.rstrip("/") for u in (fallback_urls or []) if u]
        self.base_url = primary
        self._session = session or requests.Session()

    # -- wire ------------------------------------------------------------

    def _post_chat(self, body: dict, url: str) -> dict:
        """One POST to ``{url}/chat/completions``; raises on non-2xx."""
        if not self.api_key:
            raise ModelError("FIREWORKS_API_KEY is not set (no api_key provided)")
        resp = self._session.post(
            f"{url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body),
            timeout=self.http_timeout,
        )
        if resp.status_code >= 400:
            # Carry the status code so _is_transient can retry 429/5xx and let
            # real 4xx (bad request / auth) fail fast.
            snippet = (resp.text or "")[:200]
            raise HttpError(f"HTTP {resp.status_code} from Fireworks: {snippet}", resp.status_code)
        return resp.json()

    @staticmethod
    def _messages(prompt: str, system: Optional[str]) -> List[dict]:
        msgs: List[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    @staticmethod
    def _cache_key(model: str, body: dict) -> str:
        blob = json.dumps([model, body], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- Model interface -------------------------------------------------

    def complete(self, prompt: str, **kw: Any) -> Tuple[str, Usage]:
        body = {
            "model": self.name,
            "messages": self._messages(prompt, kw.get("system")),
            "max_tokens": int(kw.get("max_tokens", 512)),
            "temperature": float(kw.get("temperature", 0.0)),
        }
        if kw.get("stop"):
            body["stop"] = kw["stop"]

        key = self._cache_key(self.name, body)

        def produce() -> Tuple[str, Usage]:
            t0 = time.monotonic()
            data = self._failover_post(body)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            return self._parse(prompt, data, latency_ms)

        return self._cached(key, self.cache_ttl, produce)

    def _failover_post(self, body: dict) -> dict:
        """Try each configured host in turn; each host gets the full retry budget."""
        last_err: Optional[Exception] = None
        for url in self._urls:
            try:
                return self._call(self._post_chat, body, url)
            except ModelError as e:
                last_err = e
                continue
        raise ModelError(
            f"Fireworks request failed across {len(self._urls)} host(s): {last_err}"
        ) from last_err

    def _parse(self, prompt: str, data: dict, latency_ms: float) -> Tuple[str, Usage]:
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"unexpected Fireworks response shape: {e}") from e
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
