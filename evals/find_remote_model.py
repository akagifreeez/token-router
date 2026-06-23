"""Discover a working Fireworks serverless model id.

Fireworks serverless model ids drift over time, and the OpenAI-compatible
``/v1/models`` listing can 500. This probes a candidate list with a 1-token call
and prints which ones actually work, so picking a valid ``--remote-model`` is
never guesswork.

    set FIREWORKS_API_KEY=...        # (PowerShell: $env:FIREWORKS_API_KEY='...')
    python evals/find_remote_model.py
"""
from __future__ import annotations

import os
import sys

import requests

BASE = "https://api.fireworks.ai/inference/v1"

CANDIDATES = [
    "accounts/fireworks/models/gpt-oss-20b",
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/qwen2p5-7b-instruct",
    "accounts/fireworks/models/qwen3-8b",
    "accounts/fireworks/models/deepseek-v3",
    "accounts/fireworks/models/mixtral-8x7b-instruct",
]


def probe(model: str, key: str) -> int:
    try:
        r = requests.post(
            BASE + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1, "temperature": 0},
            timeout=30,
        )
        return r.status_code
    except Exception:  # noqa: BLE001
        return -1


def main(argv=None) -> int:
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 2
    extra = list(argv or [])
    working = []
    for model in extra + CANDIDATES:
        code = probe(model, key)
        tag = "OK" if code == 200 else str(code)
        print(f"  [{tag}] {model}")
        if code == 200:
            working.append(model)
    print("\nWORKING:", working or "(none - check key/account access)")
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
