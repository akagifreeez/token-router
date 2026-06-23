"""Token -> cost conversion.

Prices are USD per 1,000,000 tokens (the unit Fireworks quotes). Local
inference has no API cost, so any ``backend == "local"`` usage costs $0 - but we
still count its tokens, because the leaderboard metric may be *total tokens*,
not just dollars. The Fireworks numbers here are placeholders to be confirmed
from the pricing page at kickoff (``Pricing.override`` makes that a one-liner).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# model id -> (prompt_usd_per_1M, completion_usd_per_1M). Placeholder Fireworks
# figures; finalize at kickoff. Keyed by the exact model id used in requests.
DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    # Placeholder figures; confirm on the Fireworks pricing page at kickoff.
    "accounts/fireworks/models/gpt-oss-20b": (0.10, 0.40),
    "accounts/fireworks/models/llama-v3p1-8b-instruct": (0.20, 0.20),
    "accounts/fireworks/models/llama-v3p1-70b-instruct": (0.90, 0.90),
    "accounts/fireworks/models/qwen2p5-72b-instruct": (0.90, 0.90),
}

# Used when a fireworks model isn't in the table, so cost is never silently 0.
_FALLBACK_FIREWORKS = (0.50, 0.50)


class Pricing:
    """Looks up per-token cost for a usage record."""

    def __init__(self, prices: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self._prices: Dict[str, Tuple[float, float]] = dict(DEFAULT_PRICES)
        if prices:
            self._prices.update(prices)

    def override(self, model: str, prompt_per_1m: float, completion_per_1m: float) -> None:
        """Set/replace the price for one model (call at kickoff with real figures)."""
        self._prices[model] = (float(prompt_per_1m), float(completion_per_1m))

    def cost_usd(self, *, backend: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        if backend == "local":
            return 0.0
        p_in, p_out = self._prices.get(model, _FALLBACK_FIREWORKS)
        return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000.0
