"""Offline tests for the cost ledger and pricing."""
from __future__ import annotations

import unittest

from token_router.accounting import Ledger
from token_router.models.base import Usage
from token_router.pricing import Pricing

FW_8B = "accounts/fireworks/models/llama-v3p1-8b-instruct"


class TestLedger(unittest.TestCase):
    def test_summary_splits_and_keep_rate(self):
        led = Ledger(pricing=Pricing())
        led.record("t1", "local", Usage(10, 5, "qwen2.5:3b-instruct", "local"))
        led.record("t1", "local_rate", Usage(3, 1, "qwen2.5:3b-instruct", "local"))
        led.record("t2", "remote", Usage(20, 10, FW_8B, "fireworks"))
        s = led.summary()
        self.assertEqual(s["tasks"], 2)
        self.assertEqual(s["kept_local_tasks"], 1)
        self.assertEqual(s["escalated_tasks"], 1)
        self.assertEqual(s["local_keep_rate"], 0.5)
        self.assertEqual(s["total_tokens"], 10 + 5 + 3 + 1 + 20 + 10)

    def test_cost_math(self):
        led = Ledger(pricing=Pricing())
        # local is free
        led.record("t1", "local", Usage(100, 100, "qwen2.5:3b-instruct", "local"))
        self.assertEqual(led.summary()["est_cost_usd"], 0.0)
        # fireworks 8b = (0.20, 0.20) per 1M: (20*0.2 + 10*0.2)/1e6 = 6e-6
        led.record("t2", "remote", Usage(20, 10, FW_8B, "fireworks"))
        self.assertAlmostEqual(led.summary()["est_cost_usd"], 6e-6, places=12)

    def test_unknown_model_uses_fallback_not_zero(self):
        led = Ledger(pricing=Pricing())
        led.record("t1", "remote", Usage(1_000_000, 0, "accounts/fireworks/models/unknown", "fireworks"))
        self.assertGreater(led.summary()["est_cost_usd"], 0.0)

    def test_format_summary_runs(self):
        led = Ledger(pricing=Pricing())
        led.record("t1", "local", Usage(10, 5, "qwen2.5:3b-instruct", "local"))
        text = led.format_summary()
        self.assertIn("token-router run summary", text)
        self.assertIn("local", text)


if __name__ == "__main__":
    unittest.main()
