"""Offline tests for the routing policy, using ScriptedModel for both tiers.

Every decision path is checked with assertions on *which* backend was called
(e.g. "the remote model was never called" when the local answer is kept).
"""
from __future__ import annotations

import unittest

from token_router.accounting import Ledger
from token_router.models.mock import ScriptedModel
from token_router.router import RouterConfig, RouterPolicy
from token_router.tasks import Task

CONF = "CONFIDENCE (0-100)"


def local_with_confidence(answer, confidence):
    def responder(prompt, **kw):
        return str(confidence) if CONF in prompt else answer
    return ScriptedModel(responder, name="mock-local", backend="local")


def remote_fixed(answer):
    return ScriptedModel({}, name="mock-remote", backend="fireworks", default=answer)


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.remote = remote_fixed("REMOTE-ANSWER")
        self.ledger = Ledger()

    def _route(self, task, local, config=None):
        policy = RouterPolicy(config or RouterConfig(local_keep_threshold=70))
        return policy.route(task, local, self.remote, self.ledger)

    def test_high_confidence_keeps_local(self):
        local = local_with_confidence("Paris", 95)
        task = Task(id="1", input="What is the capital of France?", category="qa")
        d = self._route(task, local)
        self.assertEqual(d.route, "local")
        self.assertEqual(d.answer, "Paris")
        self.assertEqual(self.remote.call_count, 0)
        self.assertEqual(d.confidence, 95)

    def test_low_confidence_escalates(self):
        local = local_with_confidence("Paris?", 40)
        task = Task(id="2", input="What is the capital of France?", category="qa")
        d = self._route(task, local)
        self.assertEqual(d.route, "remote(escalated)")
        self.assertEqual(d.answer, "REMOTE-ANSWER")
        self.assertEqual(self.remote.call_count, 1)

    def test_format_failure_escalates_without_rating(self):
        # Answer is not one of the allowed labels -> escalate before self-rating.
        local = local_with_confidence("banana", 99)
        task = Task(
            id="3",
            input="Classify sentiment: 'I love it'",
            category="classification",
            allowed_labels=["positive", "negative", "neutral"],
        )
        d = self._route(task, local)
        self.assertEqual(d.route, "remote(escalated)")
        self.assertEqual(self.remote.call_count, 1)

    def test_hard_task_skips_local(self):
        local = local_with_confidence("...", 99)
        long_reasoning = "Explain why " + "the system behaves this way. " * 30
        task = Task(id="4", input=long_reasoning, category="reasoning")
        d = self._route(task, local)
        self.assertEqual(d.route, "remote(hard)")
        self.assertEqual(d.difficulty, "hard")
        self.assertEqual(local.call_count, 0)  # never solved locally
        self.assertEqual(self.remote.call_count, 1)

    def test_safety_mode_forces_remote(self):
        local = local_with_confidence("Paris", 99)
        task = Task(id="5", input="What is the capital of France?", category="qa")
        d = self._route(task, local, RouterConfig(safety_mode=True))
        self.assertEqual(d.route, "remote(forced)")
        self.assertEqual(local.call_count, 0)
        self.assertEqual(self.remote.call_count, 1)

    def test_no_self_rate_keeps_local_by_default(self):
        local = ScriptedModel({}, name="mock-local", backend="local", default="Paris")
        task = Task(id="6", input="What is the capital of France?", category="qa")
        d = self._route(task, local, RouterConfig(use_self_rate=False))
        self.assertEqual(d.route, "local")
        self.assertEqual(self.remote.call_count, 0)
        # exactly one local call (the solve), no rating call
        self.assertEqual(local.call_count, 1)

    def test_verify_ok_keeps_local_answer(self):
        local = local_with_confidence("Paris", 40)  # low conf -> escalate to verify
        remote = ScriptedModel(
            lambda prompt, **kw: "OK", name="mock-remote", backend="fireworks"
        )
        task = Task(id="7", input="What is the capital of France?", category="qa")
        policy = RouterPolicy(RouterConfig(local_keep_threshold=70, use_verify=True))
        d = policy.route(task, local, remote, self.ledger)
        self.assertEqual(d.route, "remote(verified)")
        self.assertEqual(d.answer, "Paris")  # kept local answer, remote only confirmed
        self.assertEqual(remote.call_count, 1)


if __name__ == "__main__":
    unittest.main()
