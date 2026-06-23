"""Offline end-to-end test of the Agent over two tasks with mock models."""
from __future__ import annotations

import unittest

from token_router.agent import Agent
from token_router.models.mock import ScriptedModel
from token_router.tasks import Task

CONF = "CONFIDENCE (0-100)"


def _local():
    def responder(prompt, **kw):
        return "90" if CONF in prompt else "an answer"
    return ScriptedModel(responder, name="mock-local", backend="local")


class TestAgent(unittest.TestCase):
    def test_runs_tasks_and_accounts(self):
        local = _local()
        remote = ScriptedModel({}, name="mock-remote", backend="fireworks", default="R")
        agent = Agent(local, remote, warmup=True)

        self.assertEqual(local.warmups, 1)  # warmup fired

        tasks = [
            Task(id="a", input="What is the capital of France?", category="qa"),
            Task(id="b", input="What is the chemical symbol for gold?", category="qa"),
        ]
        results = agent.run(tasks)

        self.assertEqual(len(results), 2)
        self.assertEqual({r.id for r in results}, {"a", "b"})
        for r in results:
            self.assertEqual(r.route, "local")
            self.assertEqual(r.answer, "an answer")
            self.assertGreater(r.total_tokens, 0)
        self.assertEqual(remote.call_count, 0)
        self.assertGreater(len(agent.ledger.rows), 0)

    def test_warmup_can_be_skipped(self):
        local = _local()
        remote = ScriptedModel({}, name="mock-remote", backend="fireworks", default="R")
        Agent(local, remote, warmup=False)
        self.assertEqual(local.warmups, 0)


if __name__ == "__main__":
    unittest.main()
