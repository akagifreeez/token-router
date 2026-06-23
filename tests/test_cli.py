"""Offline test of the CLI run loop, with the model classes patched out."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from token_router import cli
from token_router.models.mock import ScriptedModel


def _fake_local(*args, **kwargs):
    return ScriptedModel({}, name="mock-local", backend="local", default="an answer")


def _fake_remote(*args, **kwargs):
    return ScriptedModel({}, name="mock-remote", backend="fireworks", default="R")


class TestCli(unittest.TestCase):
    def test_run_writes_results(self):
        with tempfile.TemporaryDirectory() as d:
            in_path = os.path.join(d, "tasks.jsonl")
            out_path = os.path.join(d, "results.jsonl")
            ledger_path = os.path.join(d, "ledger.jsonl")
            with open(in_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": "q1", "input": "Capital of France?", "category": "qa"}) + "\n")
                fh.write(json.dumps({"id": "q2", "input": "Symbol for gold?", "category": "qa"}) + "\n")

            with mock.patch.object(cli, "LocalModel", _fake_local), \
                 mock.patch.object(cli, "FireworksModel", _fake_remote):
                rc = cli.main([
                    "run", "--in", in_path, "--out", out_path,
                    "--ledger", ledger_path, "--no-warmup", "--no-self-rate", "--report",
                ])

            self.assertEqual(rc, 0)
            with open(out_path, "r", encoding="utf-8") as fh:
                lines = [json.loads(ln) for ln in fh if ln.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual({r["id"] for r in lines}, {"q1", "q2"})
            self.assertTrue(os.path.exists(ledger_path))

    def test_missing_tasks_file_returns_error(self):
        with tempfile.TemporaryDirectory() as d:
            in_path = os.path.join(d, "empty.jsonl")
            out_path = os.path.join(d, "out.jsonl")
            open(in_path, "w").close()  # empty
            with mock.patch.object(cli, "LocalModel", _fake_local), \
                 mock.patch.object(cli, "FireworksModel", _fake_remote):
                rc = cli.main(["run", "--in", in_path, "--out", out_path, "--no-warmup"])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
