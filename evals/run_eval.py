"""Eval harness: measure accuracy vs token cost, and sweep the keep-threshold.

Run it before kickoff to calibrate the router. Because the real tasks are
hidden, this uses a small proxy suite (``evals/proxy/*.jsonl``) spanning the
categories likely to appear, and prints the accuracy-vs-cost frontier so you can
pick the ``--threshold`` that holds accuracy comfortably above the floor.

    python evals/run_eval.py                 # full sweep, real models (needs Ollama + FIREWORKS_API_KEY)
    python evals/run_eval.py --mock          # offline smoke test of the harness itself
    python evals/run_eval.py --threshold 70  # single run at one threshold
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# Make the project root importable when run as `python evals/run_eval.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evals.scorers import score  # noqa: E402
from token_router.accounting import Ledger  # noqa: E402
from token_router.agent import Agent  # noqa: E402
from token_router.models.mock import ScriptedModel  # noqa: E402
from token_router.pricing import Pricing  # noqa: E402
from token_router.router import RouterConfig, RouterPolicy  # noqa: E402
from token_router.tasks import Task, load_tasks  # noqa: E402

PROXY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy")


def load_proxy(categories=None) -> "list[Task]":
    tasks: list[Task] = []
    for path in sorted(glob.glob(os.path.join(PROXY_DIR, "*.jsonl"))):
        cat = os.path.splitext(os.path.basename(path))[0]
        if categories and cat not in categories:
            continue
        tasks.extend(load_tasks(path))
    return tasks


def _mock_models():
    """Deterministic stand-ins so the harness runs with no network/model.

    Local self-rates high (kept local) and answers with the gold-ish token; this
    only exercises the plumbing - real calibration needs real models.
    """

    def local_responder(prompt, **kw):
        if "CONFIDENCE (0-100)" in prompt:
            return "85"
        return "mock-local-answer"

    local = ScriptedModel(local_responder, name="mock-local", backend="local")
    remote = ScriptedModel(
        {"": "mock-remote-answer"}, name="mock-remote", backend="fireworks",
        default="mock-remote-answer",
    )
    return local, remote


def _real_models(local_model: str, remote_model: str):
    from token_router.models.fireworks import FireworksModel
    from token_router.models.local import LocalModel

    return LocalModel(local_model), FireworksModel(remote_model)


def evaluate(tasks, threshold: int, *, mock: bool, local_model: str, remote_model: str) -> dict:
    config = RouterConfig(local_keep_threshold=threshold)
    local, remote = _mock_models() if mock else _real_models(local_model, remote_model)
    ledger = Ledger(pricing=Pricing())
    agent = Agent(local, remote, router=RouterPolicy(config), ledger=ledger, warmup=not mock)
    results = agent.run(tasks)

    total = 0.0
    for task, res in zip(tasks, results):
        total += score(task.category, res.answer, task.gold or "")
    accuracy = total / len(tasks) if tasks else 0.0
    summary = ledger.summary()
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "total_tokens": summary["total_tokens"],
        "est_cost_usd": summary["est_cost_usd"],
        "local_keep_rate": summary["local_keep_rate"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Accuracy-vs-cost eval + threshold sweep.")
    p.add_argument("--mock", action="store_true", help="use offline mock models (smoke test)")
    p.add_argument("--threshold", type=int, default=None, help="single threshold instead of a sweep")
    p.add_argument("--local-model", default=os.environ.get("LOCAL_MODEL", "qwen2.5:3b-instruct"))
    p.add_argument("--remote-model", default=os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"))
    p.add_argument("--categories", nargs="*", help="limit to these proxy categories")
    args = p.parse_args(argv)

    tasks = load_proxy(args.categories)
    if not tasks:
        print("no proxy tasks found", file=sys.stderr)
        return 2

    thresholds = [args.threshold] if args.threshold is not None else [0, 50, 60, 70, 80, 90, 100]
    print(f"proxy tasks: {len(tasks)}  (mock={args.mock})")
    print(f"{'thresh':>6} | {'accuracy':>8} | {'tokens':>9} | {'cost_usd':>10} | {'keep_local':>10}")
    print("-" * 56)
    for th in thresholds:
        r = evaluate(tasks, th, mock=args.mock, local_model=args.local_model, remote_model=args.remote_model)
        print(
            f"{r['threshold']:>6} | {r['accuracy']:>8.1%} | {r['total_tokens']:>9,} | "
            f"${r['est_cost_usd']:>9.6f} | {r['local_keep_rate']:>9.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
