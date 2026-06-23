"""Command-line entry point: ``token-router run --in tasks.jsonl --out results.jsonl``.

Builds the local + remote models from flags/env, runs every task through the
agent, writes results, and (with ``--report``) prints the leaderboard-shaped
summary. Model ids and the API key come from the environment by default
(``LOCAL_MODEL``, ``FIREWORKS_MODEL``, ``FIREWORKS_API_KEY``).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .accounting import Ledger
from .agent import Agent
from .models.fireworks import DEFAULT_MODEL as FW_DEFAULT, FireworksModel
from .models.local import DEFAULT_MODEL as LOCAL_DEFAULT, LocalModel
from .pricing import Pricing
from .router import RouterConfig, RouterPolicy
from .tasks import load_tasks, write_results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="token-router", description="Hybrid token-efficient routing agent.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="route a task file and write results")
    run.add_argument("--in", dest="in_path", required=True, help="input tasks (.jsonl or .json)")
    run.add_argument("--out", dest="out_path", required=True, help="where to write results (.jsonl)")
    run.add_argument("--report", action="store_true", help="print a run summary at the end")
    run.add_argument("--ledger", default="results/ledger.jsonl", help="JSONL ledger path")
    run.add_argument("--local-model", default=os.environ.get("LOCAL_MODEL", LOCAL_DEFAULT))
    run.add_argument(
        "--remote-model", default=os.environ.get("FIREWORKS_MODEL", FW_DEFAULT)
    )
    run.add_argument("--threshold", type=int, default=70, help="local keep threshold (0-100)")
    run.add_argument("--no-self-rate", action="store_true", help="disable the confidence self-rating call")
    run.add_argument("--agreement", action="store_true", help="enable second-sample agreement check")
    run.add_argument("--verify", action="store_true", help="v2: remote verifies the local answer")
    run.add_argument("--safety-mode", action="store_true", help="force every task to the remote model")
    run.add_argument("--no-warmup", action="store_true", help="skip the local model warmup call")
    return p


def _run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.in_path)
    if not tasks:
        print(f"no tasks found in {args.in_path}", file=sys.stderr)
        return 2

    config = RouterConfig(
        local_keep_threshold=args.threshold,
        use_self_rate=not args.no_self_rate,
        use_agreement=args.agreement,
        use_verify=args.verify,
        safety_mode=args.safety_mode,
    )
    local = LocalModel(args.local_model)
    remote = FireworksModel(args.remote_model)
    ledger = Ledger(pricing=Pricing(), jsonl_path=args.ledger)
    agent = Agent(
        local, remote,
        router=RouterPolicy(config), ledger=ledger, warmup=not args.no_warmup,
    )

    results = agent.run(tasks)
    write_results(args.out_path, results)
    ledger.close()

    print(f"wrote {len(results)} results -> {args.out_path}")
    if args.report:
        print(ledger.format_summary())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
