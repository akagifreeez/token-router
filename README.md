# token-router

**A hybrid, token-efficient routing agent.** It completes a set of tasks for the
**fewest tokens / lowest cost** by routing each one between a cheap **local**
model (Ollama) and a strong **remote** model (the Fireworks AI API) — trying the
cheap path first and escalating to the remote model only when a cheap confidence
signal says the local answer is unreliable.

Built for the **AMD "ACT II" — Hybrid Token-Efficient Routing Agent** track.

## The idea in one picture

```
                 ┌─────────────────────── task ───────────────────────┐
                 │                                                     │
        cheap difficulty pre-screen                                   │
                 │                                                     │
        easy/medium │                                          hard ──┘──► REMOTE
                 ▼
          LOCAL solve  (≈ free)
                 │
        cheap confidence gate
        (self-rating / format check / optional agreement)
                 │
        confident ──► keep LOCAL answer            not confident ──► REMOTE
                                                      (or remote VERIFY, v2)
```

The whole design rests on one fact: **local inference is ~free, remote calls are
the budget.** So the router is a cost-ordered cascade — spend remote tokens only
where a cheap signal says the local answer can't be trusted. On any ambiguity
(unparseable rating, failed format check, self-disagreement) it **biases to
escalation**, so worst case it degrades toward "all-remote" and stays above the
accuracy floor.

## How it holds the accuracy floor

The hidden tasks mean we can't measure accuracy directly at run time, so the
floor is protected three ways:

1. **Calibrate conservatively.** `evals/run_eval.py` sweeps the keep-threshold on
   a proxy suite and reports the accuracy-vs-cost frontier; pick a threshold that
   clears the floor *with margin*.
2. **Bias to escalation.** Anything uncertain (no rating, bad format, empty
   answer, disagreement) goes to the remote model.
3. **Panic button.** `--safety-mode` forces every task to the remote model — a
   valid, maximally-accurate (if less efficient) submission in the worst case.

## Quickstart

```bash
# 1) Run a local model server and pull the model
ollama serve &
ollama pull qwen2.5:3b-instruct

# 2) Provide a Fireworks key (never commit it)
export FIREWORKS_API_KEY=fw_...

# 3) Run the agent over a task file
pip install -e .
token-router run --in data/tasks.jsonl --out data/results.jsonl --report
```

### Docker (self-contained: model baked in)

```bash
docker build -t token-router .
docker run --rm \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -v "$PWD/data:/data" \
  token-router --in /data/tasks.jsonl --out /data/results.jsonl --report
```

The image bundles the local model, so `docker run` needs no network for local
inference (only remote calls go out). It is ~2.5–3.5 GB (model-dominated) and
wants ~4 GB RAM. For a lighter image: `--build-arg LOCAL_MODEL=qwen2.5:1.5b-instruct`.

## Input / output format

Input is JSONL (or a JSON array), one task per line:

```json
{"id": "q1", "input": "What is the capital of France?", "category": "qa"}
```

Output is JSONL, one result per line, with the routing trace:

```json
{"id": "q1", "answer": "Paris", "route": "local", "confidence": 95, "total_tokens": 24, "cost_usd": 0.0}
```

## Architecture

| Component | File | Role |
|---|---|---|
| Model interface | `token_router/models/base.py` | uniform `complete(prompt) -> (text, Usage)` |
| Remote backend | `token_router/models/fireworks.py` | Fireworks REST client + resilience |
| Local backend | `token_router/models/local.py` | Ollama REST client |
| Resilience | `token_router/models/_http.py` | retry/backoff, rate-limit, TTL cache |
| Router | `token_router/router.py` | the cascade policy (v1 + v2 flags) |
| Accounting | `token_router/accounting.py` | per-call ledger = leaderboard mirror |
| Agent / CLI | `token_router/agent.py`, `cli.py` | run loop |
| Eval | `evals/run_eval.py` | accuracy-vs-cost sweep on a proxy suite |

The HTTP resilience layer (`_http.py`) is adapted from the author's
[`hl-read`](https://github.com/akagifreeez/hl-read) project, so both model
backends get well-tested retry/backoff/rate-limit/cache behavior.

## Tuning

```bash
python evals/run_eval.py            # accuracy-vs-cost sweep across thresholds
python evals/run_eval.py --mock     # offline smoke test (no models needed)
```

Config knobs (CLI flags or `RouterConfig`): `--threshold` (local keep
threshold), `--no-self-rate`, `--agreement` (second-sample check), `--verify`
(v2: remote verifies the local answer instead of re-solving), `--safety-mode`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

All tests are offline and fully mocked — no network, no models, no credentials.

## License

MIT — see [LICENSE](LICENSE). Original work; the resilience layer is adapted from
the author's own `hl-read`.
