#!/usr/bin/env bash
# Start the local model server, wait until it answers, then run the agent.
# All args passed to `docker run` are forwarded to `token-router run`.
set -euo pipefail

ollama serve >/tmp/ollama.log 2>&1 &

# Wait for the Ollama API to come up (bounded so a broken image fails fast).
for _ in $(seq 1 60); do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

exec token-router run "$@"
