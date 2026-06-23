# Self-contained image: the local model is baked in at build time, so
# `docker run` needs no network for local inference and is reproducible for
# offline judging. Only remote (Fireworks) calls need outbound network + a key.
FROM python:3.11-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates procps zstd \
    && rm -rf /var/lib/apt/lists/*

# Local model runtime.
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Install the package first so dependency layers cache across code edits.
COPY pyproject.toml README.md ./
COPY token_router ./token_router
RUN pip install --no-cache-dir .

# Bake the local model into the image: start the server, pull, stop - all at
# build time so the runtime container is self-contained. Override at build with
# --build-arg LOCAL_MODEL=qwen2.5:1.5b-instruct for a lighter image.
ARG LOCAL_MODEL=qwen2.5:3b-instruct
ENV LOCAL_MODEL=${LOCAL_MODEL}
RUN ollama serve & \
    until curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 1; done && \
    ollama pull "${LOCAL_MODEL}" && \
    pkill ollama || true

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--in", "/data/tasks.jsonl", "--out", "/data/results.jsonl", "--report"]
