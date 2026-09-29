FROM node:22.22.2-bookworm-slim AS build
WORKDIR /app/viewer
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils && rm -rf /var/lib/apt/lists/*
COPY viewer/package.json viewer/package-lock.json ./
RUN npm ci
COPY viewer/ ./
RUN npm run build

FROM node:22.22.2-bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app/viewer
COPY --from=build --chown=node:node /app/viewer ./
COPY --chown=node:node evaluator/data/benchmarks /app/evaluator/data/benchmarks
COPY --chown=node:node evaluator/data/categories /app/evaluator/data/categories
RUN mkdir -p /tmp/s1mb-viewer-results && chown node:node /tmp/s1mb-viewer-results
USER node
CMD ["npm", "start"]

# HF Spaces prepare private data at runtime, never in a distributable layer.
FROM runtime AS space
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv
WORKDIR /app/evaluator
COPY evaluator/pyproject.toml evaluator/uv.lock evaluator/README.md evaluator/LICENSE evaluator/dataset-source.json ./
COPY evaluator/src ./src
RUN uv sync --locked --no-dev --no-editable --no-cache --python /usr/bin/python3
COPY deploy/hf-space/bootstrap.py /app/deploy/bootstrap.py
RUN mkdir -p /app/tmp && chown -R node:node /app/evaluator/data /app/tmp
USER node
EXPOSE 7860
CMD ["/app/evaluator/.venv/bin/python", "/app/deploy/bootstrap.py"]

# Preserve the local Docker behavior when no target is selected.
FROM runtime AS local
