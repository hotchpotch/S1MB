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
