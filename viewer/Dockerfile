FROM node:22.22.2-bookworm-slim AS build
WORKDIR /app/viewer
COPY viewer/package.json viewer/package-lock.json ./
RUN npm ci
COPY viewer/ ./
RUN npm run build

# Public, immutable display data is downloaded only during image construction.
FROM build AS bundled
ARG RESULTS_REVISION
RUN npm run fetch-display -- --revision "$RESULTS_REVISION"

FROM node:22.22.2-bookworm-slim AS runtime
WORKDIR /app/viewer
COPY --from=bundled --chown=node:node /app/viewer ./
USER node
CMD ["npm", "start"]

FROM runtime AS space
EXPOSE 7860
CMD ["npm", "start", "--", "--space"]

FROM runtime AS local
