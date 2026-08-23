# syntax=docker/dockerfile:1

FROM rust:1.93-slim-trixie AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY assets ./assets
COPY lemma_exceptions.json ./
COPY data/tagger/tags.json ./data/tagger/tags.json

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --locked --release \
    && cp /build/target/release/cuemap /build/cuemap

FROM scratch AS native-binary

COPY --from=builder /build/cuemap /cuemap

FROM debian:trixie-slim AS tokenizer

ARG TOKENIZER_URL="https://cuemap.dev/assets/en_tokenizer.bin.gz"
ARG TOKENIZER_SHA256="f54fd31ec463f8646d0239bb531a64e0210ed1ae02bf5e3b42aeeb9bff8305ba"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL --retry 3 "${TOKENIZER_URL}" -o /tmp/en_tokenizer.bin.gz \
    && echo "${TOKENIZER_SHA256}  /tmp/en_tokenizer.bin.gz" | sha256sum -c - \
    && gzip -dc /tmp/en_tokenizer.bin.gz > /en_tokenizer.bin \
    && rm /tmp/en_tokenizer.bin.gz

FROM debian:trixie-slim AS runtime

ARG VERSION=0.7.2
ARG REVISION=""

LABEL org.opencontainers.image.title="CueMap Engine" \
      org.opencontainers.image.description="Fast, accurate, and explainable memory recall for agents" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/cuemap-dev/cuemap"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 cuemap \
    && useradd --uid 10001 --gid cuemap --create-home \
        --home-dir /home/cuemap --shell /usr/sbin/nologin cuemap \
    && install -d -o cuemap -g cuemap /app/data /app/data/snapshots /app/assets

COPY --from=builder --chown=cuemap:cuemap /build/cuemap /app/cuemap
COPY --from=tokenizer --chown=cuemap:cuemap /en_tokenizer.bin /app/assets/en_tokenizer.bin

ENV HOME=/home/cuemap \
    RUST_LOG=info \
    CUEMAP_PORT=8080 \
    CUEMAP_DATA_DIR=/app/data \
    CUEMAP_SNAPSHOT_INTERVAL_SECONDS=60 \
    TOKENIZER_PATH=/app/assets/en_tokenizer.bin

EXPOSE 8080
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${CUEMAP_PORT:-8080}/" || exit 1

USER cuemap

CMD ["/app/cuemap", "start"]
