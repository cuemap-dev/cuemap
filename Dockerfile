FROM rust:1.75-slim as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY build.rs ./
COPY data ./data

RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app
COPY --from=builder /app/target/release/cuemap-rust .

EXPOSE 8080

CMD ["./cuemap-rust"]
