# Contributing to CueMap Rust Engine

Thanks for helping improve CueMap. This repository contains the Rust engine; the Python SDK, TypeScript SDK, MCP server, and agent plugin have separate release surfaces.

## Before you start

- Check existing issues and pull requests before opening a new one.
- For security vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
- Keep changes focused and explain user-visible behavior in the pull request.

## Local development

Install a current stable Rust toolchain and run commands from the repository root:

```bash
cargo check --locked --all-targets
cargo test --locked
```

The engine uses a compiled `nlprule` tokenizer at runtime. Set `TOKENIZER_PATH` to a compatible tokenizer file, or place `en_tokenizer.bin` under `~/.cuemap/data`, when running the server locally. The test suite supplies its own temporary fixtures where needed.

For Windows-specific changes, the pull request must pass the Windows Compatibility workflow. A local Windows run should include:

```powershell
cargo check --locked --all-targets
cargo test --locked --no-default-features --all-targets -- --test-threads=1
cargo build --locked --release
.\target\release\cuemap.exe --version
```

## Performance requirement

Every code contribution that changes the engine, ingestion, tokenization, recall, persistence, concurrency, or dependencies must rerun the release benchmark before review. Documentation-only changes are exempt unless they change benchmark claims or instructions.

Run the lexical and hybrid benchmark modes using the procedure in the [Performance section of the README](README.md#performance). Include the exact commands, hardware, dataset sizes, and console or JSON results in the pull request. The hybrid recall average latency must remain below 10 ms at every requested scale. Report P50 and P95 as well so reviewers can see tail behavior; a contribution that misses the latency ceiling or omits benchmark results is not review-ready.

Use a release build of the branch under test and keep the benchmark settings comparable with the README reference run. Do not replace the required benchmark with a microbenchmark or a smaller synthetic test.

## Retrieval architecture requirement

Candidate generation must remain semantic-model-free. Sparse lexical and structural retrieval must select the candidate set without invoking an embedding model or using semantic vectors to discover additional candidates. Semantic models may only participate in the bounded post-generation reranking path that is already exposed by the explicit semantic or hybrid modes. Any change to this boundary must include regression coverage and is not review-ready without maintainer approval.

## Making changes

- Add or update regression tests for behavior changes.
- Add ingestion fixtures when introducing a file format or Tree-sitter grammar.
- Preserve deterministic behavior and avoid introducing network or model calls into the default engine path.
- Update the README or changelog when a change affects users, configuration, compatibility, or release behavior.
- Do not commit generated build output, credentials, local snapshots, tokenizer files, or benchmark datasets.

## Pull requests

A useful pull request includes:

- a concise summary of the problem and solution;
- the test commands that were run and any platform limitations;
- documentation or changelog updates for user-facing changes;
- migration notes for persistence, API, CLI, or compatibility changes.

Keep unrelated cleanup out of feature pull requests so reviews remain easy to verify.
