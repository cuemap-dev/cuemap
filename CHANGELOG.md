# Changelog

All notable changes to the CueMap Rust Engine will be documented in this file.

## [0.7.3] - 2026-08-27

### Added
- **Portable project packages**: Added checksummed `.cuemap` packages that carry ready-to-query project snapshots, disk-backed content, and CueBridge artifacts without replaying ingestion. Matching CLI and HTTP operations support local pack/load and S3 push/pull through an already-configured AWS CLI. Imports validate paths, payload hashes, and snapshots before staged installation.
- **Project sync protocol**: Added Git-like S3 sync with immutable content-addressed package commits, local base tracking, fast-forward push/pull, conditional head updates, and explicit divergence refusal across CLI and HTTP.
- **Mobile-language ingestion**: Added Tree-sitter-backed chunking and structural cues for Swift, Dart, Objective-C, and Kotlin files, including uppercase extensions and code-fence routing.
- **Broader source ingestion**: Added Tree-sitter-backed chunking for C, C++, C#, and Bash, plus structured TOML ingestion. Ambiguous `.h` files use source syntax and Apple-project markers to distinguish Objective-C from C/C++.
- **Language-aware cue filtering**: Added keyword sets for the new mobile languages so code ingestion does not pollute lexical cues with language syntax.
- **Project memory residency**: Added configurable inactivity-based unloading, transparent demand-loading for requests targeting unloaded projects, explicit `POST /projects/{project_id}/load` and `POST /projects/{project_id}/unload` endpoints, and loaded-state reporting in project summaries. The default inactivity period is one day.

### Fixed
- **Lemmatization correctness**: Corrected common false lemmas, with regression coverage for truncated and wrong-part-of-speech outputs.

### Changed
- **License**: The CueMap Rust Engine and native engine packages are licensed under Apache-2.0 from v0.7.3 onward. Earlier releases remain under BSL-1.1.
- **Default port**: Changed the local HTTP server default from `8080` to `8735`; `CUEMAP_PORT` and the CLI `--port` option remain available for overrides.

## [0.7.2] - 2026-08-04

### Added
- **Bundled semantic reranking**: The default `semantic-encoder` build now uses qint8 `all-MiniLM-L3-v2` for memory/query embeddings and bounded hybrid reranking; no runtime model download or external service is required. The `edge` profile uses a bundled q4 build of the same model.
- **Intent classification and reranking**: Added local query/memory intent classification, confidence-weighted hybrid reranking, persisted memory annotations, the `/intent/classify` API, and intent coverage/progress fields in `/jobs/status`.
- **Caller-provided vectors**: Added per-memory `embedding`, per-query `query_embedding`, semantic recall modes, and one-vector-per-produced-chunk `embeddings` for `/ingest/content`.
- **Edge semantic profile**: Added the `edge` profile for constrained devices. It selects q4 MiniLM-L3 while lowering ANN fanout and the vector memory budget.
- **External unit-test files**: Moved all Rust unit-test modules out of `src` into `tests/unit`, preserving parent-module privacy while keeping production files focused on implementation.
- **Coverage reporting**: Added a `cargo-llvm-cov` GitHub Actions workflow with Codecov upload and a Rust README coverage badge.
- **CLI startup and handler coverage**: Added deterministic tests for layered profile/CLI configuration, snapshot-directory recovery, encryption-key precedence, context signing, KDF salt selection, configurable file/URL ingestion, lexicon inspection, default-project persistence, HTTP-backed add/recall/status/project/alias/memory/ingest flows, connection failures, recall modes, log rendering modes, static/live server bootstrap, detached readiness, and stop lifecycle handling.
- **Critical-path coverage**: Added deterministic HTTP API guard/reload and semantic-recall tests, parent-fusion and projection helper tests, CueBridge gap-expansion coverage, read-only/global route guards, recursive URL ingestion and persisted web recall tests, unconfigured backup checks, persistence corruption and local-backup tests, job-processing tests, ingester state/preview/deduplication tests, watcher event tests, and offline web-search parsing tests.
- **Engine coverage gate**: Added focused lifecycle, disk-content, temporal-chunking, semantic budget/error, structured-reranking, source-order expansion, decay/consolidation, and generic MainStats/LexiconStats tests. `src/engine.rs` now measures 94.10% lines, 93.79% regions, and 90.91% functions across the library and dedicated engine integration suite.
- **Server-health coverage gate**: Added project-context lifecycle, cue resolution/cache, symbol routing, alias filtering, snapshot version/corruption, atomic save/load, background snapshot, and local cloud-backup tests. `src/projects.rs` now measures 95.30% lines, 94.67% regions, and 95.83% functions; `src/persistence.rs` measures 90.40% lines, 87.30% regions, and 88.24% functions in the focused library/project coverage run.
- **Snapshot recovery**: New snapshots use zstd-compressed JSON so arbitrary `serde_json::Value` metadata survives restart; legacy bincode snapshots remain supported where decodable. Startup now reports per-project snapshot load failures and can discover the legacy sibling snapshot directory. Pre-v0.7.2 bincode snapshots containing dynamic JSON metadata may still require reingestion because bincode cannot decode `deserialize_any` values.

### Changed
- **Semantic defaults**: The quality/default profile reports `all-MiniLM-L3-v2` (`bundled-qint8-minilm-l3`) with a 128-token window; the previous bundled L6 model is no longer part of the release binary.
- **Structural-only core semantics**: Removed CuePacks and domain-ontology facet rules from ingestion/query planning. The remaining deterministic planner emits structural evidence, metadata, grammatical perspective, answer shape, ordering, and reference-time signals.
- **Trained embedding-only intent categories**: Removed exact-match semantic phrase/vocabulary adjustments and runtime semantic anchor lists. A tiny model-specific linear head now maps frozen MiniLM embeddings to intent scores; syntax-only query shape can only admit an uncertain recall check without relabelling the intent or changing durable-memory eligibility.
- **Leakage-guarded intent training**: Added deterministic NumPy training for the L3-qint8 and L3-q4 intent heads.
- **Release package hygiene**: Native builds and Cargo packages now expose only the `cuemap` server binary and exclude local diagnostics, benchmarks, evals, vendor archives, caches, and the retired L6 assets.

### Fixed
- **CLI stop PID validation**: Reject Unix PID values that cannot be represented as a positive `pid_t`, preventing malformed or stale PID files from turning a targeted shutdown into a process-wide signal.
- **Native npm Publishing**: Updated the GitHub Actions publisher to ship the checksum-verified tokenizer and package launcher on every supported platform, matching the local release packager.
- **Container semantic build**: Added bundled model assets to the Docker build context and synchronized the image version metadata.
- **Intent job completion**: Failed intent annotations now reach a terminal job phase while keeping `intent_ready=false`, rather than leaving ingestion permanently in `processing`.
- **Snapshot Coverage**: Added regression coverage for periodically persisting projects created after the snapshot scheduler starts.
- **Watcher deletion path normalization**: Deletion events now canonicalize the surviving parent path so files under macOS `/var` symlinked temporary roots remove the same tracking keys created during ingestion.
- **VerifyFile deadlock**: Verification now releases the cue-index read guard before deleting stale memories, preventing worker hangs during file re-ingestion cleanup.

## [0.7.1] - 2026-07-17

### Fixed
- **Native npm Packages**: Bundled the compiled English tokenizer and added a package launcher so installed engine binaries resolve runtime assets without external setup.
- **Linux Compatibility**: Built the Linux x64 package on Debian Bookworm and switched HTTP TLS to rustls, removing the runtime OpenSSL dependency and avoiding a too-new glibc baseline.
- **Container Runtime**: Consolidated the production Docker build, added checksum-pinned tokenizer assets, non-root execution, health checks, and writable snapshot directories.

## [0.7.0] - 2026-05-19

### Added
- **CuePacks v1**: Added deterministic TOML-based CuePacks as the semantic rule layer for facets, query intent cues, aliases, and policy metadata. Bundled `memory-general` is enabled by default.
- **CueBridge Artifact Runtime**: Added project-level loading for offline-compiled CueBridge `GapPack` and `AliasPack` artifacts. Normal recall runs first; GapPack expansion is only consulted when exact recall is weak, and explain output reports artifact provenance.
- **Ordered/Evidence Recall Modes**: Added opt-in `parent_fusion`, `ordered_reconstruction`, and `evidence_coverage` recall modes for long-form logs, transcripts, tickets, and multi-evidence questions.

### Changed
- **Breaking Numeric Memory IDs**: Replaced string memory IDs with per-project `u32` memory IDs across engine storage, API responses, REST routes, CLI memory commands, posting lists, and disk-content file naming. Deterministic dedupe/update identity moved to optional `source_key`.
- **Deterministic Co-Processor Core**: Removed embedding, LLM cue generation, WordNet/POS expansion, semantic bridges, pattern completion, external lexicon graphs, context expansion/speculation endpoints, graph jobs, cue proposal jobs, lexicon training jobs, and autonomous consolidation from the core engine.
- **Synchronous Facets**: Added deterministic add-time facets for source metadata, evidence shape, temporal markers, preference/dislike/ownership/recommendation/recipe/routine signals, and bounded entity cues.
- **Query Intent Routing**: Added deterministic query intent cues for count, money, duration, current/latest, preference, source-answer, temporal-window, recipe, and recommendation queries.
- **Sparse Reranking**: Added bounded deterministic facet reranking on top of the existing sparse candidate generation.
- **Market Heatmap Source**: Market heat now aggregates recently reinforced main-memory cues instead of lexicon identity entries.
- **Indexing Fixes**: Removed duplicate full-cue indexing in `add_memory`, indexed internally generated `episode:*` cues, and kept recall `expansion_depth` default at `1`.
- **Tokenizer Runtime Asset**: Removed build-time nlprule tokenizer generation; tokenizer data now loads from `TOKENIZER_PATH`, the Cuemap data directory, or bundled runtime assets.

### Removed
- **Obsolete Semantic Expansion Paths**: Removed GloVe/finalfusion, Ollama cuegen, WordNet/POS semantic expansion, semantic bridge indexes, external lexicons, pairwise cue co-occurrence graphs, pattern completion, `/context/expand`, and `/context/speculate`.
- **Obsolete Background Jobs**: Removed `ProposeCues`, `TrainLexiconFromMemory`, `UpdateGraph`, `RebuildSemanticBridges`, and `ConsolidateMemories`.
- **Obsolete CLI Surface**: Removed stale lexicon search behavior that depended on deleted automatic lexicon training.
- **Embedded Web UI**: Removed the bundled React/Vite UI, `ui` Cargo feature, static asset handler, and Node build stage. CueMap v0.7 ships as an engine/API/CLI release.

## [0.6.7] - 2026-03-15

### Added
- **Symbol-First BM25 Intent Router**: A high-performance, deterministic NL processing engine that extracts project symbols (via Aho-Corasick) and classifies intent (via BM25) for surgical code queries.
- **Supercharged Tree-Sitter Ingestion**: Expanded structural `.scm` queries, capturing deep definitions, calls, and imports.
- **Dynamic Context Expansion**: New `expansion_depth` parameter in recall to automatically retrieve and merge adjacent code chunks sharing a `parent_id` cue.
- **Aho-Corasick Integration**: Integrated the `aho-corasick` crate for near-instant (nanosecond) project-wide symbol extraction from user queries.

## [0.6.4] - 2026-03-04

### Added
- **Configuration System**: Unified `server_config.toml` with layered loading (CLI > Env > Config > Profiles).
- **Tuning Config**: Centralized engine tuning parameters (scoring, search, expansion) in `[tuning]` section.
- **Project Watch Dir**: `set-watch-dir` command to configure agent watch directories per project.
- **Multi-Hop Recall**: `depth` parameter in recall requests to enable multi-hop associative retrieval.

### Security
- **Dynamic Signing Key**: `CryptoEngine` now loads `CUEMAP_SECRET_KEY` from environment for signing grounded recall.
- **Per-Install Salt**: Implemented secure, per-install salt generation for PBKDF2 key derivation.
- **Grounded Recall Fix**: Corrected `source` field formatting in grounded recall context blocks.

### Changed
- **Ollama Setup**: Removed automatic installation of Ollama via Homebrew. Users must now install Ollama manually if they wish to use it.


## [0.6.3] - 2026-02-14

### Added

-- **Encryption**: Memory encryption using ChaCha20-Poly1305.
-- **Compression**: Compression of memories using Zstd
-- **CLI**: Enhanced CLI with all API features.

## [0.6.2] - 2026-01-24

### Added
- **Recursive URL Ingestion**: The engine now supports recursive URL ingestion, automatically following links and extracting content from nested pages.

### Fixed
- **Docker assets issues**: Fixed issues with Docker assets not being mounted correctly.
- **Project existence checks in ingest endpoints**: Added project existence checks in ingest endpoints to prevent errors.

## [0.6.1] - 2026-01-21

### Added
- **Cloud Backup Integration**: Native support for backing up project snapshots to AWS S3, Google Cloud Storage, and Azure Blob Storage. Configurable via CLI flags (`--cloud-backup`, etc.) and managed via new API endpoints (`/backup/upload`, `/backup/download`, `/backup/list`, `/backup/:id`).
- **Context Expansion API**: New `/context/expand` endpoint that uses the cue co-occurrence graph to suggest related concepts for a given query, enabling "search suggestion" or "related topics" features.
- **Prometheus Metrics**: New `/metrics` endpoint exposing internal system metrics (ingestion rate, latency, memory usage, etc.) for observability.

## [0.6.0] - 2026-01-18

### Added
- **Embedded Web UI**: A web UI is now embedded into the engine, accessible at `http://localhost:8080/ui`. The UI provides a modern interface to ingest a URL, file or text content, perform recalls and manage your lexicon. It features a physics-based graph visualization of your memory and lexicon.
- **WordNet Expansion**: The engine now automatically expands cue with synonyms using WordNet, adding related cues to your lexicon. 
- **Cue Lemmatization**: The engine now automatically lemmatizes cues, minimizing noise and improving recall accuracy.
- **More formats for Self-Learning Agent**: The engine now supports more formats for self-learning agent, including social media exports from WhatsApp and Instagram, as well as Google Takeout exports of Chrome and YouTube.
- **Lexicon Management API**: Native support for manual lexicon management.
- **Relevance Compression for LLMs via Grounded Recall API**: The engine now supports relevance compression for LLMs via grounded recall API, which returns a context string containing related memories to the query while taking into account the token budget given by the user.

### Removed
- **Single Tenant Mode**: The engine now runs as a multi-tenant application by default, each with its own memory store, lexicons and aliases.
- **Cue Expansion with Google or OpenAI models**: To keep the deterministic nature of the engine, cue expansion is now only done using WordNet. GloVe and Ollama support exists but is not enabled by default.


---

## [0.5.0] - 2025-12-28

### Added (Alias Management & Control)
- **Alias Management API**: Native support for manual alias management via `POST /aliases`, `GET /aliases`, and `POST /aliases/merge`.
- **Deterministic Cue Expansion**: Strict filtering in alias resolution to prevent "fuzzy" leaks. Aliases are now only expanded if they strictly match the source cue.
- **LLM Context Injection**: The engine now resolves existing cues from content before prompting the LLM. These "known cues" are injected into the prompt to guide semantic expansion while maintaining determinism.

### Added (Brain-Inspired Features)
- **Pattern Completion (Hippocampal CA3)**: Implemented cue co-occurrence matrix for associative recall. The engine now automatically infers missing cues based on historical co-occurrence at retrieval-time. Features a new `disable_pattern_completion` query flag for strict matching.
- **Temporal Chunking**: Memories are now automatically grouped into "episodes" based on temporal proximity and cue overlap, linked via `episode:<id>` cues. Supports `disable_temporal_chunking` at write-time.
- **Salience Bias (Amygdala)**: Introduced a dynamic salience score for memories, boosted by cue density, reinforcement frequency, and complexity. Salient memories decay slower and rank higher. Supports `disable_salience_bias` at retrieval.
- **Match Integrity**: Each recall result now includes a `match_integrity` score (0.0 - 1.0) derived from intersection strength, context agreement, and reinforcement counts.
- **Systems Consolidation**: New mechanism to periodically merge highly overlapping memories into abstracted summaries. Consolidation is strictly additive; original "Ground Truth" memories are preserved. Summaries can be ignored during retrieval via `disable_systems_consolidation`.

### Added (v0.5 Core)
- **Selective Set Intersection**: A new, more exhaustive search strategy that replaces legacy tiered search. It scans the most selective cue list and uses O(1) probes to gather intersection data.
- **Continuous Gradient Scoring**: Replaced discrete search tiers with a smooth scoring gradient based on recency and reinforcement frequency.
- **Asynchronous Intelligence Pipeline**: Background job system for LLM-based fact extraction, cue proposal, and automatic alias discovery.
- **Explainable AI**: Support for the `explain=true` flag in recall requests, providing detailed breakdowns of intersection, recency, and frequency components.
- **Expanded Chunker Support & Structural Cues**: Recursive "Ground Truth" extraction for 17+ formats:
    - **Recursive Code Extraction**: Python, Rust, TS, JS, Go, Java, PHP now use tree-sitter to capture nested functions, classes, and methods as grounded cues.
    - **Markup & Styling**: HTML extracts IDs/classes at any depth; CSS captures selectors.
    - **Structured Data**: CSV (headers), JSON/YAML (keys/indices), and XML (attributes/IDs) now provide full structural metadata.
    - **Documents**: PDF, Word (DOCX), Excel (XLSX) text extraction.
- **Binary Ingestion**: The agent now handles binary files gracefully, computing hashes and extracting text for ingestion.
- **Multi-Tenant Isolation**: Full isolation between projects, including independent taxonomies, lexicons, and memory stores.
- **Advanced Text Normalization**: Improved NLP normalization that better handles special characters and word boundaries.
- **Lexicon Resolution**: Support for training a lexicon from existing memories to map natural language tokens to canonical cues.

### Changed
- **Memory Storage**: Optimized `OrderedSet` with `get_index_of` for O(1) recency lookup.
- **Recall Weighting**: Intersection scores are now weighted by cue relevance, improving precision for complex queries.
- **Persistence**: Enhanced snapshot mechanism with reliable roundtrip verification.

### Fixed
- **Recall Boundary Issues**: Fixed cases where niche items deep in a cue list were missed by tiered search.
- **Reinforcement Precision**: Corrected log-frequency scaling to ensure exact reinforcement scores.
- **NLP Tokenization**: Fixed edge cases in `normalize_text` involving punctuation.

### Removed
- Legacy iterative search tiers (`TIER_1_DEPTH`, `TIER_2_DEPTH`).
- Unused `BinaryHeap` implementation in favor of faster unstable sorting.

---

## [0.4.0] - 2025-11-20
### Added
- Initial support for multiple projects.
- Batch ingestion optimizations for high-throughput scenarios.
- Basic telemetry and logging infrastructure.

## [0.3.0] - 2025-10-15
### Added
- REST API layer using Axum.
- Tiered search strategy (v1).
- Concurrent indexing with DashMap.

## [0.2.0] - 2025-09-05
### Added
- Persistent storage via binary snapshots.
- CLI tool for local debugging and management.
- Improved memory synchronization.

## [0.1.0] - 2025-08-10
### Added
- Initial core engine prototype.
- In-memory memory storage and basic tokenization.
- Fundamental scoring based on exact match.

---
*Note: This version represents a significant architectural shift towards more intelligent, non-blocking asynchronous operations.*
