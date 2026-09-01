# Release validation

CueMap uses two release gates. Neither gate runs the evaluation suite.

## Before publishing

From the CueMap workspace, with the five release repositories next to one another:

```bash
node rust_engine/scripts/release-preflight.cjs
```

The preflight builds and tests the Rust engine, builds the TypeScript SDK and MCP
server, verifies the Python SDK and Agent Plugin, packs local consumer artifacts,
and installs those artifacts into a clean temporary npm project. It then verifies
the native binary version, tokenizer, engine start/ingest/recall, MCP stdio
startup, tool registration, memory creation, and recall.

Run it on macOS and Windows before publishing. It does not publish packages.
The tokenizer is reused from `dist/npm-native/tokenizer` when available or
downloaded with a pinned SHA-256 checksum. If the selected Python interpreter
does not already provide the packaging tools, the preflight creates a temporary
Python environment for them and leaves the active environment unchanged.

## After publishing

In GitHub Actions, open **Post-release Smoke Test**, choose the branch containing
this workflow, enter the published version, and run it. The workflow installs the
MCP server, TypeScript SDK, and current-platform native engine package from the
public registry at the exact requested version, then runs the smoke test on:

- macOS ARM64 and x64
- Linux ARM64 and x64
- Windows x64

It also verifies that all native engine packages, the SDKs, MCP server, and Agent
Plugin exist at the requested version, have the expected license, and that the
Agent Plugin pins the matching MCP version.

Publish in this order:

1. Native engine packages
2. Python and TypeScript SDKs
3. MCP server
4. Agent Plugin
5. Post-release smoke workflow

The post-release workflow is read-only against npm and PyPI; it never publishes
or modifies a package.
