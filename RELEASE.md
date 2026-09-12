# Release validation

CueMap validates tagged release candidates before native publication and checks public registry installations afterward. The preflight does not run the full retrieval evaluation suite or the required release performance benchmark.

## Before publishing

From the CueMap workspace, with the five release repositories next to one another:

```bash
node rust_engine/scripts/release-preflight.cjs
```

The preflight builds and tests the Rust engine, runs TypeScript, MCP, and Python
real-engine integration tests, verifies the Python wheel in a clean environment
and the Agent Plugin, packs local consumer artifacts,
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

## Native publication gate

Create the matching version tag in all five release repositories before running
the engine release workflow. Companion sources are checked out at that tag.
Publication requires all five native build/test jobs, Rust coverage, and the
consumer preflight to pass. Linux binaries and Docker use Debian Bookworm.
The publish job publishes the exact tarballs produced and exercised by the
build jobs; pushes to `main` do not publish packages.

Native packages and the TypeScript SDK must reach the registry before the MCP
package can resolve its v0.7.3 dependency floor. After publication, refresh the
MCP lockfile from the registry to capture the published tarball integrity hashes.
