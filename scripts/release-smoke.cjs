#!/usr/bin/env node

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { createRequire } = require("node:module");
const { execFileSync, spawnSync } = require("node:child_process");

const ROOT_DIR = path.resolve(__dirname, "..");

function usage() {
  console.error(
    "Usage: release-smoke.cjs --version <version> [--mcp-tarball <path> --sdk-tarball <path> --engine-tarball <path>]",
  );
  process.exit(2);
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith("--")) usage();
    const key = argument.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) usage();
    args[key] = value;
    index += 1;
  }
  return args;
}

function npmCommand() {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    stdio: options.stdio || "inherit",
    cwd: options.cwd,
    env: options.env,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} exited with status ${result.status}`);
  }
  return result;
}

function packageLabel() {
  return `${process.platform}-${process.arch}`;
}

function packageName() {
  return `@cuemap-dev/engine-${packageLabel()}`;
}

function assertPackageVersion(manifest, expectedVersion, label) {
  assert.equal(manifest.version, expectedVersion, `${label} version mismatch`);
}

function assertFile(filePath, label) {
  assert.ok(fs.existsSync(filePath), `${label} is missing: ${filePath}`);
  assert.ok(fs.statSync(filePath).size > 0, `${label} is empty: ${filePath}`);
}

function resolvePackageManifest(requireFromTemp, packageName) {
  try {
    return requireFromTemp.resolve(`${packageName}/package.json`);
  } catch {
    let directory = path.dirname(requireFromTemp.resolve(packageName));
    while (directory !== path.dirname(directory)) {
      const manifest = path.join(directory, "package.json");
      if (fs.existsSync(manifest)) return manifest;
      directory = path.dirname(directory);
    }
    throw new Error(`Could not locate package.json for ${packageName}`);
  }
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function verifyCommandVersion(executable, expectedVersion) {
  const runThroughNode = process.platform === "win32"
    && path.extname(executable).toLowerCase() !== ".exe";
  const command = runThroughNode ? process.execPath : executable;
  const args = runThroughNode ? [executable, "--version"] : ["--version"];
  const result = spawnSync(command, args, { encoding: "utf8" });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `packaged cuemap --version failed: ${result.stderr}`);
  assert.match(
    `${result.stdout}${result.stderr}`,
    new RegExp(escapeRegExp(expectedVersion)),
    "packaged cuemap did not report the expected version",
  );
}

function writeMcpClient(tempDir) {
  const clientPath = path.join(tempDir, "release-mcp-client.cjs");
  const source = String.raw`const assert = require("node:assert/strict");
const { createServer } = require("node:http");
const { Client } = require("@modelcontextprotocol/sdk/client/index.js");
const { StdioClientTransport } = require("@modelcontextprotocol/sdk/client/stdio.js");

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const port = address.port;
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  return port;
}

function textOf(result) {
  return (result.content || [])
    .filter((item) => item.type === "text")
    .map((item) => item.text)
    .join("\n");
}

(async () => {
  const [serverPath, expectedVersion, dataDir, logPath] = process.argv.slice(2);
  const port = await freePort();
  const project = "release-smoke-" + process.pid;
  const client = new Client({ name: "cuemap-release-smoke", version: expectedVersion });
  const env = {
    ...process.env,
    CUEMAP_PORT: String(port),
    CUEMAP_DATA_DIR: dataDir,
    CUEMAP_LOG_PATH: logPath,
    CUEMAP_PROJECT: project,
    CUEMAP_SNAPSHOT_INTERVAL_SECONDS: "3600",
  };
  delete env.CUEMAP_BIN;
  delete env.CUEMAP_URL;

  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [serverPath],
    cwd: process.cwd(),
    env,
  });

  try {
    await client.connect(transport);
    const listed = await client.listTools();
    const names = new Set(listed.tools.map((tool) => tool.name));
    for (const required of ["cuemap_add", "cuemap_recall", "cuemap_stats"]) {
      assert.ok(names.has(required), required + " missing from published MCP server");
    }

    const memory = "Release smoke test stored this memory successfully.";
    const added = await client.callTool({
      name: "cuemap_add",
      arguments: {
        project,
        content: memory,
        cues: ["release", "smoke", "verification"],
        source_key: "release-smoke:" + process.pid,
      },
    });
    assert.match(textOf(added), /Stored memory/);

    const recalled = await client.callTool({
      name: "cuemap_recall",
      arguments: {
        projects: [project],
        query: "What did the release smoke test store?",
        semantic_mode: "lexical",
        limit: 5,
      },
    });
    assert.match(textOf(recalled), /Release smoke test stored this memory successfully/);
    console.log("MCP published-install smoke passed");
  } finally {
    await client.close().catch(() => undefined);
  }
})().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
`;
  fs.writeFileSync(clientPath, source);
  return clientPath;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const expectedVersion = args.version;
  if (!expectedVersion) usage();

  const localPackages = [args["mcp-tarball"], args["sdk-tarball"], args["engine-tarball"]];
  const usingLocalPackages = localPackages.some(Boolean);
  if (usingLocalPackages && localPackages.some((value) => !value)) usage();

  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "cuemap-release-smoke-"));
  const dataDir = path.join(tempDir, "data");
  const logPath = path.join(tempDir, "engine.log");
  fs.mkdirSync(dataDir);

  try {
    run(npmCommand(), ["init", "-y"], { cwd: tempDir, stdio: "ignore" });
    const installTargets = usingLocalPackages
      ? localPackages
      : [
        `cuemap-mcp@${expectedVersion}`,
        `cuemap@${expectedVersion}`,
        `${packageName()}@${expectedVersion}`,
      ];
    run(npmCommand(), [
      "install",
      "--no-save",
      "--ignore-scripts",
      "--package-lock=false",
      ...installTargets,
    ], { cwd: tempDir });

    const requireFromTemp = createRequire(path.join(tempDir, "release-smoke-entry.cjs"));
    const engineManifestPath = resolvePackageManifest(requireFromTemp, packageName());
    const engineRoot = path.dirname(engineManifestPath);
    const engineManifest = JSON.parse(fs.readFileSync(engineManifestPath, "utf8"));
    assertPackageVersion(engineManifest, expectedVersion, packageName());
    assert.equal(engineManifest.license, "Apache-2.0", "native engine license mismatch");

    const wrapper = path.join(engineRoot, "bin", "cuemap");
    const nativeName = process.platform === "win32" ? "cuemap-native.exe" : "cuemap-native";
    assertFile(wrapper, "native package wrapper");
    assertFile(path.join(engineRoot, "bin", nativeName), "native package executable");
    assertFile(path.join(engineRoot, "assets", "en_tokenizer.bin"), "native package tokenizer");
    assertFile(path.join(engineRoot, "LICENSE"), "native package license");
    assertFile(path.join(engineRoot, "NOTICE"), "native package notice");
    verifyCommandVersion(wrapper, expectedVersion);

    const mcpManifestPath = resolvePackageManifest(requireFromTemp, "cuemap-mcp");
    const sdkManifestPath = resolvePackageManifest(requireFromTemp, "cuemap");
    assertPackageVersion(JSON.parse(fs.readFileSync(mcpManifestPath, "utf8")), expectedVersion, "cuemap-mcp");
    assertPackageVersion(JSON.parse(fs.readFileSync(sdkManifestPath, "utf8")), expectedVersion, "cuemap");
    assert.equal(JSON.parse(fs.readFileSync(mcpManifestPath, "utf8")).license, "MIT");
    assert.equal(JSON.parse(fs.readFileSync(sdkManifestPath, "utf8")).license, "MIT");

    run(process.execPath, [
      path.join(ROOT_DIR, "scripts", "verify-npm-native-runtime.cjs"),
      wrapper,
      packageLabel(),
    ], { cwd: tempDir });

    const mcpClient = writeMcpClient(tempDir);
    run(process.execPath, [
      mcpClient,
      path.join(tempDir, "node_modules", "cuemap-mcp", "build", "index.js"),
      expectedVersion,
      dataDir,
      logPath,
    ], { cwd: tempDir });

    console.log(`Release smoke passed (${usingLocalPackages ? "local packages" : "public registry"}, ${packageLabel()})`);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

try {
  main();
} catch (error) {
  console.error(error.stack || error.message);
  process.exitCode = 1;
}
