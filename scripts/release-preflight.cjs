#!/usr/bin/env node

const assert = require("node:assert/strict");
const fs = require("node:fs");
const https = require("node:https");
const os = require("node:os");
const path = require("node:path");
const zlib = require("node:zlib");
const { spawnSync } = require("node:child_process");

const ENGINE_ROOT = path.resolve(__dirname, "..");
const WORKSPACE_ROOT = process.env.CUEMAP_WORKSPACE_ROOT || path.resolve(ENGINE_ROOT, "..");
const TOKENIZER_URL = process.env.CUEMAP_TOKENIZER_URL || "https://cuemap.dev/assets/en_tokenizer.bin.gz";
const TOKENIZER_SHA256 = "f54fd31ec463f8646d0239bb531a64e0210ed1ae02bf5e3b42aeeb9bff8305ba";

const REPOSITORIES = {
  python: path.join(WORKSPACE_ROOT, "python-sdk"),
  typescript: path.join(WORKSPACE_ROOT, "typescript-sdk"),
  mcp: path.join(WORKSPACE_ROOT, "mcp-server"),
  agent: path.join(WORKSPACE_ROOT, "agent-plugin"),
};

function npmCommand() {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}

function pythonCandidates() {
  return process.env.PYTHON
    ? [process.env.PYTHON]
    : process.platform === "win32"
      ? ["python", "py"]
      : ["python3", "python", "python3.14", "python3.13", "python3.12", "python3.11", "python3.10"];
}

function pythonCommand(tempDir) {
  const candidates = pythonCandidates();
  const probeDirectory = process.env.TMPDIR || os.tmpdir();
  let usablePython;
  for (const candidate of candidates) {
    const result = spawnSync(candidate, ["-c", "import sys"], {
      cwd: probeDirectory,
      stdio: "ignore",
    });
    if (!result.error && result.status === 0) {
      usablePython = candidate;
      break;
    }
  }
  if (!usablePython) {
    throw new Error("No usable Python interpreter found. Install Python 3.8+ or set PYTHON to a configured interpreter.");
  }

  const buildProbe = spawnSync(usablePython, ["-c", "import build; import setuptools.build_meta"], {
    cwd: probeDirectory,
    stdio: "ignore",
  });
  if (!buildProbe.error && buildProbe.status === 0) return usablePython;

  const environment = path.join(tempDir, "python-build-env");
  console.log("Python packaging tool not found; creating a temporary environment");
  run(usablePython, ["-m", "venv", environment], { cwd: ENGINE_ROOT });
  const environmentPython = process.platform === "win32"
    ? path.join(environment, "Scripts", "python.exe")
    : path.join(environment, "bin", "python");
  run(environmentPython, [
    "-m",
    "pip",
    "install",
    "--disable-pip-version-check",
    "build>=1.0.0",
    "setuptools>=61.0",
  ], { cwd: ENGINE_ROOT });
  const environmentProbe = spawnSync(environmentPython, ["-c", "import build; import setuptools.build_meta"], {
    cwd: probeDirectory,
    stdio: "ignore",
  });
  if (!environmentProbe.error && environmentProbe.status === 0) return environmentPython;
  throw new Error("Could not prepare a temporary Python environment with the `build` package.");
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd,
    env: options.env,
    encoding: "utf8",
    stdio: options.stdio || "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} exited with status ${result.status}`);
  }
  return result;
}

function runCapture(command, args, options = {}) {
  return run(command, args, {
    ...options,
    stdio: ["ignore", "pipe", "inherit"],
  });
}

function assertFile(filePath, label) {
  assert.ok(fs.existsSync(filePath), `${label} is missing: ${filePath}`);
  assert.ok(fs.statSync(filePath).size > 0, `${label} is empty: ${filePath}`);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readVersion(filePath, pattern) {
  const match = fs.readFileSync(filePath, "utf8").match(pattern);
  assert.ok(match, `Could not read a version from ${filePath}`);
  return match[1];
}

function assertVersions(expectedVersion) {
  const versions = {
    "rust_engine/Cargo.toml": readVersion(
      path.join(ENGINE_ROOT, "Cargo.toml"),
      /^version\s*=\s*"([^"]+)"/m,
    ),
    "python-sdk/pyproject.toml": readVersion(
      path.join(REPOSITORIES.python, "pyproject.toml"),
      /^version\s*=\s*"([^"]+)"/m,
    ),
    "typescript-sdk/package.json": readJson(path.join(REPOSITORIES.typescript, "package.json")).version,
    "mcp-server/package.json": readJson(path.join(REPOSITORIES.mcp, "package.json")).version,
    "agent-plugin/package.json": readJson(path.join(REPOSITORIES.agent, "package.json")).version,
    "agent-plugin/plugin.json": readJson(path.join(REPOSITORIES.agent, "plugin.json")).version,
  };

  for (const [file, version] of Object.entries(versions)) {
    assert.equal(version, expectedVersion, `${file} is ${version}, expected ${expectedVersion}`);
  }
}

function download(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.resume();
        download(new URL(response.headers.location, url).toString()).then(resolve, reject);
        return;
      }
      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`Tokenizer download returned HTTP ${response.statusCode}`));
        return;
      }
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => resolve(Buffer.concat(chunks)));
      response.on("error", reject);
    }).on("error", reject);
  });
}

async function findTokenizer(tempDir) {
  const candidates = [
    process.env.CUEMAP_TOKENIZER_PATH,
    path.join(ENGINE_ROOT, "dist", "npm-native", "tokenizer", "en_tokenizer.bin"),
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }

  console.log(`Downloading checksum-pinned tokenizer from ${TOKENIZER_URL}`);
  const compressed = await download(TOKENIZER_URL);
  const actual = require("node:crypto").createHash("sha256").update(compressed).digest("hex");
  assert.equal(actual, TOKENIZER_SHA256, `Tokenizer checksum mismatch: ${actual}`);
  const output = path.join(tempDir, "en_tokenizer.bin");
  fs.writeFileSync(output, zlib.gunzipSync(compressed));
  return output;
}

function packDirectory(directory, destination) {
  const result = runCapture(npmCommand(), ["pack", "--json", "--pack-destination", destination], {
    cwd: directory,
  });
  const records = JSON.parse(result.stdout);
  assert.ok(Array.isArray(records) && records.length > 0, `npm pack returned no tarball for ${directory}`);
  return path.join(destination, records[records.length - 1].filename);
}

async function createLocalEnginePackage(tempDir, version) {
  const packageLabel = `${process.platform}-${process.arch}`;
  const packageRoot = path.join(tempDir, `engine-${packageLabel}`);
  const binaryName = process.platform === "win32" ? "cuemap.exe" : "cuemap";
  const nativeName = process.platform === "win32" ? "cuemap-native.exe" : "cuemap-native";
  const sourceBinary = path.join(ENGINE_ROOT, "target", "release", binaryName);
  assertFile(sourceBinary, "local release engine binary");
  const tokenizer = await findTokenizer(tempDir);

  fs.mkdirSync(path.join(packageRoot, "bin"), { recursive: true });
  fs.mkdirSync(path.join(packageRoot, "assets"), { recursive: true });
  fs.copyFileSync(path.join(ENGINE_ROOT, "scripts", "npm-native-wrapper.cjs"), path.join(packageRoot, "bin", "cuemap"));
  fs.copyFileSync(sourceBinary, path.join(packageRoot, "bin", nativeName));
  fs.copyFileSync(tokenizer, path.join(packageRoot, "assets", "en_tokenizer.bin"));
  fs.copyFileSync(path.join(ENGINE_ROOT, "scripts", "npm-native-README.md"), path.join(packageRoot, "README.md"));
  fs.copyFileSync(path.join(ENGINE_ROOT, "LICENSE"), path.join(packageRoot, "LICENSE"));
  fs.copyFileSync(path.join(ENGINE_ROOT, "NOTICE"), path.join(packageRoot, "NOTICE"));

  fs.writeFileSync(path.join(packageRoot, "package.json"), `${JSON.stringify({
    name: `@cuemap-dev/engine-${packageLabel}`,
    version,
    description: `Pre-compiled CueMap Engine for ${process.platform} ${process.arch}`,
    engines: { node: ">=18" },
    os: [process.platform],
    cpu: [process.arch],
    bin: { cuemap: "bin/cuemap" },
    files: ["bin", "assets", "README.md", "LICENSE", "NOTICE"],
    repository: { type: "git", url: "https://github.com/cuemap-dev/cuemap.git" },
    author: "Kaan Demirel",
    license: "Apache-2.0",
    publishConfig: { access: "public" },
  }, null, 2)}\n`);

  if (process.platform !== "win32") {
    fs.chmodSync(path.join(packageRoot, "bin", "cuemap"), 0o755);
    fs.chmodSync(path.join(packageRoot, "bin", nativeName), 0o755);
  }
  return packDirectory(packageRoot, tempDir);
}

async function main() {
  const expectedVersion = readVersion(path.join(ENGINE_ROOT, "Cargo.toml"), /^version\s*=\s*"([^"]+)"/m);
  assertVersions(expectedVersion);
  for (const directory of Object.values(REPOSITORIES)) {
    assert.ok(fs.existsSync(directory), `Missing sibling repository: ${directory}`);
  }

  const runtimeTempBase = process.platform === "darwin" ? "/private/tmp" : os.tmpdir();
  const runtimeTempDir = fs.mkdtempSync(path.join(runtimeTempBase, "cuemap-release-runtime-"));
  const previousTmpDir = process.env.TMPDIR;
  if (process.platform === "darwin") process.env.TMPDIR = runtimeTempDir;

  try {
    console.log(`Running CueMap ${expectedVersion} local release preflight`);
    console.log("1/6 Building and testing the Rust engine");
    run("cargo", ["build", "--locked", "--release"], { cwd: ENGINE_ROOT });
    run("cargo", ["test", "--locked"], { cwd: ENGINE_ROOT });

    console.log("2/6 Building and testing the TypeScript SDK and MCP server");
    run(npmCommand(), ["test"], { cwd: REPOSITORIES.typescript });
    run(npmCommand(), ["test"], { cwd: REPOSITORIES.mcp });

    console.log("3/6 Verifying the Python SDK package");
    run(pythonCommand(runtimeTempDir), ["scripts/verify_package.py"], { cwd: REPOSITORIES.python });

    console.log("4/6 Verifying the Agent Plugin package");
    run(process.execPath, ["scripts/verify.cjs"], { cwd: REPOSITORIES.agent });

    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "cuemap-release-preflight-"));
    try {
      console.log("5/6 Packing local consumer artifacts");
      const engineTarball = await createLocalEnginePackage(tempDir, expectedVersion);
      const sdkTarball = packDirectory(REPOSITORIES.typescript, tempDir);
      const mcpTarball = packDirectory(REPOSITORIES.mcp, tempDir);

      console.log("6/6 Installing and exercising the local consumer path");
      run(process.execPath, [
        path.join(ENGINE_ROOT, "scripts", "release-smoke.cjs"),
        "--version",
        expectedVersion,
        "--mcp-tarball",
        mcpTarball,
        "--sdk-tarball",
        sdkTarball,
        "--engine-tarball",
        engineTarball,
      ], { cwd: ENGINE_ROOT });
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }

    console.log(`Local release preflight passed for CueMap ${expectedVersion} on ${process.platform}-${process.arch}`);
  } finally {
    if (previousTmpDir === undefined) delete process.env.TMPDIR;
    else process.env.TMPDIR = previousTmpDir;
    fs.rmSync(runtimeTempDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
