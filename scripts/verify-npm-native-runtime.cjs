#!/usr/bin/env node

const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");

const executable = process.argv[2];
const platformLabel = process.argv[3] || path.basename(path.dirname(path.dirname(executable || "package")));
if (!executable) {
  console.error("Usage: verify-npm-native-runtime.cjs <packaged-cuemap-command>");
  process.exit(2);
}

const memory = "Maya switched from coffee to mint tea after the April deploy.";
const project = `npm-package-smoke-${process.pid}`;
const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "cuemap-npm-runtime-"));
let child;
let output = "";

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.text();
  if (!response.ok) {
    throw new Error(`${options.method || "GET"} ${url} returned ${response.status}: ${body}`);
  }
  return body ? JSON.parse(body) : null;
}

async function waitUntilReady(baseUrl) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`CueMap exited before becoming ready.\n${output}`);
    }
    try {
      await request(`${baseUrl}/stats`, {
        headers: { "X-Project-ID": project },
      });
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw new Error(`CueMap did not become ready within 30 seconds.\n${output}`);
}

async function stopChild() {
  if (!child || child.exitCode !== null) return;
  child.kill("SIGINT");
  await Promise.race([
    new Promise((resolve) => child.once("exit", resolve)),
    new Promise((resolve) => setTimeout(resolve, 5_000)),
  ]);
  if (child.exitCode === null) child.kill("SIGKILL");
}

async function main() {
  const port = await findFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const runThroughNode = process.platform === "win32"
    && path.extname(executable).toLowerCase() !== ".exe";
  const spawnExecutable = runThroughNode ? process.execPath : executable;
  const spawnArgs = runThroughNode ? [executable] : [];
  child = spawn(
    spawnExecutable,
    [
      ...spawnArgs,
      "start",
      "--port",
      String(port),
      "--data-dir",
      dataDir,
      "--disable-snapshots",
      "--disable-bg-jobs",
    ],
    {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, TOKENIZER_PATH: "" },
    },
  );
  for (const stream of [child.stdout, child.stderr]) {
    stream.on("data", (chunk) => {
      output = `${output}${chunk}`.slice(-20_000);
    });
  }

  await waitUntilReady(baseUrl);
  const ingest = await request(`${baseUrl}/memories`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Project-ID": project,
    },
    body: JSON.stringify({ content: memory }),
  });
  if (!ingest.cues.includes("switch") || ingest.cues.includes("switched")) {
    throw new Error(`Packaged tokenizer did not lemmatize correctly: ${JSON.stringify(ingest.cues)}`);
  }

  const recall = await request(`${baseUrl}/recall`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Project-ID": project,
    },
    body: JSON.stringify({
      query_text: "What did Maya switch to after the April deploy?",
      limit: 5,
    }),
  });
  if (!recall.results.some((result) => result.content === memory)) {
    throw new Error(`Packaged recall did not return the stored memory: ${JSON.stringify(recall)}`);
  }

  console.log(`Runtime smoke passed: ${platformLabel}`);
}

main()
  .catch((error) => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  })
  .finally(async () => {
    await stopChild();
    fs.rmSync(dataDir, { recursive: true, force: true });
  });
