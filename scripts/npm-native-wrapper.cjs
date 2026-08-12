#!/usr/bin/env node

const { spawn } = require("node:child_process");
const path = require("node:path");

const packageRoot = path.resolve(__dirname, "..");
const binaryName = process.platform === "win32" ? "cuemap-native.exe" : "cuemap-native";
const binaryPath = path.join(__dirname, binaryName);
const tokenizerPath = path.join(packageRoot, "assets", "en_tokenizer.bin");

const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: "inherit",
  env: {
    ...process.env,
    TOKENIZER_PATH: process.env.TOKENIZER_PATH || tokenizerPath,
  },
});

child.on("error", (error) => {
  console.error(`Failed to start CueMap engine: ${error.message}`);
  process.exit(1);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    if (!child.killed) child.kill(signal);
  });
}

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});
