#!/usr/bin/env node

const fs = require('node:fs');
const path = require('node:path');

const LINK_SEARCH_PREFIX = 'cargo:rustc-link-search=native=';
const RUNTIME_DLL = 'onnxruntime.dll';

function runtimeDirectories(releaseDir) {
  const directories = [releaseDir];
  const buildDir = path.join(releaseDir, 'build');
  if (!fs.existsSync(buildDir)) return directories;

  for (const entry of fs.readdirSync(buildDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.startsWith('ort-sys-')) continue;
    const output = path.join(buildDir, entry.name, 'output');
    if (!fs.existsSync(output)) continue;

    for (const line of fs.readFileSync(output, 'utf8').split(/\r?\n/)) {
      if (!line.startsWith(LINK_SEARCH_PREFIX)) continue;
      const linkDir = line.slice(LINK_SEARCH_PREFIX.length).trim();
      if (!/onnxruntime/i.test(linkDir)) continue;
      const root = path.basename(linkDir).toLowerCase() === 'lib'
        ? path.dirname(linkDir)
        : linkDir;
      directories.push(linkDir, path.join(root, 'bin'), path.join(root, 'lib'), root);
    }
  }

  return [...new Set(directories)];
}

function stageRuntimeDlls(releaseDir, packageBinDir) {
  const directories = runtimeDirectories(releaseDir);
  const sourceDir = directories.find((directory) =>
    fs.existsSync(path.join(directory, RUNTIME_DLL))
  );
  if (!sourceDir) {
    throw new Error(`ONNX Runtime DLL not found. Searched: ${directories.join(', ')}`);
  }

  const dlls = fs.readdirSync(sourceDir).filter((name) => /^onnxruntime.*\.dll$/i.test(name));
  fs.mkdirSync(packageBinDir, { recursive: true });
  for (const name of dlls) {
    fs.copyFileSync(path.join(sourceDir, name), path.join(packageBinDir, name));
  }
  return dlls;
}

if (require.main === module) {
  const [releaseDir, packageBinDir] = process.argv.slice(2);
  if (!releaseDir || !packageBinDir) {
    console.error('Usage: node scripts/stage-onnxruntime-dll.cjs RELEASE_DIR PACKAGE_BIN_DIR');
    process.exitCode = 2;
  } else {
    try {
      const dlls = stageRuntimeDlls(releaseDir, packageBinDir);
      console.log(`Packaged ONNX Runtime: ${dlls.join(', ')}`);
    } catch (error) {
      console.error(error.message);
      process.exitCode = 1;
    }
  }
}

module.exports = { stageRuntimeDlls };
