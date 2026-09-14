const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { stageRuntimeDlls } = require('./stage-onnxruntime-dll.cjs');

test('stages DLLs from the ONNX Runtime archive beside its link directory', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'cuemap-ort-stage-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));

  const releaseDir = path.join(fixture, 'release');
  const packageBinDir = path.join(fixture, 'package', 'bin');
  const runtimeDir = path.join(fixture, 'cache', 'onnxruntime');
  const buildDir = path.join(releaseDir, 'build', 'ort-sys-test');
  fs.mkdirSync(path.join(runtimeDir, 'bin'), { recursive: true });
  fs.mkdirSync(path.join(runtimeDir, 'lib'));
  fs.mkdirSync(buildDir, { recursive: true });
  fs.writeFileSync(path.join(runtimeDir, 'bin', 'onnxruntime.dll'), 'runtime');
  fs.writeFileSync(path.join(runtimeDir, 'bin', 'onnxruntime_providers_shared.dll'), 'provider');
  fs.writeFileSync(
    path.join(buildDir, 'output'),
    `cargo:rustc-link-search=native=${path.join(runtimeDir, 'lib')}\n`
  );

  assert.deepEqual(stageRuntimeDlls(releaseDir, packageBinDir).sort(), [
    'onnxruntime.dll',
    'onnxruntime_providers_shared.dll',
  ]);
  assert.equal(fs.readFileSync(path.join(packageBinDir, 'onnxruntime.dll'), 'utf8'), 'runtime');
});
