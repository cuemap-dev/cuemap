const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');

function publishArtifact(file, run = spawnSync) {
  const manifestResult = run('tar', ['-xOf', file, 'package/package.json'], { encoding: 'utf8' });
  if (manifestResult.error) throw manifestResult.error;
  assert.equal(manifestResult.status, 0, manifestResult.stderr);
  const manifest = JSON.parse(manifestResult.stdout);
  assert.match(manifest.name, /^@cuemap-dev\/engine-(linux-(x64|arm64)|darwin-(x64|arm64)|win32-x64)$/);
  assert.match(manifest.version, /^\d+\.\d+\.\d+$/);
  const integrity = `sha512-${createHash('sha512').update(fs.readFileSync(file)).digest('base64')}`;
  const existing = run('npm', ['view', `${manifest.name}@${manifest.version}`, 'dist.integrity', '--json', '--registry=https://registry.npmjs.org'], { encoding: 'utf8' });
  if (existing.error) throw existing.error;
  if (existing.status === 0) {
    assert.equal(JSON.parse(existing.stdout), integrity, `Published bytes differ for ${manifest.name}@${manifest.version}`);
    console.log(`Already published with matching integrity: ${manifest.name}@${manifest.version}`);
    return;
  }
  let error;
  try { error = JSON.parse(existing.stdout).error; } catch {}
  assert.equal(error?.code, 'E404', `Cannot verify registry state: ${existing.stderr || existing.stdout}`);
  const result = run('npm', ['publish', file, '--access', 'public', '--provenance'], { stdio: 'inherit' });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `Publication failed for ${manifest.name}`);
}

module.exports = { publishArtifact };
if (require.main === module) {
  const directory = process.argv[2];
  const files = fs.readdirSync(directory).filter(file => file.endsWith('.tgz')).sort();
  assert.equal(files.length, 5, 'Expected all five verified native tarballs');
  for (const file of files) publishArtifact(path.join(directory, file));
}
