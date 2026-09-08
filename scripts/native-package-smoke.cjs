const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { createServer } = require('node:net');

(async () => {
  const root = path.resolve(process.argv[2]);
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'cuemap-native-smoke-'));
  const listener = createServer();
  await new Promise((resolve, reject) => { listener.once('error', reject); listener.listen(0, '127.0.0.1', resolve); });
  const port = listener.address().port;
  await new Promise(resolve => listener.close(resolve));
  const native = path.join(root, 'bin', process.platform === 'win32' ? 'cuemap-native.exe' : 'cuemap-native');
  const child = spawn(native, ['start', '--disable-snapshots', '--disable-bg-jobs'], {
    env: { ...process.env, CUEMAP_HOME: temp, CUEMAP_DATA_DIR: path.join(temp, 'data'),
      CUEMAP_HOST: '127.0.0.1', CUEMAP_PORT: String(port), CUEMAP_API_KEY: 'release-test-key',
      TOKENIZER_PATH: path.join(root, 'assets', 'en_tokenizer.bin') },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let logs = '';
  child.stdout.on('data', data => { logs = (logs + data).slice(-16000); });
  child.stderr.on('data', data => { logs = (logs + data).slice(-16000); });
  let spawnError;
  child.on('error', error => { spawnError = error; });
  const url = `http://127.0.0.1:${port}`;
  const headers = { 'X-API-Key': 'release-test-key', 'Content-Type': 'application/json', 'X-Project-ID': 'release-smoke' };
  try {
    const deadline = Date.now() + 30000;
    let ready = false;
    while (Date.now() < deadline) {
      if (spawnError) throw spawnError;
      if (child.exitCode !== null) throw new Error(`Engine exited: ${logs}`);
      try { ready = (await fetch(`${url}/healthz`)).status === 204; } catch {}
      if (ready) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.ok(ready, `Engine did not start: ${logs}`);
    assert.equal((await fetch(`${url}/`)).status, 401);
    const info = await (await fetch(`${url}/`, { headers })).json();
    assert.equal(info.version, manifest.version);
    assert.ok(info.capabilities.includes('project_sync_v1'));
    const added = await fetch(`${url}/memories`, { method: 'POST', headers,
      body: JSON.stringify({ content: 'Release verification remembers the cobalt lighthouse.', cues: ['cobalt', 'lighthouse'] }) });
    assert.equal(added.status, 200, await added.text());
    const recalled = await fetch(`${url}/recall`, { method: 'POST', headers,
      body: JSON.stringify({ query_text: 'cobalt lighthouse', semantic_mode: 'hybrid', auto_reinforce: false, min_intersection: 1 }) });
    assert.equal(recalled.status, 200);
    assert.match(await recalled.text(), /cobalt lighthouse/);
    const blocked = await fetch(`${url}/memories`, { method: 'POST', headers: { ...headers, Origin: 'https://untrusted.example' }, body: '{}' });
    assert.equal(blocked.status, 403);
    console.log(`Verified native package ${manifest.name}@${manifest.version}`);
  } finally {
    if (child.exitCode === null) {
      const exited = new Promise(resolve => child.once('exit', resolve));
      child.kill();
      const timer = setTimeout(() => child.kill('SIGKILL'), 5000);
      await exited;
      clearTimeout(timer);
    }
    fs.rmSync(temp, { recursive: true, force: true });
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
