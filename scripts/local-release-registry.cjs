const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const { createServer } = require('node:http');
const https = require('node:https');
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const { execFileSync } = require('node:child_process');

if (!isMainThread) {
  const packages = workerData.map((file, index) => {
    const manifest = JSON.parse(execFileSync('tar', ['-xOf', file, 'package/package.json'], { encoding: 'utf8' }));
    return { file, index, manifest, integrity: `sha512-${createHash('sha512').update(fs.readFileSync(file)).digest('base64')}` };
  });
  const server = createServer((request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const tarball = /^\/__local_tarballs\/(\d+)\.tgz$/.exec(pathname);
    if (tarball) {
      const item = packages[Number(tarball[1])];
      if (!item) { response.writeHead(404).end(); return; }
      response.writeHead(200, { 'content-type': 'application/octet-stream' });
      fs.createReadStream(item.file).pipe(response);
      return;
    }
    let name;
    try { name = decodeURIComponent(pathname.slice(1)); } catch { response.writeHead(400).end(); return; }
    const item = packages.find(item => item.manifest.name === name);
    if (item) {
      const manifest = { ...item.manifest, dist: { integrity: item.integrity,
        tarball: `http://127.0.0.1:${server.address().port}/__local_tarballs/${item.index}.tgz` } };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ name, 'dist-tags': { latest: manifest.version }, versions: { [manifest.version]: manifest } }));
      return;
    }
    const upstream = https.get(`https://registry.npmjs.org${request.url}`, { headers: { accept: 'application/json' } }, result => {
      response.writeHead(result.statusCode, { 'content-type': result.headers['content-type'] || 'application/json' });
      result.pipe(response);
    });
    upstream.setTimeout(30000, () => upstream.destroy());
    upstream.on('error', error => { response.writeHead(502).end(error.message); });
  });
  server.listen(0, '127.0.0.1', () => parentPort.postMessage(`http://127.0.0.1:${server.address().port}`));
} else {
  exports.start = files => new Promise((resolve, reject) => {
    const worker = new Worker(__filename, { workerData: files });
    worker.once('error', reject);
    worker.once('message', url => resolve({ url, close: () => worker.terminate() }));
  });
}
