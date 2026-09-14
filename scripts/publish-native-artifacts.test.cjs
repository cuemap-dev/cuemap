const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { publishArtifact } = require('./publish-native-artifacts.cjs');

for (const scenario of ['same', 'different', 'absent', 'network-error']) {
  test(`publication handles ${scenario} registry state`, () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'cuemap-publish-test-'));
    const file = path.join(directory, 'candidate.tgz');
    const bytes = Buffer.from('candidate fixture');
    fs.writeFileSync(file, bytes);
    const integrity = `sha512-${createHash('sha512').update(bytes).digest('base64')}`;
    let publishes = 0;
    const run = (command, args) => {
      if (command === 'tar') return { status: 0, stdout: JSON.stringify({name:'@cuemap-dev/engine-linux-x64',version:'0.7.3'}) };
      if (args[0] === 'publish') { publishes++; return { status: 0 }; }
      if (scenario === 'same') return { status: 0, stdout: JSON.stringify(integrity) };
      if (scenario === 'different') return { status: 0, stdout: JSON.stringify('sha512-different') };
      return { status: 1, stdout: JSON.stringify({error:{code:scenario === 'absent' ? 'E404' : 'ETIMEDOUT'}}) };
    };
    try {
      if (scenario === 'different' || scenario === 'network-error') assert.throws(() => publishArtifact(file, run));
      else publishArtifact(file, run);
      assert.equal(publishes, scenario === 'absent' ? 1 : 0);
    } finally { fs.rmSync(directory, { recursive:true, force:true }); }
  });
}
