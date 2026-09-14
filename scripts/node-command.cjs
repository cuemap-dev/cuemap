const fs = require('node:fs');
const path = require('node:path');

exports.commandFor = function commandFor(command, args) {
  if (process.platform !== 'win32' || command !== 'npm.cmd') return [command, args];
  // Run npm's JavaScript CLI directly: no cmd.exe interpolation of package paths.
  const directories = [path.dirname(process.execPath), ...(process.env.PATH || '').split(path.delimiter)];
  const candidates = [process.env.npm_execpath, ...directories.map(directory => path.join(directory, 'node_modules', 'npm', 'bin', 'npm-cli.js'))];
  const cli = candidates.find(candidate => candidate && /npm-cli\.js$/i.test(candidate) && fs.existsSync(candidate));
  if (!cli) throw new Error('Cannot locate npm-cli.js; install npm alongside Node.js');
  return [process.execPath, [cli, ...args]];
};
