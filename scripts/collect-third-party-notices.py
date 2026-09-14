"""Build a reviewable license inventory from the locked Cargo dependencies."""
import argparse
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
metadata = json.loads(subprocess.check_output(
    ['cargo', 'metadata', '--locked', '--format-version', '1'], cwd=root, text=True))
sections = ['CueMap Cargo dependency notices\n\nThis inventory includes build, test, and platform-specific dependencies.\nBundled model and tokenizer assets are described separately in NOTICE.\n']
missing = []
standard_licenses = {}
for package in sorted(metadata['packages'], key=lambda item: (item['name'], item['version'])):
    if package['id'] in metadata['workspace_members']:
        continue
    package_root = Path(package['manifest_path']).parent
    files = sorted(path for path in package_root.rglob('*') if path.is_file()
                   and re.match(r'^(LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE)([._-]|$)', path.name, re.I))
    if package.get('license_file'):
        license_file = package_root / package['license_file']
        if license_file.is_file() and license_file not in files:
            files.append(license_file)
    texts = []
    for path in files:
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if text.strip():
            texts.append((str(path.relative_to(package_root)), text))
    repository = package.get('repository') or ''
    if not texts:
        vcs = package_root / '.cargo_vcs_info.json'
        revision = json.loads(vcs.read_text()).get('git', {}).get('sha1', '') if vcs.is_file() else ''
        match = re.match(r'https://github.com/([^/]+/[^/#]+)', repository)
        if match and re.fullmatch('[0-9a-f]{40}', revision):
            repo = match[1].removesuffix('.git')
            for filename in ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING', 'LICENSE-MIT', 'LICENSE-APACHE']:
                url = f'https://raw.githubusercontent.com/{repo}/{revision}/{filename}'
                try:
                    with urllib.request.urlopen(url, timeout=15) as response:
                        text = response.read().decode()
                    texts.append((url, text))
                    break
                except (OSError, urllib.error.URLError):
                    continue
    if not texts:
        declared = package.get('license') or ''
        selected = next((license for license in ['Apache-2.0', 'MIT', 'MPL-2.0'] if license in declared), None)
        if selected:
            if selected not in standard_licenses:
                url = f'https://raw.githubusercontent.com/spdx/license-list-data/v3.26.0/text/{selected}.txt'
                with urllib.request.urlopen(url, timeout=30) as response:
                    standard_licenses[selected] = response.read().decode()
            authors = ', '.join(package.get('authors') or []) or 'See the linked source package'
            label = f'SPDX {selected} license text; authors declared by package metadata: {authors}'
            texts.append((label, standard_licenses[selected]))
    if not texts:
        missing.append(f"{package['name']} {package['version']} ({package.get('license')})")
    sections.append('\n' + '=' * 72 + f"\n{package['name']} {package['version']}\n"
                    + f"Declared license: {package.get('license')}\nRepository: {repository}\n"
                    + f"Source: https://crates.io/api/v1/crates/{package['name']}/{package['version']}/download\n")
    for label, text in texts:
        sections.append(f'\n--- {label} ---\n{text.rstrip()}\n')
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('\n'.join(sections))
print(f'Wrote dependency notice proposal to {output}')
if missing:
    raise SystemExit('Missing license texts:\n' + '\n'.join(missing))
