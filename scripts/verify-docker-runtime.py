"""Exercise an isolated release container, including its configured health check."""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--version', required=True)
args = parser.parse_args()
name = f'cuemap-release-check-{os.getpid()}'


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


def request(endpoint, payload=None, authenticated=True):
    headers = {'X-Project-ID': 'release-check', 'Content-Type': 'application/json'}
    if authenticated:
        headers['X-API-Key'] = 'release-test-key'
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url + endpoint, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


docker('run', '--detach', '--rm', '--name', name, '-e', 'CUEMAP_API_KEY=release-test-key',
       '-p', '127.0.0.1::8735', args.image)
try:
    address = docker('port', name, '8735/tcp').splitlines()[0]
    url = 'http://' + address
    deadline = time.monotonic() + 90
    while True:
        try:
            if request('/healthz', authenticated=False)[0] == 204:
                break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() > deadline:
            raise RuntimeError('Container failed to start: ' + docker('logs', name))
        time.sleep(.5)
    assert request('/', authenticated=False)[0] == 401
    status, body = request('/')
    assert status == 200 and json.loads(body)['version'] == args.version, body
    status, body = request('/memories', {'content': 'The release check remembers a cobalt lighthouse.', 'cues': ['cobalt', 'lighthouse']})
    assert status == 200, body
    status, body = request('/recall', {'query_text': 'cobalt lighthouse', 'semantic_mode': 'hybrid', 'min_intersection': 1, 'auto_reinforce': False})
    assert status == 200 and 'cobalt lighthouse' in body, body
    while docker('inspect', '--format', '{{.State.Health.Status}}', name) != 'healthy':
        if time.monotonic() > deadline:
            raise RuntimeError('Docker health check failed: ' + docker('inspect', '--format', '{{json .State.Health}}', name))
        time.sleep(1)
    print(f'Authenticated Docker runtime and health check passed: {args.image}')
finally:
    docker('stop', '--time', '5', name)
