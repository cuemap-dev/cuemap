import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
args = parser.parse_args()
samples = defaultdict(list)

def collect(value, prefix=''):
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            collect(item, name)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            samples[name].append(item)

for line in Path(args.input).read_text().splitlines():
    if line.strip():
        collect(json.loads(line))
for name, values in sorted(samples.items()):
    values.sort()
    percentile = lambda fraction: values[max(0, math.ceil(len(values) * fraction) - 1)]
    print(f'{name}: n={len(values)} mean={sum(values)/len(values):.3f} p50={percentile(.5):.3f} p95={percentile(.95):.3f}')
