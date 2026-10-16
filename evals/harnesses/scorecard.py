#!/usr/bin/env python3
"""Recall scorecard for saved CueMap BEAM, LoCoMo and LongMemEval runs.

The harnesses write slightly different result shapes. This script reads them
into one per-question form and reports the same metrics for every suite:

  Any@k    at least one expected item within the first k results
  All@k    every expected item within the first k results
  Frac@k   mean fraction of expected items within the first k results
  NDCG@k   as stored by the harness
  tok@20   context tokens in the first 20 results (guards against wins that
           come from returning more text)

Any@k is recomputed from the per-result relevance array (BEAM, LoCoMo) or the
raw hit rank (LongMemEval). All/Frac/NDCG use the per-question values the
harness stored, so evidence matching is never re-implemented here.

With `--budgets 2000,4000,8000` (BEAM only), the script also reports
any/all/frac@<B>tok: evidence found in the leading results that fit within B
context tokens, using the BEAM harness's own evidence matcher and token
counter. This compares rankings at equal context cost.

`compare` pairs questions by (record, question text, occurrence) and reports
deltas, per-question wins/losses, per-record movement, a record-clustered
bootstrap interval, per-category deltas and the largest regressions.
Standard library only; deterministic.
"""

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

KS = (1, 5, 10, 20, 50, 100)
HEADLINE = ("any@1", "any@20", "all@20", "frac@20", "ndcg@20")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261008


def _first_relevant(rel):
    for index, value in enumerate(rel):
        if value:
            return index + 1
    return None


def _stored(question, name, depth):
    value = question.get(name)
    if value is None:
        return None
    k = int(name.rsplit("_", 1)[1])
    return float(value) if k <= depth else None


def _row(suite, key, record, category, question_text, first_hit, depth, metrics, tokens20):
    row = {
        "suite": suite,
        "key": key,
        "record": record,
        "category": category or "unknown",
        "question": question_text,
        "depth": depth,
        "tokens@20": tokens20,
    }
    for k in KS:
        if k <= depth:
            row[f"any@{k}"] = 1.0 if first_hit is not None and first_hit <= k else 0.0
        row[f"all@{k}"] = _stored(metrics, f"recall_all_{k}", depth)
        row[f"frac@{k}"] = _stored(metrics, f"recall_frac_{k}", depth)
        row[f"ndcg@{k}"] = _stored(metrics, f"ndcg_{k}", depth)
    return row


def load_run(path):
    """Return (suite_name, rows) for a saved harness result file."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError(f"{path}: not a recognised harness result list")
    first = data[0]
    rows = []
    if "probing_results" in first:
        if "context" in first:
            suite = f"beam-{first.get('context')}"
        elif "sample_id" in first:
            suite = "locomo"
        else:
            suite = "probing"
        # The run's result limit is the longest relevance array; a question
        # that returned fewer results is still scored at every k up to it.
        depth = max((len(question.get("rel_array") or [])
                     for project in data for question in project.get("probing_results", [])), default=0)
        for project in data:
            record = project.get("record_idx")
            seen = Counter()
            for question in project.get("probing_results", []):
                text = question.get("question", "")
                seen[text] += 1
                rel = question.get("rel_array") or []
                if not rel:
                    continue
                row = _row(
                    suite, (record, text, seen[text]), record, question.get("category"), text,
                    _first_relevant(rel), depth, question, question.get("ctx_tokens"),
                )
                if suite.startswith("beam-"):
                    row["_expected"] = question.get("expected_contexts") or []
                    row["_contents"] = question.get("recalled_contents") or []
                rows.append(row)
        return suite, rows
    if "question_type" in first:
        suite = "longmemeval"
        seen = Counter()
        for question in data:
            if str(question.get("selected_recall_attempt", "")).startswith("skipped") or question.get("skipped"):
                continue
            text = question.get("question", "")
            record = question.get("record_idx")
            seen[(record, text)] += 1
            metrics = question.get("raw_metrics") or {}
            hit = question.get("raw_hit_rank", question.get("hit_rank"))
            hit = hit if isinstance(hit, int) and hit > 0 else None
            tokens = question.get("raw_ctx_tokens", question.get("ctx_tokens"))
            depth = max([int(name.rsplit("_", 1)[1]) for name in metrics if name.startswith("recall_frac_")] or [20])
            rows.append(_row(
                suite, (record, text, seen[(record, text)]), record, question.get("question_type"), text,
                hit, depth, metrics, tokens,
            ))
        return suite, rows
    raise ValueError(f"{path}: unrecognised result format")


def add_budget_metrics(rows, budgets):
    """Add any/all/frac@<B>tok using the BEAM harness matcher and token counter."""
    if not budgets:
        return
    harness_dir = Path(__file__).resolve().parent
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))
    import test_beam_settled as beam  # noqa: E402  (only needed for budgets)

    for row in rows:
        expected = row.get("_expected")
        contents = row.get("_contents")
        if not expected or not contents:
            continue
        matrix = beam.build_evidence_match_matrix(expected, contents)
        costs = [beam.approx_token_count(content) for content in contents]
        for budget in budgets:
            used, covered = 0, set()
            for cost, matched in zip(costs, matrix):
                if used + cost > budget:
                    break
                used += cost
                covered |= matched
            fraction = len(covered) / len(expected)
            row[f"any@{budget}tok"] = 1.0 if covered else 0.0
            row[f"all@{budget}tok"] = 1.0 if len(covered) == len(expected) else 0.0
            row[f"frac@{budget}tok"] = fraction


def budget_names(rows):
    names = set()
    for row in rows:
        names.update(name for name in row if name.endswith("tok") and "@" in name)
    return sorted(names, key=lambda name: (int(name.split("@")[1][:-3]), name.split("@")[0]))


def load_timing(path):
    engine, http = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item.get("engine_latency_ms"), (int, float)):
            engine.append(float(item["engine_latency_ms"]))
        if isinstance(item.get("http_latency_ms"), (int, float)):
            http.append(float(item["http_latency_ms"]))
    return {"engine_ms": engine, "http_ms": http}


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def latency_summary(timing):
    if not timing:
        return None
    out = {}
    for name, values in timing.items():
        if values:
            out[name] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
    return out or None


def _mean(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return statistics.fmean(values) if values else None


def metric_names(rows):
    names = []
    for k in KS:
        for prefix in ("any", "all", "frac", "ndcg"):
            name = f"{prefix}@{k}"
            if any(row.get(name) is not None for row in rows):
                names.append(name)
    return names


def summarize(rows):
    out = OrderedDict()
    out["questions"] = len(rows)
    out["records"] = len({row["record"] for row in rows})
    for name in metric_names(rows) + budget_names(rows):
        out[name] = _mean(rows, name)
    tokens = [row["tokens@20"] for row in rows if isinstance(row.get("tokens@20"), (int, float))]
    out["tokens@20"] = statistics.fmean(tokens) if tokens else None
    out["tokens@20_p95"] = _percentile(tokens, 0.95) if tokens else None
    return out


def by_category(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    return OrderedDict((name, summarize(groups[name])) for name in sorted(groups))


def _pct(value):
    return "–" if value is None else f"{100 * value:.1f}"


def _signed(value):
    return "–" if value is None else f"{100 * value:+.1f}"


def _tokens(value):
    return "–" if value is None else f"{value:,.0f}"


def print_summary(label, rows, latency):
    summary = summarize(rows)
    print(f"## {label}: {summary['questions']} questions, {summary['records']} records\n")
    columns = [name for name in ("any@1", "any@5", "any@20", "any@100", "all@20", "all@100",
                                 "frac@20", "frac@100", "ndcg@20") if name in summary]
    columns += [name for name in budget_names(rows) if not name.startswith("any@")]
    print("| " + " | ".join(["Slice", "n"] + columns + ["tok@20"]) + " |")
    print("|" + "---|" * (len(columns) + 3))
    print("| **All** | " + str(summary["questions"]) + " | "
          + " | ".join(_pct(summary[c]) for c in columns) + f" | {_tokens(summary['tokens@20'])} |")
    for category, item in by_category(rows).items():
        print(f"| {category} | {item['questions']} | "
              + " | ".join(_pct(item.get(c)) for c in columns) + f" | {_tokens(item['tokens@20'])} |")
    print()
    if latency:
        for name, item in latency.items():
            print(f"Latency {name}: n={item['n']} mean={item['mean']:.2f} p50={item['p50']:.2f} p95={item['p95']:.2f}")
        print()
    return summary


def _clustered_bootstrap(pairs, metric):
    by_record = defaultdict(list)
    for base, cand in pairs:
        if base.get(metric) is not None and cand.get(metric) is not None:
            by_record[base["record"]].append(cand[metric] - base[metric])
    records = sorted(by_record, key=str)
    if len(records) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rng.choice(records) for _ in records]
        values = [d for record in sample for d in by_record[record]]
        deltas.append(statistics.fmean(values))
    deltas.sort()
    return deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]


def compare(baseline_rows, candidate_rows, top_regressions):
    base_index = {row["key"]: row for row in baseline_rows}
    cand_index = {row["key"]: row for row in candidate_rows}
    keys = [key for key in base_index if key in cand_index]
    pairs = [(base_index[key], cand_index[key]) for key in keys]
    result = OrderedDict()
    result["paired"] = len(pairs)
    result["baseline_only"] = len(base_index) - len(pairs)
    result["candidate_only"] = len(cand_index) - len(pairs)
    paired_base = [b for b, _ in pairs]
    paired_cand = [c for _, c in pairs]
    names = [name for name in metric_names(paired_base) if name in metric_names(paired_cand)]
    names += [name for name in budget_names(paired_base) if name in budget_names(paired_cand)]
    metrics = OrderedDict()
    for name in names:
        base_mean, cand_mean = _mean(paired_base, name), _mean(paired_cand, name)
        wins = sum(1 for b, c in pairs if b.get(name) is not None and c.get(name) is not None and c[name] > b[name] + 1e-12)
        losses = sum(1 for b, c in pairs if b.get(name) is not None and c.get(name) is not None and c[name] < b[name] - 1e-12)
        record_delta = defaultdict(list)
        for b, c in pairs:
            if b.get(name) is not None and c.get(name) is not None:
                record_delta[b["record"]].append(c[name] - b[name])
        record_means = [statistics.fmean(v) for v in record_delta.values()]
        metrics[name] = {
            "baseline": base_mean,
            "candidate": cand_mean,
            "delta": None if base_mean is None or cand_mean is None else cand_mean - base_mean,
            "question_wins": wins,
            "question_losses": losses,
            "records_improved": sum(1 for v in record_means if v > 1e-12),
            "records_regressed": sum(1 for v in record_means if v < -1e-12),
            "ci95": _clustered_bootstrap(pairs, name) if name in HEADLINE or name.endswith("tok") else None,
        }
    result["metrics"] = metrics
    base_tokens = [b["tokens@20"] for b, _ in pairs if isinstance(b.get("tokens@20"), (int, float))]
    cand_tokens = [c["tokens@20"] for _, c in pairs if isinstance(c.get("tokens@20"), (int, float))]
    result["tokens@20"] = {
        "baseline": statistics.fmean(base_tokens) if base_tokens else None,
        "candidate": statistics.fmean(cand_tokens) if cand_tokens else None,
    }
    categories = OrderedDict()
    grouped = defaultdict(list)
    for b, c in pairs:
        grouped[b["category"]].append((b, c))
    for category in sorted(grouped):
        group = grouped[category]
        categories[category] = OrderedDict(
            (name, (_mean([c for _, c in group], name) or 0.0) - (_mean([b for b, _ in group], name) or 0.0))
            for name in HEADLINE if name in names
        )
        categories[category]["n"] = len(group)
    result["categories"] = categories
    regressions = []
    for b, c in pairs:
        if b.get("frac@20") is None or c.get("frac@20") is None:
            continue
        delta = c["frac@20"] - b["frac@20"]
        if delta < -1e-12:
            regressions.append((delta, b))
    regressions.sort(key=lambda item: (item[0], str(item[1]["key"])))
    result["largest_regressions"] = [
        {"frac@20_delta": delta, "record": row["record"], "category": row["category"], "question": row["question"]}
        for delta, row in regressions[:top_regressions]
    ]
    return result


def print_compare(label, result, base_latency, cand_latency):
    print(f"## {label}: paired comparison ({result['paired']} paired questions)\n")
    if result["baseline_only"] or result["candidate_only"]:
        print(f"Unpaired: {result['baseline_only']} baseline-only, {result['candidate_only']} candidate-only.\n")
    print("| Metric | Baseline | Candidate | Δ (pts) | 95% CI (record bootstrap) | Q wins / losses | Records ↑ / ↓ |")
    print("|---|---:|---:|---:|---|---:|---:|")
    for name, item in result["metrics"].items():
        if not (name in HEADLINE or name in ("any@5", "any@100", "all@100", "frac@100") or name.endswith("tok")):
            continue
        ci = item["ci95"]
        ci_text = "" if ci is None else f"{100 * ci[0]:+.1f} … {100 * ci[1]:+.1f}"
        print(f"| {name} | {_pct(item['baseline'])} | {_pct(item['candidate'])} | {_signed(item['delta'])} | {ci_text} "
              f"| {item['question_wins']} / {item['question_losses']} | {item['records_improved']} / {item['records_regressed']} |")
    tokens = result["tokens@20"]
    print(f"| tok@20 | {_tokens(tokens['baseline'])} | {_tokens(tokens['candidate'])} | | | | |")
    print()
    if base_latency or cand_latency:
        for name in ("engine_ms", "http_ms"):
            b = (base_latency or {}).get(name)
            c = (cand_latency or {}).get(name)
            if b or c:
                fmt = lambda item: "–" if not item else f"mean {item['mean']:.2f} / p50 {item['p50']:.2f} / p95 {item['p95']:.2f}"
                print(f"Latency {name}: baseline {fmt(b)}; candidate {fmt(c)}")
        print()
    headline = [name for name in HEADLINE if any(name in v for v in result["categories"].values())]
    print("| Category | n | " + " | ".join(f"Δ {name}" for name in headline) + " |")
    print("|---|---:|" + "---:|" * len(headline))
    for category, item in result["categories"].items():
        print(f"| {category} | {item['n']} | " + " | ".join(_signed(item.get(name)) for name in headline) + " |")
    print()
    if result["largest_regressions"]:
        print("Largest Frac@20 regressions:\n")
        for item in result["largest_regressions"]:
            question = item["question"].replace("\n", " ")
            question = question if len(question) <= 110 else question[:107] + "..."
            print(f"- {_signed(item['frac@20_delta'])} · record {item['record']} · {item['category']} · {question}")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary", help="summarize one or more saved runs")
    summary.add_argument("runs", nargs="+")
    summary.add_argument("--timing", action="append", default=[], help="timing JSONL, in the same order as runs")
    summary.add_argument("--json", help="write machine-readable output here")
    summary.add_argument("--budgets", default="", help="comma-separated token budgets, e.g. 2000,4000,8000 (BEAM only)")
    comp = sub.add_parser("compare", help="pair a candidate run against a baseline run")
    comp.add_argument("baseline")
    comp.add_argument("candidate")
    comp.add_argument("--baseline-timing")
    comp.add_argument("--candidate-timing")
    comp.add_argument("--top-regressions", type=int, default=10)
    comp.add_argument("--json", help="write machine-readable output here")
    comp.add_argument("--budgets", default="", help="comma-separated token budgets, e.g. 2000,4000,8000 (BEAM only)")
    args = parser.parse_args(argv)
    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]

    if args.command == "summary":
        output = OrderedDict()
        for index, path in enumerate(args.runs):
            suite, rows = load_run(path)
            add_budget_metrics(rows, budgets)
            latency = latency_summary(load_timing(args.timing[index])) if index < len(args.timing) else None
            label = f"{suite} · {Path(path).name}"
            output[path] = {
                "suite": suite,
                "summary": print_summary(label, rows, latency),
                "categories": by_category(rows),
                "latency": latency,
            }
        if args.json:
            Path(args.json).write_text(json.dumps(output, indent=2, default=str))
        return 0

    base_suite, base_rows = load_run(args.baseline)
    cand_suite, cand_rows = load_run(args.candidate)
    add_budget_metrics(base_rows, budgets)
    add_budget_metrics(cand_rows, budgets)
    if base_suite != cand_suite:
        print(f"warning: suites differ ({base_suite} vs {cand_suite})", file=sys.stderr)
    base_latency = latency_summary(load_timing(args.baseline_timing)) if args.baseline_timing else None
    cand_latency = latency_summary(load_timing(args.candidate_timing)) if args.candidate_timing else None
    result = compare(base_rows, cand_rows, args.top_regressions)
    label = f"{cand_suite} · {Path(args.candidate).name} vs {Path(args.baseline).name}"
    print_compare(label, result, base_latency, cand_latency)
    if args.json:
        result["baseline_latency"] = base_latency
        result["candidate_latency"] = cand_latency
        Path(args.json).write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
