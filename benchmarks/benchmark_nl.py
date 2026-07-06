#!/usr/bin/env python3
"""
Benchmark script for CueMap: Natural Language (NL) Capabilities
Tests pure NL ingestion (extraction) and NL recall (query resolution).
Uses Zipfian distribution for realistic patterns.
"""

import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
from dataclasses import dataclass
import random
import os
import gc
import hashlib

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    engine: str
    operation: str
    dataset_size: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    timing_summary: Optional[Dict[str, Dict[str, float]]] = None


class ZipfianGenerator:
    """Generate cues following Zipfian/Power Law distribution."""
    
    def __init__(self, num_unique_cues: int = 20000, alpha: float = 1.5):
        self.num_unique_cues = num_unique_cues
        self.alpha = alpha
        
        # Generate base cue vocabulary
        self.cues = [f"cue_{i}" for i in range(num_unique_cues)]
        
        # Pre-compute Zipfian probabilities
        ranks = np.arange(1, num_unique_cues + 1)
        self.probabilities = 1.0 / np.power(ranks, alpha)
        self.probabilities /= self.probabilities.sum()
    
    def sample_cues(self, num_cues: int = 3) -> List[str]:
        """Sample cues following Zipfian distribution."""
        return list(np.random.choice(
            self.cues,
            size=num_cues,
            replace=False,
            p=self.probabilities
        ))


class WikiLoader:
    """Load Wikipedia dataset for realistic NL benchmarks."""
    
    def __init__(self, path: str, max_texts: int = 50000, file_limit: int = 10, batch_size: int = 4096):
        self.path = path
        self.max_texts = max_texts
        self.file_limit = file_limit
        self.batch_size = batch_size
        self.texts = []
        self._seen_unique_texts = 0
        self._seen_fingerprints = set()
        self._next_text_idx = 0
        
        if (pd or pq) and os.path.exists(path):
            try:
                if os.path.isdir(path):
                    import glob
                    files = glob.glob(os.path.join(path, "*.parquet"))
                    if not files:
                        print(f"  ! No parquet files found in {path}")
                        return
                    
                    files = sorted(files)
                    if self.file_limit <= 0:
                        selected_files = files
                        random.shuffle(selected_files)
                    else:
                        selected_files = random.sample(files, min(len(files), self.file_limit))
                    print(
                        f"  Sampling up to {self.max_texts:,} Wikipedia snippets "
                        f"from {len(selected_files):,} parquet files..."
                    )
                    for f in selected_files:
                        if len(self.texts) >= self.max_texts:
                            break
                        try:
                            loaded = self._sample_file(f)
                            print(f"    - Sampled {loaded:,} rows from {os.path.basename(f)}")
                        except Exception as e:
                            print(f"    x Failed to load {os.path.basename(f)}: {e}")
                else:
                    print(f"  Sampling Wikipedia dataset from {path}...")
                    self._sample_file(path)
                
                if self.texts:
                    random.shuffle(self.texts)
                    print(
                        f"  ✓ Retained {len(self.texts):,} unique sampled snippets "
                        f"from {self._seen_unique_texts:,} unique candidate rows."
                    )
                    if len(self.texts) < self.max_texts:
                        print(
                            f"  ! Only {len(self.texts):,} unique snippets were available; "
                            f"requested {self.max_texts:,}."
                        )
            except Exception as e:
                print(f"  ✗ Failed to load Wikipedia: {e}")
                self.texts = []
        else:
            if not pd and not pq:
                print("  ! Pandas/pyarrow not installed. Using synthetic data.")
            else:
                print(f"  ! File not found: {path}. Using synthetic data.")

    @property
    def has_data(self) -> bool:
        return bool(self.texts)

    def _sample_file(self, path: str) -> int:
        loaded = 0
        if pq is not None:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(columns=["text"], batch_size=self.batch_size):
                texts = batch.column(0).to_pylist()
                for text in texts:
                    if len(self.texts) >= self.max_texts:
                        return loaded
                    loaded += 1
                    self._maybe_add_text(text)
            return loaded

        if pd is None:
            return loaded

        df = pd.read_parquet(path, columns=["text"])
        for text in df["text"].tolist():
            if len(self.texts) >= self.max_texts:
                break
            loaded += 1
            self._maybe_add_text(text)
        del df
        gc.collect()
        return loaded

    def _maybe_add_text(self, text: str):
        if not isinstance(text, str) or len(text) <= 100:
            return
        snippet = self._first_sentence(text)
        if not snippet:
            return
        if len(self.texts) >= self.max_texts:
            return
        fingerprint = hashlib.blake2b(snippet.encode("utf-8"), digest_size=16).digest()
        if fingerprint in self._seen_fingerprints:
            return
        self._seen_fingerprints.add(fingerprint)

        self._seen_unique_texts += 1
        self.texts.append(snippet)

    def _first_sentence(self, text: str) -> str:
        import re
        match = re.search(r'[.!?]\s', text)
        if match:
            return text[:match.end()].strip()
        if len(text) > 200:
            return text[:200].rsplit(' ', 1)[0] + "."
        return text.strip()

    def sample_text(self) -> str:
        """Return a unique article text (first sentence only)."""
        if self.texts:
            if self._next_text_idx >= len(self.texts):
                return None
            text = self.texts[self._next_text_idx]
            self._next_text_idx += 1
            return text
        return None

    def sample_query_from_text(self, text: str) -> str:
        """Generate a natural language query based on the text."""
        # Split into words, remove strict punctuation
        words = [w.strip(".,;:()[]\"'") for w in text.split()]
        words = [w for w in words if len(w) > 4] # distinct words
        
        if len(words) < 3:
            return "tell me about something"
            
        # Pick 2-4 random interesting words
        sample = random.sample(words, min(len(words), random.randint(2, 5)))
        return f"{' '.join(sample)}"

class CueMapNLBenchmark:
    """Benchmark harness for CueMap engines in NL mode."""

    def __init__(
        self,
        python_url: str,
        rust_url: str,
        project_id: str = None,
        wiki_path: str = None,
        wiki_reservoir_size: int = 50000,
        wiki_file_limit: int = 10,
        payload_buffer_size: int = 1000,
        query_sample_size: int = 10000,
        include_metadata: bool = False,
        batch_writes: bool = False,
    ):
        self.python_url = python_url
        self.rust_url = rust_url
        self.project_id = project_id
        self.zipf = ZipfianGenerator(num_unique_cues=20000, alpha=1.5)
        self.wiki = WikiLoader(
            wiki_path,
            max_texts=wiki_reservoir_size,
            file_limit=wiki_file_limit,
        ) if wiki_path else None
        self.payload_buffer_size = max(1, payload_buffer_size)
        self.query_sample_size = max(1, query_sample_size)
        self.include_metadata = include_metadata
        self.batch_writes = batch_writes
        self.synthetic_counter = 0
        
    def _get_headers(self) -> dict:
        """Get headers for requests."""
        headers = {}
        if self.project_id:
            headers["X-Project-ID"] = self.project_id
        return headers
    
    async def generate_memory_content(self, idx: int, embedded_cues: List[str]) -> str:
        """Generate memory content with embedded cues for extraction."""
        if self.wiki and self.wiki.has_data:
             text = self.wiki.sample_text()
             if text:
                 return text
             raise RuntimeError(
                 "Wikipedia unique snippet reservoir exhausted. Increase "
                 "--wiki-reservoir-size, reduce --sizes, or use --wiki-file-limit 0."
             )

        # Fallback to synthetic
        self.synthetic_counter += 1
        # Convert underscores to spaces for natural language embedding
        # e.g., "cue_123" -> "cue 123"
        nl_cues = [c.replace("_", " ") for c in embedded_cues]
        cues_str = ", ".join(nl_cues)
        
        templates = [
            f"Meeting notes regarding {cues_str} from session {self.synthetic_counter}.",
            f"Project update on {cues_str}: Task {self.synthetic_counter} completed.",
            f"Important reminder about {cues_str} for item {self.synthetic_counter}.",
            f"Research findings on {cues_str} for experiment {self.synthetic_counter}.",
            f"Code review comments for {cues_str} in PR {self.synthetic_counter}.",
        ]
        return random.choice(templates)
    
    async def seed_data(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        size: int
    ) -> List[str]:
        """Seed the engine with test data using Zipfian distribution."""
        memory_ids = []
        
        for i in range(size):
            cues = self.zipf.sample_cues(num_cues=random.randint(2, 5))
            # Embed cues in content for NL extraction
            content = await self.generate_memory_content(i, cues)
            
            # Send empty cues to force extraction from content
            payload = {
                "content": content,
                "cues": [], 
                "metadata": {"index": i} if self.include_metadata else None,
                "minimal_response": True,
            }
            
            async with session.post(f"{url}/memories", json=payload, headers=self._get_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    memory_ids.append(data["id"])
        
        return memory_ids

    
    async def benchmark_writes(self, session: aiohttp.ClientSession, url: str, size: int) -> Tuple[BenchmarkResult, List[str]]:
        """Benchmark write performance with bounded pre-generated batches."""
        print(
            f"  Sending {size:,} writes with payload buffer={self.payload_buffer_size:,} "
            f"and query sample={self.query_sample_size:,}"
            f"{' using /memories/batch' if self.batch_writes else ''}..."
        )

        latencies = []
        ingested_texts = []
        seen_texts = 0

        for batch_start in range(0, size, self.payload_buffer_size):
            batch_end = min(size, batch_start + self.payload_buffer_size)
            payloads = []
            for i in range(batch_start, batch_end):
                cues = self.zipf.sample_cues(num_cues=random.randint(2, 5))
                content = await self.generate_memory_content(i, cues)
                seen_texts += 1
                if len(ingested_texts) < self.query_sample_size:
                    ingested_texts.append(content)
                else:
                    replace_idx = random.randint(0, seen_texts - 1)
                    if replace_idx < self.query_sample_size:
                        ingested_texts[replace_idx] = content
                payloads.append({
                    "content": content,
                    "cues": [],
                    "metadata": {"index": i} if self.include_metadata else None,
                    "minimal_response": True,
                })

            if self.batch_writes:
                op_start = time.time()
                async with session.post(
                    f"{url}/memories/batch",
                    json={"memories": payloads, "minimal_response": True},
                    headers=self._get_headers(),
                ) as resp:
                    await resp.json()
                op_end = time.time()
                per_memory_ms = ((op_end - op_start) * 1000) / max(1, len(payloads))
                latencies.extend([per_memory_ms] * len(payloads))
            else:
                for payload in payloads:
                    op_start = time.time()
                    async with session.post(f"{url}/memories", json=payload, headers=self._get_headers()) as resp:
                        await resp.json()
                    op_end = time.time()
                    latencies.append((op_end - op_start) * 1000)

            del payloads

        total_time = sum(latencies) / 1000.0
        
        result = self._compute_result(url, "write (NL)", size, total_time, latencies)
        return result, ingested_texts
    
    async def benchmark_reads(
            self,
            session: aiohttp.ClientSession,
            url: str,
            size: int,
            num_queries: int = 1000,
            skip_seeding: bool = False,
            mode: str = "lean",
            ingested_texts: List[str] = None,
            trace_timing: bool = False
        ) -> BenchmarkResult:
            """Benchmark read (NL recall) performance with pre-generated queries."""
            if mode != "lean":
                raise ValueError("benchmark_nl.py currently supports lean recall mode only")

            # Seeding logic
            if not skip_seeding:
                print(f"  Seeding {size} memories...")
                # Use passed ingested_texts if provided, otherwise generate fresh
                if not ingested_texts:
                     # Note: seed_data needs to be updated to return texts too, but for now we skip complex seeding logic here
                     # because typically we run write benchmark first.
                     await self.seed_data(session, url, size)
            else:
                print(f"  Skipping seeding (already populated)...")
            
            # 1. Pre-generate queries so Python overhead doesn't skew throughput
            print(f"  Pre-generating {num_queries} queries...")
            payloads = []
            
            use_wiki = self.wiki and self.wiki.has_data and ingested_texts
            
            for _ in range(num_queries):
                if use_wiki:
                     # Sample from actual ingested text
                     source_text = random.choice(ingested_texts)
                     query_text = self.wiki.sample_query_from_text(source_text)
                else:
                    # Synthetic Fallback
                    query_cues = self.zipf.sample_cues(num_cues=random.randint(2, 5))
                    nl_cues = [c.replace("_", " ") for c in query_cues]
                    query_text = f"{' '.join(nl_cues)}"
                
                payload = {
                    "query_text": query_text, 
                    "cues": [], # No explicit cues
                    "limit": 5,
                    "trace_timing": trace_timing,
                }

                payload.update({
                    "auto_reinforce": False,
                    "disable_salience_bias": True,
                    "disable_alias_expansion": True,
                    "disable_cuebridge_artifacts": True,
                    "depth": 1,
                    "expansion_depth": 1,
                    "cuepacks": [],
                    "parent_fusion": "off",
                    "ordered_reconstruction": "off",
                    "evidence_coverage": "off",
                })

                payloads.append(payload)

            print("  Starting benchmark...")
            latencies = []
            timing_values: Dict[str, List[float]] = {}
            start_time = time.time() # Throughput Timer Starts Here
            
            # 2. Hot Loop: Only Network I/O
            for payload in payloads:
                op_start = time.time()
                async with session.post(f"{url}/recall", json=payload, headers=self._get_headers()) as resp:
                    body = await resp.json()
                op_end = time.time()
                
                latencies.append((op_end - op_start) * 1000)  # Convert to ms
                if trace_timing:
                    self._collect_timing_values(body.get("timing", {}), timing_values)
            
            total_time = time.time() - start_time
            
            op_name = "read (NL, Lean)"
            return self._compute_result(
                url,
                op_name,
                num_queries,
                total_time,
                latencies,
                timing_values if trace_timing else None
            )

    def _collect_timing_values(
        self,
        timing: Dict[str, Any],
        sink: Dict[str, List[float]],
        prefix: str = ""
    ):
        for key, value in timing.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._collect_timing_values(value, sink, full_key)
            elif isinstance(value, (int, float)):
                sink.setdefault(full_key, []).append(float(value))
    
    def _compute_result(
        self,
        url: str,
        operation: str,
        count: int,
        total_time: float,
        latencies: List[float],
        timing_values: Optional[Dict[str, List[float]]] = None
    ) -> BenchmarkResult:
        """Compute benchmark statistics."""
        engine = "Python" if "8000" in url else "Rust"
        latencies_sorted = sorted(latencies)
        
        timing_summary = None
        if timing_values:
            timing_summary = {}
            for key, values in timing_values.items():
                if not values:
                    continue
                sorted_values = sorted(values)
                timing_summary[key] = {
                    "avg": float(np.mean(values)),
                    "p50": float(np.percentile(sorted_values, 50)),
                    "p95": float(np.percentile(sorted_values, 95)),
                    "p99": float(np.percentile(sorted_values, 99)),
                    "max": float(max(values)),
                }

        return BenchmarkResult(
            engine=engine,
            operation=operation,
            dataset_size=count,
            total_time=total_time,
            avg_latency=np.mean(latencies),
            p50_latency=np.percentile(latencies_sorted, 50),
            p95_latency=np.percentile(latencies_sorted, 95),
            p99_latency=np.percentile(latencies_sorted, 99),
            throughput=count / total_time,
            timing_summary=timing_summary
        )
    
    async def wait_for_jobs(
        self,
        session: aiohttp.ClientSession,
        url: str,
        timeout: float = 3000.0,
        poll_interval: float = 10
    ) -> bool:
        """Wait for background jobs to complete before running recalls.
        
        Polls /jobs/status until phase is 'done' or 'idle'.
        Returns True if jobs completed, False if timeout.
        """
        start_time = time.time()
        last_progress = None
        
        while time.time() - start_time < timeout:
            try:
                async with session.get(
                    f"{url}/jobs/status",
                    timeout=aiohttp.ClientTimeout(total=5),
                    headers=self._get_headers()
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        phase = data.get("phase", "unknown")
                        
                        # Show progress updates periodically
                        progress_str = f"phase={phase}"
                        if phase == "processing":
                            progress_str += f", propose_cues={data.get('propose_cues_completed', 0)}"
                            progress_str += f", train_lexicon={data.get('train_lexicon_completed', 0)}"
                            progress_str += f", update_graph={data.get('update_graph_completed', 0)}"
                        
                        if progress_str != last_progress:
                            print(f"  ⏳ Waiting for jobs: {progress_str}")
                            last_progress = progress_str
                        
                        if phase in ("done", "idle"):
                            elapsed = time.time() - start_time
                            if elapsed > 0.1:  # Only print if we actually waited
                                print(f"  ✓ Background jobs completed in {elapsed:.1f}s")
                            return True
            except Exception as e:
                print(f"  ⚠ Error checking job status: {e}")
            
            await asyncio.sleep(poll_interval)
        
        print(f"  ⚠ Timeout waiting for jobs after {timeout}s")
        return False

    async def check_server_health(self, session: aiohttp.ClientSession, url: str) -> bool:
        """Check if server is responsive."""
        try:
            async with session.get(f"{url}/stats", timeout=aiohttp.ClientTimeout(total=5), headers=self._get_headers()) as resp:
                return resp.status == 200
        except:
            return False
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print formatted benchmark results."""
        print("\n" + "="*80)
        print("NL BENCHMARK RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\nEngine: {result.engine}")
            print(f"Operation: {result.operation}")
            print(f"Dataset Size: {result.dataset_size:,}")
            print(f"Time: {result.total_time:.2f}s")
            print(f"Throughput: {result.throughput:.0f} ops/s")
            print(f"Latency (ms): Avg={result.avg_latency:.2f}, P50={result.p50_latency:.2f}, P99={result.p99_latency:.2f}")
            if result.timing_summary:
                timing_items = [
                    item for item in result.timing_summary.items()
                    if item[0].endswith("_ms") or item[0].endswith(".total_ms")
                ]
                counter_items = [
                    item for item in result.timing_summary.items()
                    if item not in timing_items
                ]
                print("Timing breakdown (avg / p99 ms):")
                for key, stats in sorted(
                    timing_items,
                    key=lambda item: item[1]["avg"],
                    reverse=True
                )[:20]:
                    print(f"  {key}: {stats['avg']:.3f} / {stats['p99']:.3f} ms")
                if counter_items:
                    print("Trace counters (avg / p99):")
                    for key, stats in sorted(
                        counter_items,
                        key=lambda item: item[1]["avg"],
                        reverse=True
                    )[:12]:
                        print(f"  {key}: {stats['avg']:.3f} / {stats['p99']:.3f}")
            print("-" * 40)

    def save_results(self, results: List[BenchmarkResult], filename: str = "benchmark_nl_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "engine": r.engine,
                "operation": r.operation,
                "dataset_size": r.dataset_size,
                "total_time": r.total_time,
                "avg_latency_ms": r.avg_latency,
                "p50_latency_ms": r.p50_latency,
                "p95_latency_ms": r.p95_latency,
                "p99_latency_ms": r.p99_latency,
                "throughput_ops_per_sec": r.throughput,
                "timing_summary": r.timing_summary
            }
            for r in results
        ]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filename}")


async def main():
    """Main entry point."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='CueMap NL Benchmark: Python vs Rust')
    parser.add_argument('--sizes', type=str, help='Comma-separated list of sizes (e.g., 10000,100000)')
    parser.add_argument('--project-id', type=str, default="nl_test",help='Project ID for multi-tenant instance')
    parser.add_argument('--wikipedia-path', type=str, default=os.path.expanduser('~/Downloads/wikipedia/'), help='Path to Wikipedia parquet file or directory')
    parser.add_argument('--wait-for-jobs', action='store_true', help='Wait for background jobs to complete before running recall benchmarks')
    parser.add_argument('--trace-timing', action='store_true', help='Request and aggregate /recall timing breakdowns from the Rust server')
    parser.add_argument('--wiki-reservoir-size', type=int, default=50000, help='Maximum unique sampled Wikipedia snippets to keep in RAM')
    parser.add_argument('--wiki-file-limit', type=int, default=10, help='Maximum parquet files to sample from a Wikipedia directory; use 0 to scan all parquet files')
    parser.add_argument('--payload-buffer-size', type=int, default=1000, help='Maximum write payloads to keep in RAM at once')
    parser.add_argument('--query-sample-size', type=int, default=10000, help='Maximum ingested texts retained for recall query generation')
    parser.add_argument('--include-metadata', action='store_true', help='Include per-memory benchmark metadata during writes')
    parser.add_argument('--batch-writes', action='store_true', help='Write each payload buffer through /memories/batch instead of one POST per memory')
    args = parser.parse_args()
    
    print("CueMap NL Benchmark: Python vs Rust")
    print("Testing Natural Language Extraction & Query Resolution")
    print("="*80)
    
    if args.project_id:
        print(f"Running in Multi-Tenant mode for project: {args.project_id}")
    
    sizes_to_run = []
    
    if args.sizes:
        try:
            sizes_to_run = [int(s.strip()) for s in args.sizes.split(',')]
        except ValueError:
            print("Error: --sizes must be a comma-separated list of integers.")
            sys.exit(1)
    else:
        sizes_to_run = [100, 1000, 10000]
              
    print(f"\nRunning benchmark for sizes: {sizes_to_run}")

    requested_unique_writes = sum(sizes_to_run)
    effective_wiki_reservoir_size = args.wiki_reservoir_size
    effective_wiki_file_limit = args.wiki_file_limit
    if args.wikipedia_path and effective_wiki_reservoir_size < requested_unique_writes:
        effective_wiki_reservoir_size = requested_unique_writes
        print(
            f"Auto-increasing Wikipedia unique reservoir from "
            f"{args.wiki_reservoir_size:,} to {effective_wiki_reservoir_size:,} "
            f"to avoid duplicate write contents across requested sizes."
        )
        if effective_wiki_file_limit > 0:
            effective_wiki_file_limit = 0
            print(
                "Auto-setting --wiki-file-limit to 0 so the loader can scan enough "
                "parquet files to fill the unique reservoir."
            )
    
    benchmark = CueMapNLBenchmark(
        python_url="http://localhost:8000",
        rust_url="http://localhost:8080",
        project_id=args.project_id,
        wiki_path=args.wikipedia_path,
        wiki_reservoir_size=effective_wiki_reservoir_size,
        wiki_file_limit=effective_wiki_file_limit,
        payload_buffer_size=args.payload_buffer_size,
        query_sample_size=args.query_sample_size,
        include_metadata=args.include_metadata,
        batch_writes=args.batch_writes,
    )
    
    try:
        results = []
        async with aiohttp.ClientSession() as session:
             # Health check
            print("\nChecking server health...")
            try:
                py_healthy = await benchmark.check_server_health(session, benchmark.python_url)
                if not py_healthy: print(f"Warning: Python server at {benchmark.python_url} not reachable")
            except: pass
            
            rust_healthy = await benchmark.check_server_health(session, benchmark.rust_url)
            if not rust_healthy:
                 raise Exception(f"Rust server not responding at {benchmark.rust_url}")
            
            print("✓ Rust server is healthy\n")

            for idx, size in enumerate(sizes_to_run):
                print(f"\n{'='*60}")
                print(f"Benchmarking with {size:,} memories [{idx+1}/{len(sizes_to_run)}]")
                print(f"{'='*60}")
                
                num_queries = 1000 if size <= 10000 else 500
                
                py_writes_done = False
                rust_writes_done = False
                
                py_texts = []
                rust_texts = []
                
                if py_healthy:
                    print(f"\n[Python] Write (NL) benchmark ({size:,} ops)...")
                    try:
                        res, texts = await benchmark.benchmark_writes(session, benchmark.python_url, size)
                        results.append(res)
                        py_texts = texts
                        print(f"  ✓ Completed in {res.total_time:.2f}s ({res.throughput:.0f} ops/s)")
                        py_writes_done = True
                    except Exception as e: print(f"  ✗ Failed: {e}")

                print(f"[Rust] Write (NL) benchmark ({size:,} ops)...")
                try:
                    res, texts = await benchmark.benchmark_writes(session, benchmark.rust_url, size)
                    angle_bracket_hack = "<" # prevent format issues
                    results.append(res)
                    rust_texts = texts
                    print(f"  ✓ Completed in {res.total_time:.2f}s ({res.throughput:.0f} ops/s)")
                    rust_writes_done = True
                    
                    # Wait for background jobs before recall benchmarks (if flag set)
                    if args.wait_for_jobs:
                        await benchmark.wait_for_jobs(session, benchmark.rust_url)
                except Exception as e: print(f"  ✗ Failed: {e}")
                
                if py_healthy:
                    print(f"\n[Python] Read (NL) benchmark ({num_queries} queries)...")
                    try:
                        res = await benchmark.benchmark_reads(
                            session, benchmark.python_url, size, num_queries, 
                            skip_seeding=py_writes_done,
                            mode="lean",
                            ingested_texts=py_texts
                        )
                        results.append(res)
                        print(f"  ✓ Completed in {res.total_time:.2f}s (P99: {res.p99_latency:.2f}ms)")
                    except Exception as e: print(f"  ✗ Failed: {e}")

                print(f"[Rust] Read (NL, Lean) benchmark ({num_queries} queries)...")
                try:
                    res = await benchmark.benchmark_reads(
                        session, benchmark.rust_url, size, num_queries, 
                        skip_seeding=rust_writes_done,
                        mode="lean",
                        ingested_texts=rust_texts,
                        trace_timing=args.trace_timing
                    )
                    results.append(res)
                    print(f"  ✓ Completed in {res.total_time:.2f}s (P99: {res.p99_latency:.2f}ms)")
                except Exception as e: print(f"  ✗ Failed: {e}")

        benchmark.print_results(results)
        benchmark.save_results(results)
        
        print("\n" + "="*80)
        print("✓ NL Benchmark completed successfully!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
