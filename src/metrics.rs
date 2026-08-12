//! Prometheus-style metrics collection for Ops observability.
//!
//! Provides atomic counters and gauges exposed via `/metrics` endpoint
//! in Prometheus text exposition format.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Maximum latency samples to keep for P99 calculation
const LATENCY_WINDOW_SIZE: usize = 1000;

/// Collects and exposes Prometheus-format metrics
pub struct MetricsCollector {
    /// Total memory ingestions since startup
    pub ingestion_count: AtomicU64,
    /// Total recall requests since startup
    pub recall_count: AtomicU64,
    /// Sliding window of recent recall latencies (ms)
    recall_latencies: RwLock<VecDeque<f64>>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            ingestion_count: AtomicU64::new(0),
            recall_count: AtomicU64::new(0),
            recall_latencies: RwLock::new(VecDeque::with_capacity(LATENCY_WINDOW_SIZE)),
        }
    }

    /// Record a memory ingestion event
    pub fn record_ingestion(&self) {
        self.ingestion_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a recall request with its latency
    pub fn record_recall(&self, latency_ms: f64) {
        self.recall_count.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut latencies) = self.recall_latencies.write() {
            if latencies.len() >= LATENCY_WINDOW_SIZE {
                latencies.pop_front();
            }
            latencies.push_back(latency_ms);
        }
    }

    /// Calculate P99 latency from the sliding window
    pub fn get_p99_latency(&self) -> f64 {
        if let Ok(latencies) = self.recall_latencies.read() {
            if latencies.is_empty() {
                return 0.0;
            }

            let mut sorted: Vec<f64> = latencies.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let p99_index = ((sorted.len() as f64) * 0.99).ceil() as usize - 1;
            let p99_index = p99_index.min(sorted.len() - 1);
            sorted[p99_index]
        } else {
            0.0
        }
    }

    /// Get average latency from the sliding window
    pub fn get_avg_latency(&self) -> f64 {
        if let Ok(latencies) = self.recall_latencies.read() {
            if latencies.is_empty() {
                return 0.0;
            }
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        }
    }

    /// Get the number of latency samples in the window
    pub fn get_sample_count(&self) -> usize {
        if let Ok(latencies) = self.recall_latencies.read() {
            latencies.len()
        } else {
            0
        }
    }
}

/// Get current process memory usage in bytes (RSS)
/// Uses getrusage() which works on both Linux and macOS
pub fn get_memory_usage_bytes() -> u64 {
    #[cfg(unix)]
    {
        use std::mem::MaybeUninit;

        let mut rusage = MaybeUninit::<libc::rusage>::uninit();
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, rusage.as_mut_ptr()) };

        if ret == 0 {
            let rusage = unsafe { rusage.assume_init() };
            // ru_maxrss is in kilobytes on Linux, bytes on macOS
            #[cfg(target_os = "macos")]
            {
                rusage.ru_maxrss as u64
            }
            #[cfg(not(target_os = "macos"))]
            {
                (rusage.ru_maxrss as u64) * 1024
            }
        } else {
            0
        }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

#[cfg(test)]
#[path = "../tests/unit/metrics.rs"]
mod tests;
