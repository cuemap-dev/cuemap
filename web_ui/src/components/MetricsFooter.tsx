import { useState, useEffect } from 'react';

interface MetricsData {
    ingestionRate: number;
    recallRequests: number;
    recallP99: number;
    recallAvg: number;
    memoryBytes: number;
    totalMemories: number;
    lexiconSize: number;
    totalProjects: number;
    activeJobs: number;
}

function parsePrometheusMetrics(text: string): MetricsData {
    const lines = text.split('\n');
    const metrics: Record<string, number> = {};

    for (const line of lines) {
        if (line.startsWith('#') || line.trim() === '') continue;
        const match = line.match(/^(\w+)\s+([\d.]+)/);
        if (match) {
            metrics[match[1]] = parseFloat(match[2]);
        }
    }

    return {
        ingestionRate: metrics['cuemap_ingestion_rate'] || 0,
        recallRequests: metrics['cuemap_recall_requests_total'] || 0,
        recallP99: metrics['cuemap_recall_latency_p99'] || 0,
        recallAvg: metrics['cuemap_recall_latency_avg'] || 0,
        memoryBytes: metrics['cuemap_memory_usage_bytes'] || 0,
        totalMemories: metrics['cuemap_total_memories'] || 0,
        lexiconSize: metrics['cuemap_lexicon_size'] || 0,
        totalProjects: metrics['cuemap_total_projects'] || 0,
        activeJobs: metrics['cuemap_active_jobs'] || 0,
    };
}

function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export default function MetricsFooter() {
    const [metrics, setMetrics] = useState<MetricsData | null>(null);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
    const [error, setError] = useState<boolean>(false);

    const fetchMetrics = async () => {
        try {
            const res = await fetch('/metrics');
            if (!res.ok) throw new Error('Failed to fetch metrics');
            const text = await res.text();
            const parsed = parsePrometheusMetrics(text);
            setMetrics(parsed);
            setLastUpdated(new Date());
            setError(false);
        } catch (err) {
            console.error('Failed to fetch metrics:', err);
            setError(true);
        }
    };

    useEffect(() => {
        // Initial fetch
        fetchMetrics();

        // Refresh every 60 seconds
        const interval = setInterval(fetchMetrics, 60000);

        return () => clearInterval(interval);
    }, []);

    if (!metrics) {
        return (
            <footer className="metrics-footer">
                <span className="metrics-loading">Loading metrics...</span>
            </footer>
        );
    }

    return (
        <footer className="metrics-footer">
            <div className="metrics-item">
                <span className="metrics-label">Memories</span>
                <span className="metrics-value">{metrics.totalMemories.toLocaleString()}</span>
            </div>

            <div className="metrics-divider" />

            <div className="metrics-item">
                <span className="metrics-label">Lexicon</span>
                <span className="metrics-value">{metrics.lexiconSize.toLocaleString()}</span>
            </div>

            <div className="metrics-divider" />

            <div className="metrics-item">
                <span className="metrics-label">Ingests</span>
                <span className="metrics-value">{metrics.ingestionRate.toLocaleString()}</span>
            </div>

            <div className="metrics-divider" />

            <div className="metrics-item">
                <span className="metrics-label">Recalls</span>
                <span className="metrics-value">{metrics.recallRequests.toLocaleString()}</span>
            </div>

            <div className="metrics-divider" />

            <div className="metrics-item">
                <span className="metrics-label">Average Recall Latency</span>
                <span className="metrics-value">{metrics.recallAvg.toFixed(1)}ms</span>
            </div>

            <div className="metrics-divider" />

            <div className="metrics-item">
                <span className="metrics-label">Memory</span>
                <span className="metrics-value">{formatBytes(metrics.memoryBytes)}</span>
            </div>

            {metrics.activeJobs > 0 && (
                <>
                    <div className="metrics-divider" />
                    <div className="metrics-item metrics-jobs">
                        <span className="metrics-status-dot active" />
                        <span className="metrics-label">Jobs</span>
                        <span className="metrics-value">{metrics.activeJobs}</span>
                    </div>
                </>
            )}

            <div className="metrics-spacer" />

            <div className="metrics-item metrics-status">
                <span className={`metrics-status-dot ${error ? 'error' : 'ok'}`} />
                <span className="metrics-label">
                    {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : 'Connecting...'}
                </span>
            </div>
        </footer>
    );
}
