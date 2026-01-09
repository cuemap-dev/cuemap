interface InspectorProps {
    results: any[];
    latency: number;
}

// Generate a concise one-liner explaining why this result ranked at this position
const generateRankExplanation = (r: any, rank: number): string => {
    const parts: string[] = [];

    // Primary signal: reinforcement_score = log10(reinforcement_count)
    // 0 = never reinforced, 1 = 10 reinforcements, 2 = 100 reinforcements
    if (r.reinforcement_score && r.reinforcement_score > 0.1) {
        const approxCount = Math.round(Math.pow(10, r.reinforcement_score));
        parts.push(`reinforced ~${approxCount}x`);
    }

    // Secondary signals: salience = cue_density + 0.5(if >5 cues) + 0.1(per reinforce)
    // Typical range: ~2.5 to 5.0. >4.0 is high.
    if (r.salience_score && r.salience_score > 4.0) {
        parts.push(`high salience (${r.salience_score.toFixed(1)})`);
    }

    // Match quality
    if (r.intersection_count) {
        parts.push(`${r.intersection_count} cue${r.intersection_count > 1 ? 's' : ''} matched`);
    }

    // Recency: 0-2 range, higher = more recent
    if (r.recency_score && r.recency_score > 0.8) {
        parts.push('recent');
    }

    if (parts.length === 0) {
        return `#${rank}: Default ranking`;
    }

    return `#${rank}: ${parts.join(' • ')}`;
};

// Format timestamp to human-readable relative time
// NOTE: created_at is Unix timestamp in SECONDS, not milliseconds
const formatRelativeTime = (timestamp: string | number): string => {
    if (!timestamp) return '';

    // Convert seconds to milliseconds for JavaScript Date
    const timestampMs = typeof timestamp === 'number' ? timestamp * 1000 : parseFloat(timestamp) * 1000;
    const date = new Date(timestampMs);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffSecs < 60) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

const Inspector: React.FC<InspectorProps> = ({ results, latency }) => {
    return (
        <div style={{ padding: '20px', color: '#fff', height: '100%', overflowY: 'auto' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px', color: '#94a3b8', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Recall Results
            </h3>

            {latency > 0 && (
                <div style={{ marginBottom: '15px', fontSize: '0.8rem', color: '#64748b' }}>
                    Engine Latency: {latency.toFixed(2)}ms
                </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {results.length === 0 && (
                    <div style={{ color: '#475569', fontStyle: 'italic', textAlign: 'center', marginTop: '40px' }}>
                        No memories retrieved yet.
                    </div>
                )}
                {results.map((r, index) => (
                    <div key={r.memory_id} style={{
                        padding: '15px',
                        background: '#1e293b',
                        borderRadius: '8px',
                        borderLeft: `4px solid ${r.reinforcement_score > 0.1 ? '#4ade80' : r.salience_score > 4.0 ? '#fbbf24' : '#64748b'}`
                    }}>
                        {/* Rank explanation one-liner */}
                        <div style={{
                            fontSize: '0.7rem',
                            color: '#3b82f6',
                            marginBottom: '6px',
                            fontFamily: 'monospace',
                            letterSpacing: '0.02em'
                        }}>
                            {generateRankExplanation(r, index + 1)}
                        </div>

                        {/* Content */}
                        <div style={{ fontSize: '0.9rem', marginBottom: '8px', color: '#e2e8f0', lineHeight: '1.4' }}>
                            {r.content}
                        </div>

                        {/* Metadata row: timestamp + reinforcement + salience + hits */}
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            fontSize: '0.7rem',
                            color: '#64748b',
                            marginTop: '8px',
                            paddingTop: '8px',
                            borderTop: '1px solid #334155'
                        }}>
                            <div style={{ display: 'flex', gap: '12px' }}>
                                {r.created_at && (
                                    <span title={new Date(r.created_at * 1000).toLocaleString()}>
                                        {formatRelativeTime(r.created_at)}
                                    </span>
                                )}
                                <span title="Reinforcement Score">
                                    R: {(r.reinforcement_score ?? 0).toFixed(2)}
                                </span>
                                <span title="Salience Score">
                                    S: {(r.salience_score ?? 0).toFixed(2)}
                                </span>
                            </div>
                            <div>
                                {r.intersection_count && (
                                    <span title="Number of matching cues">
                                        {r.intersection_count} hits
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Inspector;
