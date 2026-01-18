import React, { useState, useRef, useEffect, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Search, Trash2 } from 'lucide-react';

interface LexiconEntry {
    memory_id: string;
    content: string;
    token: string;
    reinforcement_score: number;
    created_at: number;
    affected_memories_count: number;
}

interface LexiconInspectResponse {
    cue: string;
    outgoing: LexiconEntry[];
    incoming: LexiconEntry[];
}

interface GraphNode {
    id: string;
    label: string;
    group: 'token' | 'canonical' | 'center' | 'incoming' | 'outgoing';
    entry?: LexiconEntry;
    highlighted?: boolean;
    x?: number;
    y?: number;
}

interface GraphLink {
    source: string | GraphNode;
    target: string | GraphNode;
}

interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}

interface LexiconGraphResponse {
    nodes: { id: string; label: string; group: string }[];
    links: { source: string; target: string }[];
    total_entries: number;
}

interface LexiconSurgeonProps {
    projectId: string;
}

const LexiconSurgeon: React.FC<LexiconSurgeonProps> = ({ projectId }) => {
    const [searchCue, setSearchCue] = useState('');
    const [data, setData] = useState<LexiconInspectResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
    const [selectedEntry, setSelectedEntry] = useState<LexiconEntry | null>(null);
    const [highlightedNodes, setHighlightedNodes] = useState<Set<string>>(new Set());
    const [wireToken, setWireToken] = useState('');
    const [wireCanonical, setWireCanonical] = useState('');
    const fgRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 600, height: 400 });
    const [synonyms, setSynonyms] = useState<{ existing: string[], new: string[] } | null>(null);

    // Fetch synonyms when wireCanonical changes
    useEffect(() => {
        const fetchSynonyms = async () => {
            if (!wireCanonical || wireCanonical.length < 2) {
                setSynonyms(null);
                return;
            }

            try {
                
                const headers: Record<string, string> = {};
                if (projectId) {
                    headers['X-Project-ID'] = projectId;
                }

                const res = await fetch(`/lexicon/synonyms/${encodeURIComponent(wireCanonical.trim())}`, { headers });
                if (res.ok) {
                    const json = await res.json();
                    setSynonyms({
                        existing: json.existing_in_graph,
                        new: json.new_only
                    });

                    // Optional: Move existing suggested nodes closer to the canonical if it exists
                    if (fgRef.current && json.existing_in_graph.length > 0) {
                        const nodes = fgRef.current.graphData().nodes;
                        const canonicalNode = nodes.find((n: any) => n.id === json.cue);

                        if (canonicalNode && canonicalNode.x !== undefined && canonicalNode.y !== undefined) {
                            json.existing_in_graph.forEach((syn: string) => {
                                const synNode = nodes.find((n: any) => n.id === syn);
                                if (synNode) {
                                    // Move closer to canonical (random offset)
                                    synNode.vx = (canonicalNode.x - synNode.x) * 0.1;
                                    synNode.vy = (canonicalNode.y - synNode.y) * 0.1;
                                }
                            });
                            // Wake up simulation
                            fgRef.current.d3ReheatSimulation();
                        }
                    }
                }
            } catch (err) {
                console.error("Failed to fetch synonyms", err);
            }
        };

        const timeoutId = setTimeout(fetchSynonyms, 500); // Debounce 500ms
        return () => clearTimeout(timeoutId);
    }, [wireCanonical]);

    // Auto-resize
    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight
                });
            }
        };
        window.addEventListener('resize', updateSize);
        updateSize();
        return () => window.removeEventListener('resize', updateSize);
    }, []);

    // Load full Lexicon graph on mount
    useEffect(() => {
        const loadFullGraph = async () => {
            try {
                const headers: Record<string, string> = {};
                if (projectId) {
                    headers['X-Project-ID'] = projectId;
                }

                const res = await fetch('/lexicon/graph', { headers });
                const json: LexiconGraphResponse = await res.json();

                const nodes: GraphNode[] = json.nodes.map(n => ({
                    id: n.id,
                    label: n.label,
                    group: n.group as 'token' | 'canonical',
                    highlighted: false
                }));

                const links: GraphLink[] = json.links.map(l => ({
                    source: l.source,
                    target: l.target
                }));

                setGraphData({ nodes, links });
            } catch (err) {
                console.error('Failed to load Lexicon graph:', err);
            }
        };

        loadFullGraph();
    }, []);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!searchCue.trim()) {
            // Clear search - show full graph, remove highlights
            setHighlightedNodes(new Set());
            setData(null);
            setSelectedEntry(null);
            // Reset graph to full view
            if (fgRef.current) {
                fgRef.current.zoomToFit(500, 50);
            }
            return;
        }

        setLoading(true);
        setSelectedEntry(null);
        try {
            
            const headers: Record<string, string> = {};
            if (projectId) {
                headers['X-Project-ID'] = projectId;
            }

            const res = await fetch(`/lexicon/inspect/${encodeURIComponent(searchCue.trim())}`, { headers });
            const json: LexiconInspectResponse = await res.json();
            setData(json);

            // Build highlighted node set
            const highlighted = new Set<string>();
            highlighted.add(json.cue);

            json.incoming.forEach(entry => {
                highlighted.add(entry.token);
            });

            json.outgoing.forEach(entry => {
                highlighted.add(entry.content);
            });

            setHighlightedNodes(highlighted);

            // Find the searched cue node and zoom to it
            setTimeout(() => {
                if (fgRef.current) {
                    const node = graphData.nodes.find(n => n.id === json.cue || n.label === json.cue);
                    if (node && node.x !== undefined && node.y !== undefined) {
                        fgRef.current.centerAt(node.x, node.y, 500);
                        fgRef.current.zoom(3, 500);
                    }
                }
            }, 100);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleNodeClick = useCallback((node: GraphNode, event: MouseEvent) => {
        // Ctrl+Click to fill wire form
        if (event.ctrlKey || event.metaKey) {
            // First Ctrl+Click fills canonical, second fills token
            if (!wireCanonical) {
                setWireCanonical(node.id);
            } else if (!wireToken) {
                setWireToken(node.id);
            } else {
                // Both filled, start over with canonical
                setWireCanonical(node.id);
                setWireToken('');
            }
            return;
        }

        // Regular click - show entry details if highlighted
        if (highlightedNodes.has(node.id) && data) {
            const entry = [...data.incoming, ...data.outgoing].find(e =>
                e.token === node.id || e.content === node.id
            );
            if (entry) {
                setSelectedEntry(entry);
            }
        }
    }, [highlightedNodes, data, wireCanonical, wireToken]);

    const handleUnwire = async () => {
        if (!selectedEntry) return;

        try {
            
            const headers: Record<string, string> = {};
            if (projectId) {
                headers['X-Project-ID'] = projectId;
            }

            const res = await fetch(`/lexicon/entry/${encodeURIComponent(selectedEntry.memory_id)}`, {
                method: 'DELETE',
                headers
            });
            const json = await res.json();

            if (json.status === 'deleted') {
                setSelectedEntry(null);
                // Reload the full graph
                const graphRes = await fetch('/lexicon/graph', { headers });
                const graphJson: LexiconGraphResponse = await graphRes.json();

                const nodes: GraphNode[] = graphJson.nodes.map(n => ({
                    id: n.id,
                    label: n.label,
                    group: n.group as 'token' | 'canonical',
                    highlighted: false
                }));

                const links: GraphLink[] = graphJson.links.map(l => ({
                    source: l.source,
                    target: l.target
                }));

                setGraphData({ nodes, links });

                // Re-run search if there was one
                if (searchCue) {
                    const searchRes = await fetch(`/lexicon/inspect/${encodeURIComponent(searchCue.trim())}`, { headers });
                    const searchJson: LexiconInspectResponse = await searchRes.json();
                    setData(searchJson);

                    const highlighted = new Set<string>();
                    highlighted.add(searchJson.cue);
                    searchJson.incoming.forEach(entry => highlighted.add(entry.token));
                    searchJson.outgoing.forEach(entry => highlighted.add(entry.content));
                    setHighlightedNodes(highlighted);
                }
            }
        } catch (err) {
            console.error(err);
        }
    };

    const handleWire = async () => {
        if (!wireToken.trim() || !wireCanonical.trim()) return;

        try {
            const headers: Record<string, string> = { 'Content-Type': 'application/json' };
            if (projectId) {
                headers['X-Project-ID'] = projectId;
            }

            const canonicalToSearch = wireCanonical.trim();

            const res = await fetch('/lexicon/wire', {
                method: 'POST',
                headers,
                body: JSON.stringify({ token: wireToken.trim(), canonical: canonicalToSearch })
            });
            const json = await res.json();

            if (json.status === 'wired') {
                setWireToken('');
                setWireCanonical('');

                // Reload full graph
                const graphRes = await fetch('/lexicon/graph', { headers });
                const graphJson: LexiconGraphResponse = await graphRes.json();

                const nodes: GraphNode[] = graphJson.nodes.map(n => ({
                    id: n.id,
                    label: n.label,
                    group: n.group as 'token' | 'canonical',
                    highlighted: false
                }));

                const links: GraphLink[] = graphJson.links.map(l => ({
                    source: l.source,
                    target: l.target
                }));

                setGraphData({ nodes, links });

                // Auto-search the canonical to zoom in and show the new connection
                setSearchCue(canonicalToSearch);

                // Trigger inspect for the canonical
                const searchRes = await fetch(`/lexicon/inspect/${encodeURIComponent(canonicalToSearch)}`, { headers: projectId ? { 'X-Project-ID': projectId } : {} });
                const searchJson: LexiconInspectResponse = await searchRes.json();
                setData(searchJson);

                // Build highlighted nodes
                const highlighted = new Set<string>();
                highlighted.add(searchJson.cue);
                searchJson.incoming.forEach(entry => highlighted.add(entry.token));
                searchJson.outgoing.forEach(entry => highlighted.add(entry.content));
                setHighlightedNodes(highlighted);

                // Zoom to the wired connection after graph simulation settles
                // Use longer delay and get nodes from actual graph ref
                setTimeout(() => {
                    if (fgRef.current) {
                        // Get node with coordinates from the live graph
                        const graphNodes = fgRef.current.graphData().nodes;
                        const targetNode = graphNodes.find((n: any) => n.id === canonicalToSearch);
                        if (targetNode && targetNode.x !== undefined && targetNode.y !== undefined) {
                            fgRef.current.centerAt(targetNode.x, targetNode.y, 800);
                            fgRef.current.zoom(3, 800);
                        }
                    }
                }, 2500); // Wait for graph simulation to stabilize
            }
        } catch (err) {
            console.error(err);
        }
    };

    // Calculate node degrees for sizing
    const nodeDegrees = React.useMemo(() => {
        const degrees: Record<string, number> = {};
        graphData.links.forEach(link => {
            const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
            const targetId = typeof link.target === 'string' ? link.target : link.target.id;
            degrees[sourceId] = (degrees[sourceId] || 0) + 1;
            degrees[targetId] = (degrees[targetId] || 0) + 1;
        });
        return degrees;
    }, [graphData.links]);

    // Hub threshold (nodes with degree >= this are "hubs")
    const hubThreshold = React.useMemo(() => {
        const values = Object.values(nodeDegrees);
        if (values.length === 0) return 5;
        const sorted = [...values].sort((a, b) => b - a);
        // Top 5% nodes are hubs
        const idx = Math.max(0, Math.floor(sorted.length * 0.05));
        return sorted[idx] || 5;
    }, [nodeDegrees]);

    const drawNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const x = node.x!;
        const y = node.y!;
        const label = node.label || '';

        const isHighlighted = highlightedNodes.size === 0 || highlightedNodes.has(node.id);
        const isSearchedCue = data?.cue === node.id;

        // Get node degree (number of connections)
        const degree = nodeDegrees[node.id] || 1;
        const isHub = degree >= hubThreshold;

        // Semantic Zoom Levels based on globalScale:
        // globalScale < 0.5: Level 0 (far) - no labels
        // globalScale 0.5-2: Level 1 (mid) - hub labels only
        // globalScale >= 2: Level 2 (close) - all labels
        const zoomLevel = globalScale < 0.5 ? 0 : (globalScale < 2 ? 1 : 2);

        let color: string;
        let glowColor: string;
        let alpha = isHighlighted ? 1 : 0.15;

        // Base radius scaled by degree (logarithmic for hubs)
        // Small nodes: 3, regular: 5-8, hubs: 10-20
        const baseRadius = isSearchedCue
            ? 12
            : Math.max(3, Math.min(20, 3 + Math.log2(degree + 1) * 3));
        const radius = baseRadius;

        if (isSearchedCue) {
            color = '#f59e0b'; // Amber for searched cue
            glowColor = '#fbbf24';
        } else if (node.group === 'token') {
            color = '#3b82f6'; // Blue for tokens
            glowColor = '#60a5fa';
        } else {
            color = '#22c55e'; // Green for canonicals
            glowColor = '#4ade80';
        }

        ctx.save();
        ctx.globalAlpha = alpha;

        // Glow effect for highlighted nodes and hubs
        if (isHighlighted && (highlightedNodes.size > 0 || isHub)) {
            ctx.shadowColor = glowColor;
            ctx.shadowBlur = isSearchedCue ? 20 : (isHub ? 15 : 8);
        }

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI, false);
        ctx.fill();

        // Bright center for searched cue and large hubs
        if (isSearchedCue || (isHub && degree > 10)) {
            ctx.shadowBlur = 5;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.beginPath();
            ctx.arc(x, y, radius * 0.35, 0, 2 * Math.PI, false);
            ctx.fill();
        }

        // Label rendering based on LOD
        const shouldShowLabel =
            isSearchedCue || // Always show searched cue
            (zoomLevel >= 2 && (isHighlighted || highlightedNodes.size === 0)) || // Close zoom: all
            (zoomLevel === 1 && isHub && (isHighlighted || highlightedNodes.size === 0)); // Mid zoom: hubs only

        if (shouldShowLabel) {
            const fontSize = Math.max(10, 12 / globalScale);
            ctx.font = `${isHub ? 'bold ' : ''}${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillStyle = isHub ? '#ffffff' : '#e2e8f0';
            ctx.shadowBlur = isHub ? 4 : 0;
            ctx.shadowColor = 'rgba(0,0,0,0.8)';
            ctx.fillText(label, x, y + radius + 3);
        }

        ctx.restore();
    }, [highlightedNodes, data, nodeDegrees, hubThreshold]);

    const formatRelativeTime = (timestamp: number): string => {
        if (!timestamp) return '';
        const date = new Date(timestamp * 1000);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    return (
        <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', color: '#fff' }}>
            {/* Search Bar */}
            <div style={{ padding: '15px', borderBottom: '1px solid #334155' }}>
                <form onSubmit={handleSearch} style={{ display: 'flex', gap: '10px' }}>
                    <input
                        type="text"
                        value={searchCue}
                        onChange={e => setSearchCue(e.target.value)}
                        placeholder="Search to zoom & highlight (leave empty to see all)"
                        style={{
                            flex: 1,
                            padding: '10px 15px',
                            borderRadius: '6px',
                            border: '1px solid #334155',
                            background: '#1e293b',
                            color: 'white',
                            fontSize: '0.9rem',
                            outline: 'none'
                        }}
                    />
                    <button type="submit" disabled={loading} style={{
                        padding: '0 20px',
                        borderRadius: '6px',
                        background: '#f59e0b',
                        border: 'none',
                        color: 'black',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontWeight: 600
                    }}>
                        {loading ? 'Loading...' : <><Search size={18} /> {searchCue ? 'Focus' : 'Reset'}</>}
                    </button>
                </form>
            </div>

            {/* Main Content */}
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                {/* Graph Visualization */}
                <div ref={containerRef} style={{ flex: 2, minWidth: 0, background: '#0f172a', position: 'relative' }}>
                    {graphData.nodes.length > 0 ? (
                        <ForceGraph2D
                            ref={fgRef}
                            width={dimensions.width}
                            height={dimensions.height}
                            graphData={graphData}
                            nodeLabel="label"
                            nodeCanvasObjectMode={() => 'replace'}
                            nodeCanvasObject={drawNode}
                            linkColor={() => highlightedNodes.size > 0 ? 'rgba(148, 163, 184, 0.1)' : 'rgba(148, 163, 184, 0.3)'}
                            linkWidth={1}
                            onNodeClick={handleNodeClick}
                            backgroundColor="#0f172a"
                            cooldownTicks={100}
                            onEngineStop={() => fgRef.current?.zoomToFit(400, 50)}
                            d3VelocityDecay={0.4}
                            d3AlphaDecay={0.02}
                        />
                    ) : (
                        <div style={{
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: '#64748b',
                            fontStyle: 'italic'
                        }}>
                            Loading Lexicon...
                        </div>
                    )}

                    {/* Stats overlay */}
                    <div style={{
                        position: 'absolute',
                        top: '10px',
                        left: '10px',
                        background: 'rgba(15, 23, 42, 0.9)',
                        padding: '8px 12px',
                        borderRadius: '6px',
                        fontSize: '0.75rem',
                        color: '#94a3b8'
                    }}>
                        {graphData.nodes.length} nodes · {graphData.links.length} links
                        {highlightedNodes.size > 0 && ` · ${highlightedNodes.size} highlighted`}
                    </div>

                    {/* Legend */}
                    <div style={{
                        position: 'absolute',
                        bottom: '10px',
                        left: '10px',
                        background: 'rgba(15, 23, 42, 0.9)',
                        padding: '10px',
                        borderRadius: '6px',
                        fontSize: '0.75rem'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#f59e0b' }}></span>
                            <span>Searched Cue</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#3b82f6' }}></span>
                            <span>Token</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#22c55e' }}></span>
                            <span>Canonical</span>
                        </div>
                    </div>
                </div>

                {/* Results Panel */}
                <div style={{
                    flex: 1,
                    minWidth: 280,
                    maxWidth: 400,
                    background: '#1e293b',
                    borderLeft: '1px solid #334155',
                    padding: '15px',
                    overflowY: 'auto'
                }}>
                    <h3 style={{
                        marginTop: 0,
                        marginBottom: '15px',
                        color: '#94a3b8',
                        fontSize: '0.85rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                    }}>
                        {data ? `"${data.cue}" Connections` : 'Lexicon Entry'}
                    </h3>

                    {selectedEntry ? (
                        <div style={{
                            background: '#0f172a',
                            borderRadius: '8px',
                            padding: '15px',
                            borderLeft: '4px solid #f59e0b'
                        }}>
                            <div style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '8px' }}>
                                {selectedEntry.token}
                            </div>
                            <div style={{ fontSize: '1rem', color: '#e2e8f0', marginBottom: '12px' }}>
                                → {selectedEntry.content}
                            </div>
                            <div style={{
                                display: 'flex',
                                gap: '12px',
                                fontSize: '0.7rem',
                                color: '#64748b',
                                borderTop: '1px solid #334155',
                                paddingTop: '10px'
                            }}>
                                <span title="Reinforcement score - how many times this mapping has been used">
                                    Used {Math.round(selectedEntry.reinforcement_score)}x
                                </span>
                                <span>{formatRelativeTime(selectedEntry.created_at)}</span>
                            </div>

                            {/* Impact warning - shows main memories that would be affected */}
                            {selectedEntry.affected_memories_count > 0 && (
                                <div style={{
                                    marginTop: '8px',
                                    padding: '6px 8px',
                                    background: 'rgba(239, 68, 68, 0.1)',
                                    borderRadius: '4px',
                                    fontSize: '0.7rem',
                                    color: '#ef4444'
                                }}>
                                    ⚠️ Unwiring affects {selectedEntry.affected_memories_count} main memor{selectedEntry.affected_memories_count === 1 ? 'y' : 'ies'}
                                </div>
                            )}

                            {/* Actions */}
                            <div style={{
                                display: 'flex',
                                gap: '8px',
                                marginTop: '15px'
                            }}>
                                <button onClick={handleUnwire} style={{
                                    flex: 1,
                                    padding: '8px',
                                    borderRadius: '4px',
                                    border: '1px solid #ef4444',
                                    background: 'transparent',
                                    color: '#ef4444',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '4px',
                                    fontSize: '0.75rem'
                                }}>
                                    <Trash2 size={14} /> Unwire
                                </button>
                            </div>
                        </div>
                    ) : data ? (
                        <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                            <div style={{ marginBottom: '12px' }}>
                                <strong style={{ color: '#3b82f6' }}>Incoming</strong> ({data.incoming.length}):
                                <div style={{ marginTop: '4px', color: '#64748b' }}>
                                    {data.incoming.slice(0, 5).map(e => e.token).join(', ')}
                                    {data.incoming.length > 5 && '...'}
                                </div>
                            </div>
                            <div>
                                <strong style={{ color: '#22c55e' }}>Outgoing</strong> ({data.outgoing.length}):
                                <div style={{ marginTop: '4px', color: '#64748b' }}>
                                    {data.outgoing.slice(0, 5).map(e => e.content).join(', ')}
                                    {data.outgoing.length > 5 && '...'}
                                </div>
                            </div>
                            <div style={{ marginTop: '12px', fontStyle: 'italic', color: '#475569' }}>
                                Click a highlighted node to see details
                            </div>
                        </div>
                    ) : (
                        <div style={{ color: '#475569', fontStyle: 'italic' }}>
                            Search for a cue to see its connections
                        </div>
                    )}

                    {/* Wire New Connection Form */}
                    <div style={{
                        marginTop: '20px',
                        padding: '15px',
                        background: '#0f172a',
                        borderRadius: '8px',
                        borderLeft: '4px solid #22c55e'
                    }}>
                        <h4 style={{
                            marginTop: 0,
                            marginBottom: '6px',
                            color: '#22c55e',
                            fontSize: '0.8rem',
                            textTransform: 'uppercase'
                        }}>
                            Wire New Connection
                        </h4>
                        <div style={{ fontSize: '0.65rem', color: '#64748b', marginBottom: '10px' }}>
                            💡 Ctrl+Click nodes to fill: 1st = canonical, 2nd = token
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            <input
                                type="text"
                                placeholder="Token (e.g., 'favourite')"
                                value={wireToken}
                                onChange={e => setWireToken(e.target.value)}
                                list="node-suggestions"
                                autoComplete="off"
                                style={{
                                    padding: '8px 12px',
                                    borderRadius: '4px',
                                    border: wireToken ? '1px solid #22c55e' : '1px solid #334155',
                                    background: '#1e293b',
                                    color: 'white',
                                    fontSize: '0.8rem'
                                }}
                            />
                            <input
                                type="text"
                                placeholder="Canonical (e.g., 'favorite')"
                                value={wireCanonical}
                                onChange={e => setWireCanonical(e.target.value)}
                                list="node-suggestions"
                                autoComplete="off"
                                style={{
                                    padding: '8px 12px',
                                    borderRadius: '4px',
                                    border: wireCanonical ? '1px solid #f59e0b' : '1px solid #334155',
                                    background: '#1e293b',
                                    color: 'white',
                                    fontSize: '0.8rem'
                                }}
                            />

                            {/* Synonym Suggestions */}
                            {synonyms && (synonyms.existing.length > 0 || synonyms.new.length > 0) && (
                                <div style={{
                                    marginTop: '4px',
                                    padding: '8px',
                                    background: 'rgba(30, 41, 59, 0.5)',
                                    borderRadius: '4px',
                                    border: '1px solid #334155'
                                }}>
                                    <div style={{ fontSize: '0.65rem', color: '#94a3b8', marginBottom: '4px' }}>
                                        🤖 WordNet Suggestions:
                                    </div>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                        {/* Existing nodes (Green-ish) */}
                                        {synonyms.existing.map(syn => (
                                            <button
                                                key={syn}
                                                onClick={() => setWireToken(syn)}
                                                title="Exists in graph - Click to use as Token"
                                                style={{
                                                    fontSize: '0.65rem',
                                                    padding: '2px 6px',
                                                    borderRadius: '10px',
                                                    border: '1px solid #22c55e',
                                                    background: 'rgba(34, 197, 94, 0.1)',
                                                    color: '#86efac',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                {syn}
                                            </button>
                                        ))}
                                        {/* New nodes (Blue-ish) */}
                                        {synonyms.new.map(syn => (
                                            <button
                                                key={syn}
                                                onClick={() => setWireToken(syn)}
                                                title="New term - Click to use as Token"
                                                style={{
                                                    fontSize: '0.65rem',
                                                    padding: '2px 6px',
                                                    borderRadius: '10px',
                                                    border: '1px solid #3b82f6',
                                                    background: 'rgba(59, 130, 246, 0.1)',
                                                    color: '#93c5fd',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                {syn}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {/* Datalist for autocomplete suggestions */}
                            <datalist id="node-suggestions">
                                {graphData.nodes.slice(0, 200).map(n => (
                                    <option key={n.id} value={n.id} />
                                ))}
                            </datalist>
                            <button
                                onClick={handleWire}
                                disabled={!wireToken.trim() || !wireCanonical.trim()}
                                style={{
                                    padding: '8px',
                                    borderRadius: '4px',
                                    border: 'none',
                                    background: wireToken.trim() && wireCanonical.trim() ? '#22c55e' : '#334155',
                                    color: wireToken.trim() && wireCanonical.trim() ? 'black' : '#64748b',
                                    cursor: wireToken.trim() && wireCanonical.trim() ? 'pointer' : 'not-allowed',
                                    fontSize: '0.8rem',
                                    fontWeight: 600
                                }}
                            >
                                Wire Connection
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LexiconSurgeon;
