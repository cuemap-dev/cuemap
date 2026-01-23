import React, { useState, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Link2, Upload, FileText, CheckCircle, XCircle, Loader2, ArrowRight } from 'lucide-react';

const MiniGraph: React.FC<{
    endpoint: string,
    projectId: string,
    refreshTrigger: number,
    nodeColor: string,
    title: string,
    statsKey: 'total_memories' | 'total_cues'
}> = ({ endpoint, projectId, refreshTrigger, nodeColor, title, statsKey }) => {
    const [data, setData] = useState({ nodes: [], links: [] });
    const [count, setCount] = useState(0);
    const [dimensions, setDimensions] = useState({ width: 400, height: 500 });
    const fgRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Auto-resize to fill container
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

    // Fetch accurate count from /stats
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const headers: Record<string, string> = {};
                if (projectId) headers['X-Project-ID'] = projectId;

                const res = await fetch('/stats', { headers });
                const json = await res.json();
                if (json[statsKey] !== undefined) {
                    setCount(json[statsKey]);
                }
            } catch (err) {
                console.error('Stats fetch error:', err);
            }
        };
        fetchStats();
    }, [projectId, refreshTrigger, statsKey]);

    // Fetch graph data (limited for performance)
    useEffect(() => {
        const fetchData = async () => {
            try {
                const headers: Record<string, string> = {};
                if (projectId) headers['X-Project-ID'] = projectId;

                const url = endpoint.includes('?') ? `${endpoint}&limit=500` : `${endpoint}?limit=500`;
                const res = await fetch(url, { headers });
                const json = await res.json();

                if (json.nodes) {
                    setData(json);
                    if (fgRef.current) {
                        // Stronger repulsion for dense graphs
                        fgRef.current.d3Force('charge').strength(-100);
                        fgRef.current.d3Force('link').distance(180);
                    }
                }
            } catch (err) {
                console.error(err);
            }
        };
        fetchData();
    }, [endpoint, projectId, refreshTrigger]);

    return (
        <div
            ref={containerRef}
            style={{
                width: '100%',
                height: '100%',
                background: '#0f172a',
                borderRadius: '16px',
                border: '1px solid #334155',
                overflow: 'hidden',
                position: 'relative'
            }}
        >
            <div style={{
                position: 'absolute',
                top: 10,
                left: 10,
                zIndex: 10,
                background: 'rgba(15, 23, 42, 0.8)',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid rgba(148, 163, 184, 0.1)',
                color: '#e2e8f0',
                fontSize: '0.8rem',
                fontWeight: 600
            }}>
                {title} <span style={{ color: '#94a3b8', fontWeight: 400 }}>({count})</span>
            </div>

            <ForceGraph2D
                ref={fgRef}
                width={dimensions.width}
                height={dimensions.height}
                graphData={data}
                nodeColor={() => nodeColor}
                nodeRelSize={4}
                linkColor={() => 'rgba(255,255,255,0.1)'}
                backgroundColor="#0f172a"
                d3AlphaDecay={0.02}
                d3VelocityDecay={0.3}
                onEngineStop={() => {
                    if (fgRef.current) {
                        fgRef.current.zoomToFit(400, 50);
                    }
                }}
            />
        </div>
    );
};

interface IngestionScreenProps {
    projectId: string;
    onComplete: () => void;
}

interface IngestionItem {
    id: string;
    type: 'url' | 'file' | 'text';
    label: string;
    status: 'pending' | 'processing' | 'success' | 'error';
    error?: string;
}

const IngestionScreen: React.FC<IngestionScreenProps> = ({ projectId, onComplete }) => {
    const [activeMode, setActiveMode] = useState<'url' | 'file' | 'text'>('url');
    const [urlInput, setUrlInput] = useState('');
    const [textInput, setTextInput] = useState('');
    const [items, setItems] = useState<IngestionItem[]>([]);
    const [isIngesting, setIsIngesting] = useState(false);

    // Crawl options for URL mode
    const [crawlDepth, setCrawlDepth] = useState<number>(0);
    const [sameDomainOnly, setSameDomainOnly] = useState<boolean>(true);

    const [refreshTrigger, setRefreshTrigger] = useState(0);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Job progress tracking
    interface JobProgress {
        phase: string;
        writes_completed: number;
        writes_total: number;
        propose_cues_completed: number;
        propose_cues_total: number;
        train_lexicon_completed: number;
        train_lexicon_total: number;
        update_graph_completed: number;
        update_graph_total: number;
    }
    const [jobProgress, setJobProgress] = useState<JobProgress | null>(null);
    const [showJobProgress, setShowJobProgress] = useState(false);
    const [lastIngestFilename, setLastIngestFilename] = useState<string>('');

    // Auto-refresh graphs every 10s
    useEffect(() => {
        const interval = setInterval(() => {
            setRefreshTrigger(n => n + 1);
        }, 10000);
        return () => clearInterval(interval);
    }, []);

    // Poll job progress every 5 seconds when ingestion is done
    useEffect(() => {
        if (!showJobProgress) return;

        const pollProgress = async () => {
            try {
                const res = await fetch('/jobs/status', {
                    headers: { 'X-Project-ID': projectId }
                });
                const data = await res.json();
                setJobProgress(data);

                // Stop polling when all jobs are done
                if (data.phase === 'done' || data.phase === 'idle') {
                    setTimeout(() => {
                        setShowJobProgress(false);
                        setRefreshTrigger(n => n + 1); // Refresh graphs
                    }, 2000);
                }
            } catch (err) {
                console.error('Job status fetch error:', err);
            }
        };

        pollProgress(); // Initial fetch
        const interval = setInterval(pollProgress, 5000);
        return () => clearInterval(interval);
    }, [showJobProgress, projectId]);

    const updateItemStatus = (id: string, status: IngestionItem['status'], error?: string) => {
        setItems(prev => prev.map(item =>
            item.id === id ? { ...item, status, error } : item
        ));
    };

    const ingestUrl = async (url: string): Promise<void> => {
        const payload: any = { url };
        if (crawlDepth > 0) {
            payload.depth = crawlDepth;
            payload.same_domain_only = sameDomainOnly;
        }

        const response = await fetch('/ingest/url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Project-ID': projectId,
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to ingest URL');
        }
    };

    const ingestFile = async (file: File): Promise<void> => {
        const extension = file.name.split('.').pop()?.toLowerCase() || '';
        const binaryTypes = ['pdf', 'docx', 'xlsx', 'xls', 'doc'];

        if (binaryTypes.includes(extension)) {
            // Use multipart upload for binary files (PDF, Office docs)
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/ingest/file', {
                method: 'POST',
                headers: {
                    'X-Project-ID': projectId,
                },
                body: formData,
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Failed to ingest file');
            }
        } else {
            // Read text-based files and send as content
            const content = await file.text();

            const response = await fetch('/ingest/content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Project-ID': projectId,
                },
                body: JSON.stringify({
                    content: content,
                    filename: file.name,
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Failed to ingest file');
            }
        }
    };

    const ingestText = async (text: string): Promise<void> => {
        const response = await fetch('/ingest/content', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Project-ID': projectId,
            },
            body: JSON.stringify({
                content: text,
                filename: 'memory.txt', // Treat as plain text
            }),
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to add memory');
        }
    };

    const handleAddUrl = () => {
        if (!urlInput.trim()) return;

        const newItem: IngestionItem = {
            id: `url-${Date.now()}`,
            type: 'url',
            label: urlInput.trim(),
            status: 'pending',
        };
        setItems(prev => [...prev, newItem]);
        setUrlInput('');
    };

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (!files) return;

        const newItems: IngestionItem[] = Array.from(files).map((file, idx) => ({
            id: `file-${Date.now()}-${idx}`,
            type: 'file' as const,
            label: file.name,
            status: 'pending' as const,
            file,
        }));

        setItems(prev => [...prev, ...newItems]);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleAddText = () => {
        if (!textInput.trim()) return;

        const lines = textInput.trim().split('\n').filter(line => line.trim());
        const newItems: IngestionItem[] = lines.map((line, idx) => ({
            id: `text-${Date.now()}-${idx}`,
            type: 'text' as const,
            label: line.length > 60 ? line.substring(0, 60) + '...' : line,
            status: 'pending' as const,
            fullText: line,
        }));

        setItems(prev => [...prev, ...newItems]);
        setTextInput('');
    };

    const startIngestion = async () => {
        if (items.length === 0) return;

        setIsIngesting(true);
        const pendingItems = items.filter(i => i.status === 'pending');

        // Set filename for progress display
        if (pendingItems.length > 0) {
            const firstLabel = pendingItems[0].label;
            setLastIngestFilename(pendingItems.length === 1 ? firstLabel : `${pendingItems.length} items`);
        }

        for (const item of pendingItems) {
            updateItemStatus(item.id, 'processing');

            try {
                if (item.type === 'url') {
                    await ingestUrl(item.label);
                } else if (item.type === 'file') {
                    // Get the file from the original input
                    const fileItem = items.find(i => i.id === item.id) as any;
                    if (fileItem?.file) {
                        await ingestFile(fileItem.file);
                    }
                } else if (item.type === 'text') {
                    const textItem = items.find(i => i.id === item.id) as any;
                    await ingestText(textItem?.fullText || item.label);
                }
                updateItemStatus(item.id, 'success');
            } catch (err: any) {
                updateItemStatus(item.id, 'error', err.message);
            }
        }

        setIsIngesting(false);

        // Enable job progress polling once writes are complete
        if (pendingItems.length > 0) {
            setShowJobProgress(true);
        }
    };

    const allDone = items.length > 0 && items.every(i => i.status === 'success' || i.status === 'error');
    const successCount = items.filter(i => i.status === 'success').length;

    return (
        <div className="ingestion-screen">
            {/* Graph Grid Background */}
            <div className="graph-grid">
                <div className="graph-panel">
                    <MiniGraph
                        endpoint="/graph"
                        projectId={projectId}
                        refreshTrigger={refreshTrigger}
                        nodeColor="#4ade80"
                        title="Memories"
                        statsKey="total_memories"
                    />
                </div>
                <div className="graph-panel">
                    <MiniGraph
                        endpoint="/lexicon/graph"
                        projectId={projectId}
                        refreshTrigger={refreshTrigger}
                        nodeColor="#60a5fa"
                        title="Lexicon"
                        statsKey="total_cues"
                    />
                </div>
            </div>

            {/* Floating Ingest Form */}
            <div className="ingestion-container">
                <div className="ingestion-header">
                    <h1>Ingest Content</h1>
                    <p>Add URLs, files, or text memories to your brain.</p>
                </div>

                {/* Job Progress Bar */}
                {showJobProgress && jobProgress && (
                    <div className="job-progress-bar">
                        <span className="progress-filename">{lastIngestFilename}</span>
                        <span className="progress-separator">|</span>
                        <span className="progress-stat">{jobProgress.writes_total} chunks processed</span>
                        <span className="progress-separator">|</span>
                        <span className={`progress-task ${jobProgress.propose_cues_completed >= jobProgress.writes_total ? 'completed' :
                            jobProgress.propose_cues_completed > 0 ? 'in-progress' : 'waiting'
                            }`}>
                            Expanding cues: {jobProgress.propose_cues_completed}/{jobProgress.writes_total}
                        </span>
                        <span className="progress-separator">|</span>
                        <span className={`progress-task ${jobProgress.train_lexicon_completed >= jobProgress.writes_total ? 'completed' :
                            jobProgress.train_lexicon_completed > 0 ? 'in-progress' : 'waiting'
                            }`}>
                            Training lexicon: {jobProgress.train_lexicon_completed}/{jobProgress.writes_total}
                        </span>
                        <span className="progress-separator">|</span>
                        <span className={`progress-task ${jobProgress.update_graph_completed >= jobProgress.writes_total ? 'completed' :
                            jobProgress.update_graph_completed > 0 ? 'in-progress' : 'waiting'
                            }`}>
                            Updating graph: {jobProgress.update_graph_completed}/{jobProgress.writes_total}
                        </span>
                        {jobProgress.phase === 'done' && (
                            <span className="progress-done">✓</span>
                        )}
                    </div>
                )}

                {/* Mode Selector */}
                <div className="mode-selector">
                    <button
                        className={`mode-btn ${activeMode === 'url' ? 'active' : ''}`}
                        onClick={() => setActiveMode('url')}
                    >
                        <Link2 size={18} /> URL
                    </button>
                    <button
                        className={`mode-btn ${activeMode === 'file' ? 'active' : ''}`}
                        onClick={() => setActiveMode('file')}
                    >
                        <Upload size={18} /> File
                    </button>
                    <button
                        className={`mode-btn ${activeMode === 'text' ? 'active' : ''}`}
                        onClick={() => setActiveMode('text')}
                    >
                        <FileText size={18} /> Text
                    </button>
                </div>

                {/* Input Area */}
                <div className="input-area">
                    {activeMode === 'url' && (
                        <div className="url-input">
                            <input
                                type="url"
                                value={urlInput}
                                onChange={e => setUrlInput(e.target.value)}
                                placeholder="https://docs.example.com/intro"
                                onKeyDown={e => e.key === 'Enter' && handleAddUrl()}
                            />
                            <button onClick={handleAddUrl} disabled={!urlInput.trim()}>
                                Add URL
                            </button>

                            {/* Crawl Options */}
                            <div className="crawl-options">
                                <div className="depth-selector">
                                    <label>Crawl Depth:</label>
                                    <select
                                        value={crawlDepth}
                                        onChange={e => setCrawlDepth(Number(e.target.value))}
                                    >
                                        <option value={0}>Single Page</option>
                                        <option value={1}>Depth 1 (follow links)</option>
                                        <option value={2}>Depth 2 (2 levels)</option>
                                        <option value={3}>Depth 3 (full docs)</option>
                                    </select>
                                </div>
                                {crawlDepth > 0 && (
                                    <label className="checkbox-option">
                                        <input
                                            type="checkbox"
                                            checked={sameDomainOnly}
                                            onChange={e => setSameDomainOnly(e.target.checked)}
                                        />
                                        Same domain only
                                    </label>
                                )}
                            </div>
                        </div>
                    )}

                    {activeMode === 'file' && (
                        <div className="file-input">
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleFileSelect}
                                multiple
                                accept=".txt,.md,.json,.yaml,.yml,.csv,.py,.js,.ts,.rs,.go,.java,.html,.css,.pdf,.docx,.xlsx"
                            />
                            <div className="file-drop-zone" onClick={() => fileInputRef.current?.click()}>
                                <Upload size={32} />
                                <p>Click to select files or drag & drop</p>
                            </div>
                        </div>
                    )}

                    {activeMode === 'text' && (
                        <div className="text-input">
                            <textarea
                                value={textInput}
                                onChange={e => setTextInput(e.target.value)}
                                placeholder="Enter memories (one per line):

I love coffee in the morning
Paris is my favorite city
Python is great for data science"
                                rows={6}
                            />
                            <button onClick={handleAddText} disabled={!textInput.trim()}>
                                Add Memories
                            </button>
                        </div>
                    )}
                </div>

                {/* Queue */}
                {items.length > 0 && (
                    <div className="ingestion-queue">
                        <h3>📋 Ingestion Queue ({items.length} items)</h3>
                        <div className="queue-list">
                            {items.map(item => (
                                <div key={item.id} className={`queue-item ${item.status}`}>
                                    <span className="item-type">
                                        {item.type === 'url' && <Link2 size={14} />}
                                        {item.type === 'file' && <Upload size={14} />}
                                        {item.type === 'text' && <FileText size={14} />}
                                    </span>
                                    <span className="item-label">{item.label}</span>
                                    <span className="item-status">
                                        {item.status === 'pending' && '⏳'}
                                        {item.status === 'processing' && <Loader2 size={14} className="spin" />}
                                        {item.status === 'success' && <CheckCircle size={14} color="#22c55e" />}
                                        {item.status === 'error' && <XCircle size={14} color="#ef4444" />}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Actions */}
                <div className="ingestion-actions">
                    {!allDone ? (
                        <button
                            className="ingest-btn"
                            onClick={startIngestion}
                            disabled={isIngesting || items.filter(i => i.status === 'pending').length === 0}
                        >
                            {isIngesting ? (
                                <>
                                    <Loader2 size={18} className="spin" />
                                    Ingesting...
                                </>
                            ) : (
                                <>
                                    Start Ingestion ({items.filter(i => i.status === 'pending').length} pending)
                                </>
                            )}
                        </button>
                    ) : (
                        <button className="continue-btn" onClick={onComplete}>
                            Continue to Recall <ArrowRight size={18} />
                            <span className="success-count">
                                ({jobProgress?.writes_total || successCount} chunks processed)
                            </span>
                        </button>
                    )}
                </div>
            </div>

            <style>{`
                .ingestion-screen {
                    min-height: 100vh;
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    position: relative;
                    overflow: hidden;
                }

                .graph-grid {
                    position: absolute;
                    inset: 0;
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 0;
                }

                .graph-panel {
                    height: 100vh;
                    overflow: hidden;
                }

                .ingestion-container {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(30, 41, 59, 0.5);
                    backdrop-filter: blur(12px);
                    border-radius: 16px;
                    padding: 32px;
                    max-width: 600px;
                    width: 90%;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
                    border: 1px solid #475569;
                    z-index: 100;
                    max-height: 85vh;
                    overflow-y: auto;
                }

                .ingestion-header {
                    text-align: center;
                    margin-bottom: 24px;
                }

                .ingestion-header h1 {
                    color: white;
                    margin: 0 0 8px 0;
                    font-size: 1.8rem;
                }

                .ingestion-header p {
                    color: #94a3b8;
                    margin: 0;
                }

                .mode-selector {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 20px;
                    background: #0f172a;
                    padding: 6px;
                    border-radius: 10px;
                }

                .mode-btn {
                    flex: 1;
                    padding: 12px 16px;
                    border: none;
                    border-radius: 8px;
                    background: transparent;
                    color: #94a3b8;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    font-size: 0.95rem;
                    font-weight: 500;
                    transition: all 0.2s;
                }

                .mode-btn:hover {
                    color: white;
                }

                .mode-btn.active {
                    background: #3b82f6;
                    color: white;
                }

                .input-area {
                    margin-bottom: 20px;
                }

                .url-input, .text-input {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }

                .url-input input, .text-input textarea {
                    width: 100%;
                    padding: 14px 16px;
                    border-radius: 8px;
                    border: 1px solid #334155;
                    background: #0f172a;
                    color: white;
                    font-size: 1rem;
                    outline: none;
                    box-sizing: border-box;
                }

                .url-input input:focus, .text-input textarea:focus {
                    border-color: #3b82f6;
                }

                .text-input textarea {
                    resize: vertical;
                    min-height: 120px;
                    font-family: inherit;
                }

                .url-input button, .text-input button {
                    padding: 12px 20px;
                    border-radius: 8px;
                    border: none;
                    background: #3b82f6;
                    color: white;
                    font-weight: 500;
                    cursor: pointer;
                    transition: background 0.2s;
                    align-self: flex-end;
                }

                .url-input button:hover:not(:disabled), .text-input button:hover:not(:disabled) {
                    background: #2563eb;
                }

                .url-input button:disabled, .text-input button:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                .crawl-options {
                    display: flex;
                    gap: 16px;
                    align-items: center;
                    padding: 12px;
                    background: rgba(15, 23, 42, 0.6);
                    border-radius: 8px;
                    border: 1px solid #334155;
                }

                .depth-selector {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .depth-selector label {
                    color: #94a3b8;
                    font-size: 0.9rem;
                }

                .depth-selector select {
                    padding: 8px 12px;
                    border-radius: 6px;
                    border: 1px solid #334155;
                    background: #0f172a;
                    color: white;
                    font-size: 0.9rem;
                    cursor: pointer;
                }

                .depth-selector select:hover {
                    border-color: #3b82f6;
                }

                .checkbox-option {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    color: #94a3b8;
                    font-size: 0.9rem;
                    cursor: pointer;
                }

                .checkbox-option input[type="checkbox"] {
                    width: 16px;
                    height: 16px;
                    accent-color: #3b82f6;
                }

                .file-input input[type="file"] {
                    display: none;
                }

                .file-drop-zone {
                    border: 2px dashed #334155;
                    border-radius: 12px;
                    padding: 20px 20px;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.2s;
                    color: #94a3b8;
                }

                .file-drop-zone:hover {
                    border-color: #3b82f6;
                    background: rgba(59, 130, 246, 0.05);
                }

                .file-drop-zone p {
                    margin: 12px 0 8px;
                    color: white;
                }

                .file-types {
                    font-size: 0.8rem;
                    color: #64748b;
                }

                .ingestion-queue {
                    background: #0f172a;
                    border-radius: 10px;
                    padding: 16px;
                    margin-bottom: 20px;
                }

                .ingestion-queue h3 {
                    color: white;
                    margin: 0 0 12px 0;
                    font-size: 0.95rem;
                }

                .queue-list {
                    max-height: 200px;
                    overflow-y: auto;
                }

                .queue-item {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 10px 12px;
                    background: #1e293b;
                    border-radius: 6px;
                    margin-bottom: 6px;
                    font-size: 0.9rem;
                }

                .queue-item.processing {
                    border-left: 3px solid #3b82f6;
                }

                .queue-item.success {
                    border-left: 3px solid #22c55e;
                }

                .queue-item.error {
                    border-left: 3px solid #ef4444;
                }

                .item-type {
                    color: #64748b;
                    display: flex;
                }

                .item-label {
                    flex: 1;
                    color: #e2e8f0;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }

                .item-status {
                    display: flex;
                    align-items: center;
                }

                .spin {
                    animation: spin 1s linear infinite;
                }

                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }

                .ingestion-actions {
                    text-align: center;
                }

                .ingest-btn, .continue-btn {
                    padding: 14px 28px;
                    border-radius: 10px;
                    border: none;
                    font-size: 1rem;
                    font-weight: 600;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    transition: all 0.2s;
                }

                .ingest-btn {
                    background: #3b82f6;
                    color: white;
                }

                .ingest-btn:hover:not(:disabled) {
                    background: #2563eb;
                }

                .ingest-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                .continue-btn {
                    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
                    color: white;
                }

                .continue-btn:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
                }

                .success-count {
                    font-weight: 400;
                    opacity: 0.9;
                }

                /* Job Progress Bar */
                .job-progress-bar {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px 16px;
                    background: rgba(15, 23, 42, 0.95);
                    border: 1px solid #334155;
                    border-radius: 8px;
                    margin-bottom: 16px;
                    font-size: 0.875rem;
                    flex-wrap: wrap;
                }

                .progress-filename {
                    color: #f1f5f9;
                    font-weight: 500;
                    max-width: 200px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                .progress-separator {
                    color: #475569;
                }

                .progress-stat {
                    color: #94a3b8;
                }

                .progress-task {
                    font-weight: 500;
                }

                .progress-task.completed {
                    color: #4ade80;
                }

                .progress-task.in-progress {
                    color: #facc15;
                }

                .progress-task.waiting {
                    color: #64748b;
                }

                .progress-done {
                    color: #4ade80;
                    font-size: 1rem;
                    margin-left: 4px;
                }
            `}</style>
        </div>
    );
};

export default IngestionScreen;
