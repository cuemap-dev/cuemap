import React, { useState } from 'react';
import { Search, Brain, BookOpen, Upload } from 'lucide-react';
import GraphVisualizer from './components/GraphVisualizer';
import Inspector from './components/Inspector';
import LexiconSurgeon from './components/LexiconSurgeon';

import IngestionScreen from './components/IngestionScreen';
import './App.css';

type Tab = 'ingest' | 'recall' | 'lexicon';

function App() {
  const [projectId, setProjectId] = useState<string | null>(() => {
    return new URLSearchParams(window.location.search).get('project');
  });
  const [activeTab, setActiveTab] = useState<Tab>('ingest');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState(0);
  const [fastMode, setFastMode] = useState(false);
  const [projectsList, setProjectsList] = useState<any[]>([]);
  const [stats, setStats] = useState<{ total_memories: number; total_cues: number } | null>(null);

  // Fetch projects on mount
  React.useEffect(() => {
    fetch('/projects')
      .then(res => res.json())
      .then(data => setProjectsList(data.projects || []))
      .catch(console.error);
  }, []);

  // Fetch stats when entering recall or changing project
  React.useEffect(() => {
    if (activeTab === 'recall') {
      const headers: Record<string, string> = {};
      if (projectId) headers['X-Project-ID'] = projectId;

      fetch('/stats', { headers })
        .then(res => res.json())
        .then(data => setStats(data))
        .catch(console.error);
    }
  }, [activeTab, projectId, results]); // Refresh on results change (after search/ingest)

  const handleProjectChange = (id: string) => {
    setProjectId(id);
    setActiveTab('ingest');
    const url = new URL(window.location.href);
    url.searchParams.set('project', id);
    window.history.pushState({}, '', url);
  };

  const handleCreateProject = () => {
    const id = prompt("Enter new Project ID (alphanumeric, 3-64 chars):");
    if (id && /^[a-zA-Z0-9-_]{3,64}$/.test(id)) {
      // Optimistically add to list and select it
      // In multi-tenant mode, project is created on first write, but we can switch context immediately
      if (!projectsList.some(p => p.project_id === id)) {
        setProjectsList(prev => [...prev, { project_id: id }]);
      }
      handleProjectChange(id);
    } else if (id) {
      alert("Invalid Project ID. Use 3-64 alphanumeric characters, hyphens, or underscores.");
    }
  };

  const handleIngestionComplete = () => {
    setActiveTab('recall');
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (projectId) {
        headers['X-Project-ID'] = projectId;
      }

      const res = await fetch('/recall', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          query_text: query,
          limit: 5,
          explain: true,
          fast_mode: fastMode
        })
      });
      const data = await res.json();
      setResults(data.results || []);
      setLatency(data.engine_latency || 0);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Extract memory IDs for highlighting with rank
  const highlightedMemoryMap = new Map(results.map((r, index) => [r.memory_id, index + 1]));

  return (
    <div className="app-container">
      {/* 1. Header Row with Tabs */}
      <header className="app-header">
        <h1>CueMap <span className="beta-tag">BRAIN</span></h1>

        {/* Project Selector */}
        <div className="project-selector" style={{ marginLeft: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <select
            value={projectId || ''}
            onChange={(e) => handleProjectChange(e.target.value)}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              background: '#334155',
              color: 'white',
              border: '1px solid #475569',
              fontSize: '0.9rem'
            }}
          >
            <option value="" disabled>Select Project</option>
            {projectsList.map((p: any) => (
              <option key={p.project_id} value={p.project_id}>{p.project_id}</option>
            ))}
          </select>
          <button
            onClick={handleCreateProject}
            title="Create New Project"
            style={{
              background: '#3b82f6',
              border: 'none',
              borderRadius: '4px',
              color: 'white',
              width: '28px',
              height: '28px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              fontSize: '1.2rem',
              lineHeight: 1
            }}
          >
            +
          </button>
        </div>

        <nav style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
          <button
            onClick={() => setActiveTab('ingest')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              background: activeTab === 'ingest' ? '#22c55e' : 'transparent',
              color: activeTab === 'ingest' ? 'white' : '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.9rem',
              fontWeight: 500
            }}
          >
            <Upload size={18} /> Ingest
          </button>
          <button
            onClick={() => setActiveTab('recall')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              background: activeTab === 'recall' ? '#3b82f6' : 'transparent',
              color: activeTab === 'recall' ? 'white' : '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.9rem',
              fontWeight: 500
            }}
          >
            <Brain size={18} /> Recall
          </button>
          <button
            onClick={() => setActiveTab('lexicon')}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              background: activeTab === 'lexicon' ? '#f59e0b' : 'transparent',
              color: activeTab === 'lexicon' ? 'black' : '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.9rem',
              fontWeight: 500
            }}
          >
            <BookOpen size={18} /> Lexicon
          </button>
        </nav>
      </header>

      {!projectId ? (
        <main className="app-main" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
          <div style={{ textAlign: 'center' }}>
            <Brain size={64} style={{ marginBottom: '16px', opacity: 0.5 }} />
            <h2>Select or Create a Project</h2>
            <p>Choose a project from the dropdown or click "+" to create one.</p>
          </div>
        </main>
      ) : (
        <>
          {activeTab === 'ingest' && (
            <main className="app-main" style={{ flex: 1, overflow: 'auto' }}>
              <IngestionScreen projectId={projectId} onComplete={handleIngestionComplete} />
            </main>
          )}

          {activeTab === 'recall' && (
            <>
              <div className="search-bar-container">
                <form onSubmit={handleSearch} style={{ display: 'flex', gap: '10px', width: '100%' }}>
                  <input
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="Ask the Brain... (e.g., 'favorite drinks')"
                    style={{
                      flex: 1,
                      padding: '12px 20px',
                      borderRadius: '8px',
                      border: '1px solid #334155',
                      background: '#1e293b',
                      color: 'white',
                      fontSize: '1rem',
                      outline: 'none',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setFastMode(!fastMode)}
                    title={fastMode ? "Fast Mode: O(1) Lookup (Exact Match)" : "Normal Mode: Semantic Search + Pattern Completion"}
                    style={{
                      padding: '0 16px',
                      borderRadius: '8px',
                      background: fastMode ? '#eab308' : '#334155',
                      border: 'none',
                      color: 'white',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                      transition: 'all 0.2s',
                      fontWeight: 500
                    }}>
                    <span style={{ fontSize: '1.2rem' }}>⚡</span>
                    {fastMode ? 'Fast' : 'Normal'}
                  </button>
                  <button type="submit" disabled={loading} style={{
                    padding: '0 24px',
                    borderRadius: '8px',
                    background: '#3b82f6',
                    border: 'none',
                    color: 'white',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.2s'
                  }}>
                    {loading ? <span className="loader"></span> : <Search size={24} />}
                  </button>
                </form>
              </div>

              {/* 3. Main Content Row (Graph + Results) */}
              <main className="app-main split-view">
                <div className="graph-pane" style={{ position: 'relative' }}>
                  <GraphVisualizer highlightedMemoryMap={highlightedMemoryMap} />

                  {/* Stats Overlay */}
                  {stats && (
                    <div style={{
                      position: 'absolute',
                      top: '10px',
                      left: '10px',
                      background: 'rgba(15, 23, 42, 0.9)',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      fontSize: '0.75rem',
                      color: '#94a3b8',
                      pointerEvents: 'none', // Allow clicks to pass through to graph
                      border: '1px solid rgba(148, 163, 184, 0.1)'
                    }}>
                      {stats.total_memories.toLocaleString()} memories · {stats.total_cues.toLocaleString()} cues
                    </div>
                  )}
                </div>

                <div className="results-pane">
                  <Inspector results={results} latency={latency} />
                </div>
              </main>
            </>
          )}

          {activeTab === 'lexicon' && (
            <main className="app-main" style={{ flex: 1 }}>
              <LexiconSurgeon projectId={projectId} />
            </main>
          )}
        </>
      )}
    </div>
  );
}

export default App;

