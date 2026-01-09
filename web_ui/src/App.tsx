import React, { useState } from 'react';
import { Search, Brain, BookOpen } from 'lucide-react';
import GraphVisualizer from './components/GraphVisualizer';
import Inspector from './components/Inspector';
import LexiconSurgeon from './components/LexiconSurgeon';
import './App.css';

type Tab = 'recall' | 'lexicon';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('recall');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState(0);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const params = new URLSearchParams(window.location.search);
      const projectId = params.get('project');
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
          explain: true
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
        <nav style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
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

      {activeTab === 'recall' ? (
        <>
          {/* 2. Search Row (Top) */}
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
            <div className="graph-pane">
              <GraphVisualizer highlightedMemoryMap={highlightedMemoryMap} />
            </div>

            <div className="results-pane">
              <Inspector results={results} latency={latency} />
            </div>
          </main>
        </>
      ) : (
        <main className="app-main" style={{ flex: 1 }}>
          <LexiconSurgeon />
        </main>
      )}
    </div>
  );
}

export default App;
