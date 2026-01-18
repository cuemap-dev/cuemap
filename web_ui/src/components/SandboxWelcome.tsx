import React, { useState } from 'react';
import { Play, Shield, Zap, Clock } from 'lucide-react';

interface SandboxWelcomeProps {
    onStart: (projectId: string) => void;
}

const SandboxWelcome: React.FC<SandboxWelcomeProps> = ({ onStart }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const startSandbox = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch('/sandbox/create', { method: 'POST' });
            const data = await res.json();

            if (res.ok && data.project_id) {
                onStart(data.project_id);
            } else {
                setError(data.error || 'Failed to initialize sandbox. Capacity might be reached.');
            }
        } catch (err) {
            setError('Connection error. Is the CueMap engine running?');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="sandbox-welcome">
            <div className="welcome-card">
                <div className="welcome-header">
                    <div className="logo-icon">
                        <Zap size={32} color="#3b82f6" fill="#3b82f6" />
                    </div>
                    <h1>CueMap <span className="sandbox-badge">SANDBOX</span></h1>
                    <p className="subtitle">High-performance brain-inspired memory store.</p>
                </div>

                <div className="features-grid">
                    <div className="feature-item">
                        <Shield size={20} className="feature-icon" />
                        <div>
                            <h3>Anonymous Access</h3>
                            <p>No account or login required for testing.</p>
                        </div>
                    </div>
                    <div className="feature-item">
                        <Zap size={20} className="feature-icon" />
                        <div>
                            <h3>Zero Latency</h3>
                            <p>Sub-1ms recall on 10,000 memories.</p>
                        </div>
                    </div>
                    <div className="feature-item">
                        <Clock size={20} className="feature-icon" />
                        <div>
                            <h3>Ephemeral</h3>
                            <p>Data exists in RAM and clears after 5 mins of inactivity.</p>
                        </div>
                    </div>
                </div>

                {error && <div className="error-message">{error}</div>}

                <button
                    className="start-button"
                    onClick={startSandbox}
                    disabled={loading}
                >
                    {loading ? (
                        <span className="loader"></span>
                    ) : (
                        <>
                            <Play size={20} fill="currentColor" />
                            <span>Launch Anonymous Sandbox</span>
                        </>
                    )}
                </button>

                <p className="footer-note">
                    By launching, you agree that data is public and will be deleted automatically.
                </p>
            </div>
        </div>
    );
};

export default SandboxWelcome;
