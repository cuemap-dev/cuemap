import React, { useEffect, useState, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

interface GraphData {
    nodes: any[];
    links: any[];
}

interface GraphProps {
    highlightedMemoryMap: Map<string, number>;
}

const GraphVisualizer: React.FC<GraphProps> = ({ highlightedMemoryMap }) => {
    const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
    const fgRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Auto-resize logic
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
        updateSize(); // Initial call

        return () => window.removeEventListener('resize', updateSize);
    }, []);

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const projectId = params.get('project');

        let url = '/graph?limit=500';
        if (projectId) {
            url += `&project=${projectId}`;
        }

        fetch(url)
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    console.error("Graph Error:", data.error);
                    return;
                }
                setData(data);
                // Adjust physics for better spread
                if (fgRef.current) {
                    fgRef.current.d3Force('charge').strength(-40);
                    fgRef.current.d3Force('link').distance(60);
                }
            })
            .catch(err => console.error(err));
    }, []);

    // Custom node rendering with bloom/glow effect
    const drawNodeWithBloom = (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const x = node.x!;
        const y = node.y!;
        const isHighlighted = highlightedMemoryMap.has(node.id);
        const rank = highlightedMemoryMap.get(node.id);
        const isMemory = node.group === 'memory';

        // Node properties
        let baseColor: string;
        let glowColor: string;
        let nodeRadius: number;
        let glowIntensity: number;

        if (isHighlighted) {
            // HIGHLIGHTED MEMORIES (in recall results): Bright green with strong glow + rank
            baseColor = '#22c55e';  // Bright Green
            glowColor = '#4ade80';
            nodeRadius = 10;
            glowIntensity = 25;
        } else if (isMemory) {
            // REGULAR MEMORIES: Bright green (slightly dimmer than highlighted), bigger than cues
            baseColor = '#4ade80'; // Lime Green
            glowColor = '#22c55e';
            nodeRadius = 6;
            glowIntensity = 8;
        } else {
            // CUE NODES: Small blue dots
            baseColor = '#60a5fa'; // Blue
            glowColor = '#3b82f6';
            nodeRadius = 3;
            glowIntensity = 5;
        }

        // Save context state
        ctx.save();

        // Apply bloom/glow effect
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = glowIntensity;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;

        // Draw the glowing node
        ctx.fillStyle = baseColor;
        ctx.beginPath();
        ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI, false);
        ctx.fill();

        // Draw a bright center for highlighted nodes
        if (isHighlighted) {
            ctx.shadowBlur = 5;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.beginPath();
            ctx.arc(x, y, nodeRadius * 0.4, 0, 2 * Math.PI, false);
            ctx.fill();
        }

        // Restore context and disable shadow for text
        ctx.restore();

        // Draw rank number for highlighted nodes
        if (rank !== undefined) {
            const fontSize = Math.max(10, 14 / globalScale);
            ctx.font = `bold ${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Text shadow for visibility
            ctx.fillStyle = '#000000';
            ctx.fillText(rank.toString(), x + 1, y + 1);

            // White text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(rank.toString(), x, y);
        }
    };

    return (
        <div ref={containerRef} style={{ height: '100%', width: '100%', overflow: 'hidden', background: '#0f172a' }}>
            <ForceGraph2D
                ref={fgRef}
                width={dimensions.width}
                height={dimensions.height}
                graphData={data}
                nodeLabel="label"

                // Use custom rendering for ALL nodes (replace default)
                nodeCanvasObjectMode={() => 'replace'}
                nodeCanvasObject={drawNodeWithBloom}

                // Links with subtle glow
                linkColor={() => 'rgba(99, 102, 241, 0.15)'}
                linkWidth={1}

                backgroundColor="#0f172a"

                onEngineStop={() => {
                    if (fgRef.current) {
                        fgRef.current.zoomToFit(400, 50);
                    }
                }}
            />
        </div>
    );
};

export default GraphVisualizer;
