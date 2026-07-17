/**
 * RSMF Page — main route component combining visualizer + control panel.
 */
import { StratumVisualizer } from './components/StratumVisualizer';
import { RsmfControlPanel } from './components/RsmfControlPanel';
import { useRsmfStore } from './stores/rsmf-store';

export default function RsmfPage() {
  const { spectrumSnapshot, coherenceHistory } = useRsmfStore();

  return (
    <div className="flex h-full w-full bg-[#0a0a0f] text-white overflow-hidden">
      {/* Left: Visualizer */}
      <div className="flex-1 relative">
        <StratumVisualizer
          spectrum={spectrumSnapshot}
          coherenceValues={
            coherenceHistory.length > 0
              ? Array(spectrumSnapshot.length).fill(
                  coherenceHistory[coherenceHistory.length - 1],
                )
              : undefined
          }
        />

        {/* Overlay title */}
        <div className="absolute top-4 left-4 pointer-events-none">
          <h1 className="text-xl font-bold tracking-tight">
            Resonant Stratified Manifold Flow
          </h1>
          <p className="text-xs text-white/40 mt-1">
            Full-parameter training on constrained VRAM • Symplectic phase flow • MoE support
          </p>
        </div>
      </div>

      {/* Right: Control Panel */}
      <div className="w-80 border-l border-white/10 overflow-y-auto bg-black/20">
        <RsmfControlPanel />
      </div>
    </div>
  );
}
