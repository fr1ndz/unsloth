/**
 * Stratum Visualizer — pure Canvas 2D spectral decomposition renderer.
 * No external 3D dependencies required.
 *
 * Each stratum layer is visualized as a ring of singular values (σ).
 * Color encodes coherence health; size encodes spectral energy.
 */
import { useRef, useEffect, useCallback } from 'react';

interface StratumVisualizerProps {
  spectrum: number[][];
  coherenceValues?: number[];
  selectedLayer?: number | null;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else { r = c; b = x; }
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}

export function StratumVisualizer({ spectrum, coherenceValues, selectedLayer }: StratumVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  const draw = useCallback((time: number) => {
    const canvas = canvasRef.current;
    if (!canvas || spectrum.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const layerSpacing = Math.min(h / (spectrum.length + 1), 60);

    // Clear with fade trail
    ctx.fillStyle = 'rgba(10, 10, 15, 0.15)';
    ctx.fillRect(0, 0, w, h);

    for (let li = 0; li < spectrum.length; li++) {
      const sigma = spectrum[li];
      const coherence = coherenceValues?.[li] ?? 0.5;
      const isSelected = selectedLayer === li;
      const yBase = cy + (li - spectrum.length / 2) * layerSpacing;

      // Hue: coherence 0→red(0), 1→green(120)
      const hue = coherence * 120;
      const lightness = isSelected ? 70 : 50;
      const [r, g, b] = hslToRgb(hue, 90, lightness);

      for (let i = 0; i < sigma.length; i++) {
        const angle = (i / sigma.length) * Math.PI * 2 + time * 0.0005;
        const radius = 40 + sigma[i] * 20;
        const x = cx + Math.cos(angle) * radius;
        const y = yBase + Math.sin(time * 0.001 + i * 0.3) * 3;
        const size = 2 + sigma[i] * 4;

        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${isSelected ? 0.95 : 0.7})`;
        ctx.fill();

        // Glow for selected
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(x, y, size * 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},0.15)`;
          ctx.fill();
        }
      }

      // Layer label
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '10px monospace';
      ctx.fillText(`L${li}`, cx + 80, yBase + 3);
    }

    // Background particles
    for (let p = 0; p < 30; p++) {
      const px = ((Math.sin(time * 0.0003 + p * 7.3) + 1) / 2) * w;
      const py = ((Math.cos(time * 0.0002 + p * 4.1) + 1) / 2) * h;
      ctx.beginPath();
      ctx.arc(px, py, 0.8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(100,130,200,0.12)';
      ctx.fill();
    }

    animRef.current = requestAnimationFrame(() => draw(performance.now()));
  }, [spectrum, coherenceValues, selectedLayer]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth * window.devicePixelRatio;
        canvas.height = parent.clientHeight * window.devicePixelRatio;
        canvas.style.width = `${parent.clientWidth}px`;
        canvas.style.height = `${parent.clientHeight}px`;
      }
    };
    resize();
    window.addEventListener('resize', resize);

    animRef.current = requestAnimationFrame(() => draw(performance.now()));

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animRef.current);
    };
  }, [draw]);

  return (
    <div className="w-full h-full min-h-[400px] relative">
      <canvas ref={canvasRef} className="block w-full h-full" />
      {spectrum.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-white/30">
          <div className="text-center">
            <div className="text-4xl mb-4">🔬</div>
            <div className="text-lg font-medium">RSMF Stratum Visualizer</div>
            <div className="text-sm mt-2">Initialize a model to see spectral decomposition</div>
          </div>
        </div>
      )}
    </div>
  );
}
