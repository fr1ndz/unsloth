// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Analytics01Icon,
  CheckmarkCircle01Icon,
  Delete01Icon,
  InformationCircleIcon,
  Loading02Icon,
  Scissor01Icon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { authFetch } from "@/features/auth";
import { useToast } from "@/hooks/use-toast";

interface LayerScore {
  name: string;
  module_type: string;
  num_params: number;
  importance: number;
  weight_magnitude: number;
  prune_candidate: boolean;
  cumulative_ratio: number;
}

interface PruningAnalysis {
  total_params: number;
  target_ratio: number;
  actual_ratio: number;
  actual_pruned_params: number;
  estimated_speedup: number;
  layers_to_prune: string[];
  layer_scores: LayerScore[];
  elapsed_seconds: number;
}

interface PruningPanelProps {
  /** Whether a model is currently loaded in the export backend. */
  modelLoaded: boolean;
  /** Called after successful pruning with the output path. */
  onPruneComplete?: (outputPath: string) => void;
}

export function PruningPanel({ modelLoaded, onPruneComplete }: PruningPanelProps) {
  const { toast } = useToast();
  const [ratio, setRatio] = useState(30); // percentage 0-100
  const [method, setMethod] = useState<"magnitude" | "wanda">("magnitude");
  const [analyzing, setAnalyzing] = useState(false);
  const [applying, setApplying] = useState(false);
  const [analysis, setAnalysis] = useState<PruningAnalysis | null>(null);
  const [saveDir, setSaveDir] = useState("");

  // Debounced analysis on ratio change
  useEffect(() => {
    if (!modelLoaded) return;
    const timer = setTimeout(() => {
      void analyzePruning();
    }, 400);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ratio, method, modelLoaded]);

  const analyzePruning = useCallback(async () => {
    if (!modelLoaded) return;
    setAnalyzing(true);
    try {
      const resp = await authFetch("/api/export/pruning/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ratio: ratio / 100,
          method,
          granularity: "layer",
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Analysis failed" }));
        throw new Error(err.detail || "Analysis failed");
      }
      const data = (await resp.json()) as PruningAnalysis;
      setAnalysis(data);
    } catch (e) {
      toast({ title: "Pruning analysis failed", description: String(e), variant: "destructive" });
    } finally {
      setAnalyzing(false);
    }
  }, [modelLoaded, ratio, method, toast]);

  const applyPruning = useCallback(async () => {
    if (!analysis || !saveDir.trim()) {
      toast({ title: "Please specify a save directory", variant: "destructive" });
      return;
    }
    setApplying(true);
    try {
      const resp = await authFetch("/api/export/pruning/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          save_directory: saveDir.trim(),
          ratio: ratio / 100,
          method,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Pruning failed" }));
        throw new Error(err.detail || "Pruning failed");
      }
      const data = await resp.json();
      toast({ title: "Pruning complete", description: data.message });
      onPruneComplete?.(data.details?.output_path ?? saveDir);
    } catch (e) {
      toast({ title: "Pruning failed", description: String(e), variant: "destructive" });
    } finally {
      setApplying(false);
    }
  }, [analysis, saveDir, ratio, method, toast, onPruneComplete]);

  const formatParams = (n: number) => {
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
    return String(n);
  };

  return (
    <div className="flex flex-col gap-4 rounded-xl border border-border bg-card p-5">
      {/* Header */}
      <div className="flex items-center gap-2">
        <HugeiconsIcon icon={Scissor01Icon} className="size-5 text-primary" />
        <span className="text-sm font-semibold">Structured Pruning</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <button type="button" className="text-foreground/50 hover:text-foreground">
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            Remove least important layers/neurons based on weight magnitude scoring.
            Training-free — analyzes the loaded model and produces a smaller version.
          </TooltipContent>
        </Tooltip>
      </div>

      {!modelLoaded && (
        <div className="rounded-lg border border-dashed border-muted-foreground/30 p-4 text-center text-xs text-muted-foreground">
          Load a checkpoint first to enable pruning analysis.
        </div>
      )}

      {modelLoaded && (
        <>
          {/* Ratio Slider */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-medium text-muted-foreground">
                Pruning Ratio
              </Label>
              <span className="text-sm font-bold tabular-nums text-primary">
                {ratio}%
              </span>
            </div>
            <Slider
              value={[ratio]}
              onValueChange={([v]) => setRatio(v)}
              min={0}
              max={90}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>0% (keep all)</span>
              <span>90% (aggressive)</span>
            </div>
          </div>

          {/* Method selector */}
          <div className="flex items-center gap-3">
            <Label className="text-xs font-medium text-muted-foreground">Method</Label>
            <div className="flex gap-1 rounded-lg border border-border p-0.5">
              {(["magnitude", "wanda"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMethod(m)}
                  className={cn(
                    "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                    method === m
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {m === "magnitude" ? "Magnitude" : "Wanda"}
                </button>
              ))}
            </div>
          </div>

          {/* Analysis Results */}
          {analyzing && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <HugeiconsIcon icon={Loading02Icon} className="size-3 animate-spin" />
              Analyzing layer importance...
            </div>
          )}

          {analysis && !analyzing && (
            <div className="flex flex-col gap-3 rounded-lg border border-border bg-muted/30 p-3">
              {/* Stats row */}
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <div className="text-[10px] text-muted-foreground">Total Params</div>
                  <div className="text-sm font-bold tabular-nums">{formatParams(analysis.total_params)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-muted-foreground">Will Remove</div>
                  <div className="text-sm font-bold tabular-nums text-destructive">
                    {formatParams(analysis.actual_pruned_params)}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-muted-foreground">Est. Speedup</div>
                  <div className="text-sm font-bold tabular-nums text-primary">
                    {analysis.estimated_speedup.toFixed(1)}×
                  </div>
                </div>
              </div>

              {/* Layer list (scrollable, top 20) */}
              <div className="max-h-48 overflow-y-auto rounded border border-border bg-background">
                <table className="w-full text-[11px]">
                  <thead className="sticky top-0 bg-muted/50 text-left text-muted-foreground">
                    <tr>
                      <th className="px-2 py-1 font-medium">Layer</th>
                      <th className="px-2 py-1 font-medium text-right">Params</th>
                      <th className="px-2 py-1 font-medium text-right">Importance</th>
                      <th className="px-2 py-1 font-medium text-center">Prune?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.layer_scores.slice(0, 30).map((l) => (
                      <tr
                        key={l.name}
                        className={cn(
                          "border-t border-border/50",
                          l.prune_candidate && "bg-destructive/5",
                        )}
                      >
                        <td className="max-w-[180px] truncate px-2 py-1 font-mono text-[10px]" title={l.name}>
                          {l.name.split(".").slice(-2).join(".")}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums">{formatParams(l.num_params)}</td>
                        <td className="px-2 py-1 text-right tabular-nums">{l.importance.toFixed(4)}</td>
                        <td className="px-2 py-1 text-center">
                          {l.prune_candidate ? (
                            <HugeiconsIcon icon={Delete01Icon} className="inline size-3 text-destructive" />
                          ) : (
                            <HugeiconsIcon icon={CheckmarkCircle01Icon} className="inline size-3 text-emerald-500" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Save + Apply */}
              <div className="flex items-end gap-2">
                <div className="flex-1">
                  <Label className="text-[10px] text-muted-foreground">Save Directory</Label>
                  <input
                    type="text"
                    value={saveDir}
                    onChange={(e) => setSaveDir(e.target.value)}
                    placeholder="e.g. ./pruned-model"
                    className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs outline-none focus:ring-1 focus:ring-ring"
                  />
                </div>
                <Button
                  size="sm"
                  onClick={() => void applyPruning()}
                  disabled={applying || !saveDir.trim()}
                  className="shrink-0"
                >
                  {applying ? (
                    <>
                      <HugeiconsIcon icon={Loading02Icon} className="mr-1 size-3 animate-spin" />
                      Pruning...
                    </>
                  ) : (
                    <>
                      <HugeiconsIcon icon={Analytics01Icon} className="mr-1 size-3" />
                      Apply Pruning
                    </>
                  )}
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
