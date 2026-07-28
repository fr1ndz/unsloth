// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { useHubModelSearch } from "@/features/hub/hooks/use-hub-model-search";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import {
  type LocalModelInfo,
  useTrainingConfigStore,
} from "@/features/training";
import { useDebouncedValue, useHfTokenValidation } from "@/hooks";
import { useHardwareInfo } from "@/hooks/use-hardware-info";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import {
  AlertCircleIcon,
  FolderSearchIcon,
  InformationCircleIcon,
  Key01Icon,
  MoreHorizontalIcon,
  Package01Icon,
  Scissor01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useSearch } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { ModelCheckpoints } from "./api/export-api";
import { ExportRunPanel } from "./components/export-run-panel";
import { PruningPanel } from "./components/pruning-panel";
import { MethodPicker } from "./components/method-picker";
import { QuantPicker } from "./components/quant-picker";
import {
  EXPORT_METHODS,
  type ExportMethod,
  GUIDE_STEPS,
  MERGED_FORMATS,
  type MergedFormatOption,
  QUANT_OPTIONS,
  buildQuantSizeLabels,
  getEstimatedSize,
  mergedFormatPayload,
} from "./constants";
import { useExportSizeEstimate } from "./hooks/use-export-size-estimate";
import {
  getCachedCheckpoints,
  getCachedLocalModels,
  refreshCheckpoints,
  refreshLocalModels,
} from "./export-navigation-cache";
import {
  isExportPanelActive,
  useExportRuntimeStore,
} from "./stores/export-runtime-store";
import { exportTourSteps } from "./tour";

const SEARCH_INPUT_REASONS = new Set([
  "input-change",
  "input-paste",
  "input-clear",
]);

// GGUF LoRA output float types (Q8_0 first / default). Q8_0 falls back to F16 per tensor for dims
// not divisible by the block size (32); no "auto" - the choice is explicit.
const LORA_GGUF_OUTTYPES = ["q8_0", "f16", "bf16", "f32"] as const;

type SourceTab = "local" | "checkpoint" | "hf";
type SourceMode = "checkpoint" | "model";

function safePathSegment(
  value: string | null | undefined,
  fallback = "model",
  maxLength = 250,
): string {
  const safe = (value ?? "")
    .replace(/[^a-zA-Z0-9._-]/g, "-")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, maxLength)
    .replace(/[._-]+$/g, "");
  return safe || fallback;
}

function buildRelativeSaveDirectory(
  exportMethod: ExportMethod | null,
  sourceMode: SourceMode,
  sourceBaseModelName: string,
  selectedModelIdx: string | null,
  checkpoint: string | null,
): string {
  if (exportMethod === "gguf") {
    const rawName =
      sourceMode === "checkpoint"
        ? (checkpoint ?? selectedModelIdx ?? sourceBaseModelName)
        : sourceBaseModelName;
    return `${safePathSegment(rawName)}-GGUF`;
  }
  // Merged / LoRA: a checkpoint keeps the "<run>/<checkpoint>" layout under outputs.
  if (sourceMode === "checkpoint" && selectedModelIdx && checkpoint) {
    return `${selectedModelIdx}/${checkpoint}`;
  }
  // Local / HF source (no checkpoint): name from the model id to avoid "model/null".
  const rawName =
    sourceMode === "checkpoint"
      ? (checkpoint ?? selectedModelIdx ?? sourceBaseModelName)
      : sourceBaseModelName;
  return `${safePathSegment(rawName)}-${exportMethod === "lora" ? "adapter" : "merged"}`;
}

function siblingGgufDirectory(sourcePath: string): string | null {
  const trimmed = sourcePath.trim().replace(/[\\/]+$/, "");
  if (!trimmed) return null;
  const slash = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  // Lowercase `_gguf` matches the backend's intermediate dir (core/export/export.py);
  // `_GGUF` would relocate+delete that sibling.
  if (slash < 0) return `${trimmed}_gguf`;
  const parent =
    slash === 0 || (slash === 2 && /^[A-Za-z]:/.test(trimmed))
      ? trimmed.slice(0, slash + 1)
      : trimmed.slice(0, slash);
  const name = trimmed.slice(slash + 1);
  if (!name) return null;
  const sep =
    parent.endsWith("/") || parent.endsWith("\\")
      ? ""
      : trimmed.includes("\\")
        ? "\\"
        : "/";
  return `${parent}${sep}${name}_gguf`;
}

export function ExportPage() {
  const { hfToken, setHfToken } = useTrainingConfigStore(
    useShallow((s) => ({
      hfToken: s.hfToken,
      setHfToken: s.setHfToken,
    })),
  );

  // ---- API-driven checkpoint state ----
  const [models, setModels] = useState<ModelCheckpoints[]>(
    () => getCachedCheckpoints() ?? [],
  );
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(
    getCachedCheckpoints() === null,
  );
  const [checkpointError, setCheckpointError] = useState<string | null>(null);

  const [selectedModelIdx, setSelectedModelIdx] = useState<string | null>(null);
  const [checkpoint, setCheckpoint] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<SourceMode>("checkpoint");
  const [modelSource, setModelSource] = useState<"hf" | "local">("hf");
  const [modelInput, setModelInput] = useState("");
  const [selectedSourceModel, setSelectedSourceModel] = useState<string | null>(
    null,
  );
  const [localModelInput, setLocalModelInput] = useState("");
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>(
    () => getCachedLocalModels() ?? [],
  );
  const [isLoadingLocalModels, setIsLoadingLocalModels] = useState(
    getCachedLocalModels() === null,
  );
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);
  const debouncedModelQuery = useDebouncedValue(modelInput);
  const debouncedHfToken = useDebouncedValue(hfToken, 500);

  // Seed the method + quants from a live run so that navigating away and back
  // (which remounts this page) keeps the method card selected and the run panel
  // showing its logs/progress. The run itself lives in the global store; only
  // this form state is local and would otherwise reset to null on remount.
  const [exportMethod, setExportMethod] = useState<ExportMethod | null>(() => {
    const s = useExportRuntimeStore.getState();
    return isExportPanelActive(s) && s.summary ? s.summary.method : null;
  });
  const [quantLevels, setQuantLevels] = useState<string[]>(() => {
    const s = useExportRuntimeStore.getState();
    return isExportPanelActive(s) && s.summary?.method === "gguf"
      ? s.summary.quantLevels
      : [];
  });
  // GGUF importance matrix (required for the IQ quants) and merged-export precision.
  const [useImatrix, setUseImatrix] = useState(false);
  // Merged precision: one or more MERGED_FORMATS values, exported in one run. Seed from a live run
  // so navigating away and back (which remounts this page) keeps the selection, like exportMethod.
  const [selectedFormats, setSelectedFormats] = useState<string[]>(() => {
    const s = useExportRuntimeStore.getState();
    return isExportPanelActive(s) &&
      s.summary?.method === "merged" &&
      s.summary.mergedFormats.length > 0
      ? s.summary.mergedFormats
      : ["16-bit"];
  });
  // LoRA-only export: optionally also emit a GGUF LoRA adapter, and its output float type.
  const [loraAsGguf, setLoraAsGguf] = useState(false);
  const [loraGgufOuttype, setLoraGgufOuttype] = useState<string>("q8_0");
  // GGUF method: export the full model as GGUF quants, or (for an adapter checkpoint) a GGUF LoRA.
  const [ggufTarget, setGgufTarget] = useState<"model" | "lora">("model");

  const hardware = useHardwareInfo();
  // GGUF LoRA conversion is rejected on the macOS / MLX path, so gate it out on a Mac host.
  const isMacHost = usePlatformStore((s) => s.deviceType) === "mac";
  // Real CUDA (not ROCm); gates the NVIDIA-only compressed-tensors formats.
  const hasNvidia = hardware.cuda != null && hardware.rocm == null;
  // Only gray out on an authoritative unsupported response; while unloaded the backend route guard
  // stays authoritative. The backend supplies the precise reason; the fallback below is a backstop.
  const exportUnsupported =
    hardware.loaded && hardware.exportSupported === false;
  const exportUnsupportedMessage =
    hardware.exportUnsupportedMessage ??
    "Export requires a supported accelerator (NVIDIA, AMD, or Intel GPU, or Apple Silicon) with PyTorch or MLX installed.";
  const availableFormats = useMemo<MergedFormatOption[]>(
    () =>
      MERGED_FORMATS.filter((f) => {
        // compressed-tensors (llm-compressor) is the NVIDIA path; shown only on an NVIDIA GPU.
        if (f.backend === "compressed") return hasNvidia;
        // Portable torchao is the fallback for hosts without the NVIDIA compressed path, i.e. a
        // CPU / non-NVIDIA box. Hidden on NVIDIA (use compressed-tensors) and on macOS/MLX (the
        // backend rejects quantized export there).
        if (f.backend === "torchao") return !hasNvidia && !isMacHost;
        // Plain 16-bit is available everywhere.
        return true;
      }),
    [hasNvidia, isMacHost],
  );
  const toggleFormat = useCallback((value: string) => {
    setSelectedFormats((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value],
    );
  }, []);
  // availableFormats already drops NVIDIA-only formats on other hardware, so no pruning needed.
  // IQ quants are imatrix-only: force imatrix on when one is selected, else llama.cpp rejects it.
  const requiresImatrix = quantLevels.some(
    (q) => QUANT_OPTIONS.find((o) => o.value === q)?.imatrix,
  );
  const effectiveImatrix = useImatrix || requiresImatrix;

  // Whether the inline export panel is expanded. The panel also shows itself
  // whenever a run is active/terminal (see `panelActive`), so it survives
  // navigation even though this local flag resets on remount.
  const [panelOpen, setPanelOpen] = useState(false);

  const [destination, setDestination] = useState<"local" | "hub">("local");
  const [customSaveDirectory, setCustomSaveDirectory] = useState<string | null>(
    null,
  );
  const [hfUsername, setHfUsername] = useState("");
  const [modelName, setModelName] = useState("");
  const [privateRepo, setPrivateRepo] = useState(false);

  // Export run state lives in the global runtime store so it keeps running and
  // streaming in the background, in parallel with training and inference.
  const runExport = useExportRuntimeStore((s) => s.runExport);
  const resetExportRun = useExportRuntimeStore((s) => s.reset);
  const isExporting = useExportRuntimeStore((s) => s.isExporting);
  const panelActive = useExportRuntimeStore(isExportPanelActive);

  const hfComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const localComboboxAnchorRef = useRef<HTMLDivElement>(null);
  const selectingHfModelRef = useRef(false);
  const hfModelInputRef = useRef("");
  const localModelInputRef = useRef("");

  const tour = useGuidedTourController({
    id: "export",
    steps: exportTourSteps,
  });

  // ---- Fetch checkpoints on mount ----
  useEffect(() => {
    let cancelled = false;
    const hadCache = getCachedCheckpoints() !== null;
    refreshCheckpoints()
      .then((models: ModelCheckpoints[]) => {
        if (!cancelled) {
          setModels(models);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled && !hadCache) {
          setCheckpointError(
            err instanceof Error ? err.message : "Failed to load checkpoints",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingCheckpoints(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Apply the ?run= deep link (e.g. from a finished run's "Export to GGUF"
  // button) once its run appears in the checkpoint list: select the run and
  // default to GGUF. The main checkpoint is auto-selected further below, after
  // the model-change effect that clears the checkpoint.
  const { run: preselectRun } = useSearch({ from: "/export" });
  const appliedRunRef = useRef<string | null>(null);
  useEffect(() => {
    if (!preselectRun) {
      // Deep link cleared (e.g. navigated to /export via the sidebar): stop
      // treating the previously preselected run specially.
      appliedRunRef.current = null;
      return;
    }
    if (models.length === 0) return;
    if (appliedRunRef.current === preselectRun) return;
    const match = models.find((m) => m.name === preselectRun);
    if (!match) return;
    appliedRunRef.current = preselectRun;
    setSourceMode("checkpoint");
    setSelectedModelIdx(match.name);
    setExportMethod("gguf");
  }, [preselectRun, models]);

  // ---- Fetch local models for direct export ----
  useEffect(() => {
    let cancelled = false;
    const hadCache = getCachedLocalModels() !== null;
    void refreshLocalModels()
      .then((models: LocalModelInfo[]) => {
        if (cancelled) return;
        setLocalModels(models);
      })
      .catch((error: unknown) => {
        if (cancelled || hadCache) return;
        setLocalModelsError(
          error instanceof Error
            ? error.message
            : "Failed to load local models",
        );
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoadingLocalModels(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- Derived state ----
  const selectedModelData = useMemo(
    () =>
      selectedModelIdx != null
        ? (models.find((m) => m.name === selectedModelIdx) ?? null)
        : null,
    [models, selectedModelIdx],
  );

  const checkpointsForModel = useMemo(
    () => selectedModelData?.checkpoints ?? [],
    [selectedModelData],
  );

  // Derive training info from selected model's API metadata
  const baseModelName = selectedModelData?.base_model ?? "—";
  const isAdapter = !!selectedModelData?.peft_type;
  const isQuantized = !!selectedModelData?.is_quantized;
  // isAdapter / isQuantized come from the checkpoint's metadata and are stale in "model" source
  // mode (a direct base export), so treat both as false outside checkpoint mode to avoid wrongly
  // gating the methods.
  const effectiveIsAdapter = sourceMode === "checkpoint" && isAdapter;
  const effectiveIsQuantized = sourceMode === "checkpoint" && isQuantized;
  const loraRank = selectedModelData?.lora_rank ?? null;
  const trainingMethodLabel = selectedModelData?.peft_type
    ? "LoRA / QLoRA"
    : "Full Fine-tune";
  const sourceBaseModelName =
    sourceMode === "model" ? (selectedSourceModel ?? "—") : baseModelName;

  // For a full fine-tune checkpoint the weights live in the checkpoint dir
  // itself (its base_model may be a local/custom path that can't be sized), so
  // size that dir; for LoRA adapters the export merges into the base model.
  const sizeTargetModel = useMemo(() => {
    if (sourceMode === "checkpoint" && !isAdapter) {
      const cp = checkpointsForModel.find((c) => c.display_name === checkpoint);
      if (cp?.path) {
        return cp.path;
      }
    }
    return sourceBaseModelName;
  }, [
    sourceMode,
    isAdapter,
    checkpointsForModel,
    checkpoint,
    sourceBaseModelName,
  ]);

  // Real (MoE-aware) fp16 size, used to scale the GGUF quant estimates.
  const { fp16Bytes } = useExportSizeEstimate(
    sizeTargetModel,
    debouncedHfToken,
  );
  const quantSizeLabels = useMemo(
    () => buildQuantSizeLabels(fp16Bytes),
    [fp16Bytes],
  );

  const {
    results: hfResults,
    isLoading: isLoadingHfModels,
    error: hfSearchError,
  } = useHubModelSearch(debouncedModelQuery, {
    accessToken: debouncedHfToken || undefined,
    excludeGguf: true,
    // Curated unsloth listing by default, but a typed query searches the whole
    // Hub (unsloth floated first) so non-unsloth base models stay selectable.
    ownerScope: debouncedModelQuery.trim() ? "all" : "unsloth",
  });
  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const hfResultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (
      selectedSourceModel &&
      modelSource === "hf" &&
      !ids.includes(selectedSourceModel)
    ) {
      ids.push(selectedSourceModel);
    }
    return ids;
  }, [hfResults, modelSource, selectedSourceModel]);

  const exportableLocalModels = useMemo(
    () =>
      localModels.filter((m) => {
        if (m.path.endsWith(".gguf")) return false;
        if (m.id.toLowerCase().includes("-gguf")) return false;
        return true;
      }),
    [localModels],
  );

  const localMetaById = useMemo(() => {
    const map = new Map<string, LocalModelInfo>();
    for (const model of exportableLocalModels) map.set(model.id, model);
    return map;
  }, [exportableLocalModels]);

  const localResultIds = useMemo(() => {
    const ids = exportableLocalModels.map((model) => model.id);
    const manual = localModelInput.trim();
    if (manual && !ids.includes(manual)) {
      ids.unshift(manual);
    }
    return ids;
  }, [exportableLocalModels, localModelInput]);

  const localFilteredIds = useMemo(() => {
    const q = localModelInput.trim().toLowerCase();
    if (!q) return localResultIds;
    return localResultIds.filter((id) => {
      const meta = localMetaById.get(id);
      if (id.toLowerCase().includes(q)) return true;
      if (meta?.display_name.toLowerCase().includes(q)) return true;
      if (meta?.path.toLowerCase().includes(q)) return true;
      return false;
    });
  }, [localMetaById, localModelInput, localResultIds]);

  const exportGuideSteps = useMemo(
    () =>
      sourceMode === "model"
        ? [
            "Select a Hugging Face or local model to export from",
            "GGUF is used for non-finetuned model exports",
            "Pick one or more GGUF quantization levels",
            "Click Export and choose your destination",
            "Test your model and compare outputs in Chat",
          ]
        : GUIDE_STEPS,
    [sourceMode],
  );
  const sourceTab: SourceTab =
    sourceMode === "checkpoint" ? "checkpoint" : modelSource;

  // Reset checkpoint when the selected model changes
  useEffect(() => {
    setCheckpoint(null);
  }, [selectedModelIdx]);

  // Default to the newest checkpoint when none is chosen (checkpoints are sorted newest-first).
  // Declared after the reset effect above so it runs last and isn't clobbered back to null. Covers
  // both a ?run= deep link and a plain finetune opened without an explicit checkpoint pick.
  useEffect(() => {
    if (sourceMode !== "checkpoint") return;
    if (checkpoint != null || checkpointsForModel.length === 0) return;
    setCheckpoint(checkpointsForMo

... [OUTPUT TRUNCATED - 24,230 chars omitted out of 74,157 total] ...

            icon={Search01Icon}
                                      className="size-4"
                                    />
                                  </InputGroupAddon>
                                </ComboboxInput>
                                <ComboboxContent anchor={hfComboboxAnchorRef}>
                                  {isLoadingHfModels ? (
                                    <div className="flex items-center justify-center py-4 gap-2 text-xs text-muted-foreground">
                                      <Spinner className="size-4" /> Searching…
                                    </div>
                                  ) : (
                                    <ComboboxEmpty>
                                      No models found
                                    </ComboboxEmpty>
                                  )}
                                  <ComboboxList className="p-1 !max-h-none !overflow-visible">
                                    {(id: string) => (
                                      <ComboboxItem
                                        key={id}
                                        value={id}
                                        className="gap-2"
                                      >
                                        <span className="block min-w-0 flex-1 truncate">
                                          {id}
                                        </span>
                                      </ComboboxItem>
                                    )}
                                  </ComboboxList>
                                </ComboboxContent>
                              </Combobox>
                            </div>
                            {(tokenValidationError ?? hfSearchError) && (
                              <p className="text-xs text-destructive">
                                {tokenValidationError ?? hfSearchError}
                              </p>
                            )}
                          </div>
                          {/* No persistent "trust remote code" toggle: custom code is
                              consented per model via the load-time review dialog. */}
                          <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-medium text-muted-foreground">
                              Hugging Face Token (Optional)
                            </label>
                            <InputGroup>
                              <InputGroupAddon>
                                <HugeiconsIcon
                                  icon={Key01Icon}
                                  className="size-4"
                                />
                              </InputGroupAddon>
                              <InputGroupInput
                                type="password"
                                autoComplete="new-password"
                                name="hf-token-export-source"
                                placeholder="hf_..."
                                value={hfToken}
                                onChange={(e) => setHfToken(e.target.value)}
                              />
                            </InputGroup>
                            {isCheckingToken && (
                              <p className="text-xs text-muted-foreground">
                                Checking token…
                              </p>
                            )}
                          </div>
                        </>
                      ) : (
                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-medium text-muted-foreground">
                            Local Model Path
                          </label>
                          <div ref={localComboboxAnchorRef}>
                            <Combobox
                              items={localResultIds}
                              filteredItems={localFilteredIds}
                              filter={null}
                              value={localModelInput || null}
                              onValueChange={(id) => {
                                const next = id ?? "";
                                localModelInputRef.current = next;
                                setLocalModelInput(next);
                                setSelectedSourceModel(next || null);
                              }}
                              onInputValueChange={handleLocalSourceInputChange}
                              itemToStringValue={(id) => id}
                              autoHighlight={true}
                            >
                              <ComboboxInput
                                placeholder={
                                  isLoadingLocalModels
                                    ? "Scanning local and cached models..."
                                    : "./models/my-model"
                                }
                                className="w-full"
                                onBlur={() =>
                                  applyLocalSourceModel(
                                    localModelInputRef.current,
                                  )
                                }
                                onKeyDown={(event) => {
                                  if (event.key !== "Enter") return;
                                  event.preventDefault();
                                  applyLocalSourceModel(
                                    localModelInputRef.current,
                                  );
                                }}
                              >
                                <InputGroupAddon>
                                  <HugeiconsIcon
                                    icon={FolderSearchIcon}
                                    className="size-4"
                                  />
                                </InputGroupAddon>
                              </ComboboxInput>
                              <ComboboxContent anchor={localComboboxAnchorRef}>
                                {isLoadingLocalModels ? (
                                  <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                                    <Spinner className="size-4" /> Scanning...
                                  </div>
                                ) : localModelsError ? (
                                  <div className="px-3 py-2 text-xs text-red-500">
                                    {localModelsError}
                                  </div>
                                ) : (
                                  <ComboboxEmpty>
                                    No local models found
                                  </ComboboxEmpty>
                                )}
                                <ComboboxList className="p-1 !max-h-none !overflow-visible">
                                  {(id: string) => {
                                    const model = localMetaById.get(id);
                                    const source =
                                      model?.source === "hf_cache"
                                        ? "HF cache"
                                        : model?.source === "custom"
                                          ? "Custom Folders"
                                          : "Local dir";
                                    return (
                                      <ComboboxItem
                                        key={id}
                                        value={id}
                                        className="gap-2"
                                      >
                                        <span className="block min-w-0 flex-1 truncate">
                                          {model?.display_name ?? id}
                                        </span>
                                        <span className="ml-auto shrink-0 text-ui-10 text-muted-foreground">
                                          {source}
                                        </span>
                                      </ComboboxItem>
                                    );
                                  }}
                                </ComboboxList>
                              </ComboboxContent>
                            </Combobox>
                          </div>
                          {isLoadingLocalModels ? (
                            <p className="text-ui-10 text-muted-foreground">
                              Scanning local models...
                            </p>
                          ) : localModelsError ? (
                            <p className="text-ui-10 text-red-500">
                              {localModelsError}
                            </p>
                          ) : (
                            <p className="text-ui-10 text-muted-foreground">
                              {exportableLocalModels.length > 0
                                ? `${exportableLocalModels.length} local/cached models found`
                                : "No local models found. Enter path manually."}
                            </p>
                          )}
                        </div>
                      )}

                      <div className="rounded-xl bg-foreground/[0.04] p-3">
                        <p className="text-ui-11 text-muted-foreground">
                          Direct model exports currently support GGUF only.
                        </p>
                      </div>
                    </div>
                  )}

                  {sourceMode === "checkpoint" && (
                    <div className="rounded-xl bg-foreground/[0.04] p-3 flex flex-col gap-2">
                      <span className="text-ui-11 font-medium text-muted-foreground uppercase tracking-wider">
                        Training Info
                      </span>
                      <div className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">
                            Base Model
                          </span>
                          <span className="font-medium">{baseModelName}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Method</span>
                          <span className="font-medium">
                            {trainingMethodLabel}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">
                            Checkpoints
                          </span>
                          <span className="font-medium">
                            {checkpointsForModel.length}
                          </span>
                        </div>
                        {isAdapter && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">
                              LoRA Rank
                            </span>
                            <span className="font-medium">{loraRank}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex flex-col gap-2.5">
                  <span className="text-xs font-medium text-muted-foreground">
                    Quick Guide
                  </span>
                  <ol className="flex flex-col gap-3">
                    {exportGuideSteps.map((step, i) => (
                      <li
                        key={step}
                        className="flex items-start gap-2 text-xs text-muted-foreground"
                      >
                        <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-foreground/10 text-ui-10 font-semibold">
                          {i + 1}
                        </span>
                        {step}
                      </li>
                    ))}
                  </ol>
                </div>
              </div>

              {exportUnsupported && (
                <Alert variant="destructive">
                  <HugeiconsIcon icon={AlertCircleIcon} className="size-4" />
                  <AlertTitle>Export unavailable</AlertTitle>
                  <AlertDescription>
                    {exportUnsupportedMessage}
                  </AlertDescription>
                </Alert>
              )}

              <MethodPicker
                value={exportMethod}
                onChange={handleMethodChange}
                disabledMethods={
                  exportUnsupported
                    ? ["merged", "lora", "gguf"]
                    : !effectiveIsAdapter && effectiveIsQuantized
                      ? ["merged", "lora", "gguf"]
                      : effectiveIsAdapter
                        ? []
                        : ["lora"]
                }
                disabledReason={
                  exportUnsupported
                    ? exportUnsupportedMessage
                    : !effectiveIsAdapter && effectiveIsQuantized
                      ? "Pre-quantized (BNB 4-bit) models cannot be exported without LoRA adapters"
                      : effectiveIsAdapter
                        ? undefined
                        : "LoRA-only export needs a LoRA adapter checkpoint"
                }
              />

              {exportMethod === "merged" && !exportUnsupported && (
                <div className="space-y-3">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium">Precision</div>
                      <span className="text-ui-11 text-muted-foreground/70">
                        — select one or more
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {availableFormats
                        .filter((f) => f.common)
                        .map((f) => {
                          const active = selectedFormats.includes(f.value);
                          return (
                            <Button
                              key={f.value}
                              type="button"
                              variant={active ? "default" : "outline"}
                              size="sm"
                              onClick={() => toggleFormat(f.value)}
                              title={f.hint}
                            >
                              {f.label}
                              {f.needsCalibration ? " *" : ""}
                            </Button>
                          );
                        })}

                      {availableFormats.some((f) => !f.common) && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild={true}>
                            <Button type="button" variant="outline" size="sm">
                              More formats
                              {selectedFormats.some((v) =>
                                availableFormats.find(
                                  (f) => f.value === v && !f.common,
                                ),
                              )
                                ? " ✓"
                                : "…"}
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="start" className="w-64">
                            <DropdownMenuLabel>
                              Additional formats
                            </DropdownMenuLabel>
                            <DropdownMenuSeparator />
                            {availableFormats
                              .filter((f) => !f.common)
                              .map((f) => (
                                <DropdownMenuCheckboxItem
                                  key={f.value}
                                  checked={selectedFormats.includes(f.value)}
                                  onCheckedChange={() => toggleFormat(f.value)}
                                  onSelect={(e) => e.preventDefault()}
                                >
                                  <span className="flex flex-col">
                                    <span>
                                      {f.label}
                                      {f.needsCalibration ? " *" : ""}
                                    </span>
                                    <span className="text-ui-10 text-muted-foreground">
                                      {f.hint}
                                    </span>
                                  </span>
                                </DropdownMenuCheckboxItem>
                              ))}
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                    </div>

                    {selectedFormats.length > 0 && (
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span className="text-ui-11 text-muted-foreground">
                          {selectedFormats.length} selected:{" "}
                          {selectedFormats
                            .map(
                              (v) =>
                                MERGED_FORMATS.find((f) => f.value === v)
                                  ?.label ?? v,
                            )
                            .join(", ")}
                        </span>
                        {selectedFormats.length > 1 && (
                          <button
                            type="button"
                            onClick={() => setSelectedFormats(["16-bit"])}
                            className="text-ui-11 text-muted-foreground/70 hover:text-foreground transition-colors"
                          >
                            Reset to 16-bit
                          </button>
                        )}
                      </div>
                    )}

                    {hubMultiFormat && (
                      <div className="text-ui-11 text-amber-600 dark:text-amber-500">
                        Hub export supports one format at a time (each writes to
                        the repository root). Select a single format, or export
                        locally to produce several at once.
                      </div>
                    )}

                    {selectedFormats.some(
                      (v) =>
                        MERGED_FORMATS.find((f) => f.value === v)
                          ?.needsCalibration,
                    ) && (
                      <div className="text-ui-11 text-muted-foreground">
                        * calibrates on data (uses a small calibration set).
                      </div>
                    )}

                    {!hasNvidia && (
                      <div className="text-ui-11 text-muted-foreground">
                        No NVIDIA GPU detected: compressed-tensors formats are
                        hidden. 16-bit and portable FP8/INT8 (torchao) still
                        work here and load in vLLM.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {exportMethod === "lora" &&
                effectiveIsAdapter &&
                !exportUnsupported && (
                  <div className="space-y-3">
                    <div className="space-y-2">
                      <div className="text-sm font-medium">Adapter format</div>
                      <div className="flex flex-wrap gap-2">
                        <Button
                          type="button"
                          variant={loraAsGguf ? "outline" : "default"}
                          size="sm"
                          onClick={() => setLoraAsGguf(false)}
                          title="Standard PEFT adapter (adapter_model.safetensors)."
                        >
                          Adapter (safetensors)
                        </Button>
                        <Button
                          type="button"
                          variant={loraAsGguf ? "default" : "outline"}
                          size="sm"
                          disabled={isMacHost}
                          onClick={() => setLoraAsGguf(true)}
                          title={
                            isMacHost
                              ? "GGUF LoRA export is not available on macOS/MLX. Use the safetensors adapter."
                              : "llama.cpp GGUF LoRA, loadable with `llama-cli --lora`."
                          }
                        >
                          GGUF adapter
                        </Button>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {isMacHost
                          ? "GGUF LoRA is not available on macOS/MLX; exporting the safetensors adapter."
                          : loraAsGguf
                            ? "Converts the adapter to a GGUF LoRA (llama.cpp `--lora`). The base model stays separate."
                            : "Standard PEFT adapter files. Pair with the base model at inference."}
                      </div>
                    </div>

                    {loraAsGguf && (
                      <div className="space-y-1.5">
                        <div className="text-sm font-medium">Output type</div>
                        <Select
                          value={loraGgufOuttype}
                          onValueChange={(v) => setLoraGgufOuttype(v)}
                        >
                          <SelectTrigger className="w-full sm:w-56">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {LORA_GGUF_OUTTYPES.map((t) => (
                              <SelectItem key={t} value={t}>
                                {t.toUpperCase()}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                  </div>
                )}

              {exportMethod === "gguf" && !exportUnsupported && (
                <div className="space-y-3">
                  {effectiveIsAdapter && !isMacHost && (
                    <div className="space-y-2">
                      <div className="text-sm font-medium">Export target</div>
                      <div className="flex flex-wrap gap-2">
                        <Button
                          type="button"
                          variant={
                            ggufTarget === "model" ? "default" : "outline"
                          }
                          size="sm"
                          onClick={() => setGgufTarget("model")}
                          title="Merge the adapter into the base model, then quantize the full model to GGUF."
                        >
                          Full model
                        </Button>
                        <Button
                          type="button"
                          variant={
                            ggufTarget === "lora" ? "default" : "outline"
                          }
                          size="sm"
                          onClick={() => setGgufTarget("lora")}
                          title="Export just the adapter as a GGUF LoRA (llama.cpp `--lora`); the base model stays separate."
                        >
                          LoRA adapter
                        </Button>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {ggufTarget === "lora"
                          ? "Converts the adapter to a GGUF LoRA (llama.cpp `--lora`). The base model stays separate."
                          : "Merges the adapter into the base model, then quantizes the full model to GGUF."}
                      </div>
                    </div>
                  )}

                  {ggufAsLora ? (
                    <div className="space-y-1.5">
                      <div className="text-sm font-medium">Output type</div>
                      <Select
                        value={loraGgufOuttype}
                        onValueChange={(v) => setLoraGgufOuttype(v)}
                      >
                        <SelectTrigger className="w-full sm:w-56">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {LORA_GGUF_OUTTYPES.map((t) => (
                            <SelectItem key={t} value={t}>
                              {t.toUpperCase()}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  ) : (
                    <>
                      <QuantPicker
                        value={quantLevels}
                        onChange={setQuantLevels}
                        sizes={quantSizeLabels}
                      />
                      <div className="flex items-center justify-between gap-3 rounded-lg border p-3">
                        <div className="space-y-0.5">
                          <div className="text-sm font-medium">
                            Importance matrix (imatrix)
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {requiresImatrix
                              ? "Required for the selected IQ low-bit quant. Auto-downloads the upstream Unsloth imatrix for the base model."
                              : "Improves quant quality and unlocks the IQ low-bit quants. Auto-downloads the upstream Unsloth imatrix for the base model."}
                          </div>
                        </div>
                        <Switch
                          checked={effectiveImatrix}
                          onCheckedChange={setUseImatrix}
                          disabled={requiresImatrix}
                        />
                      </div>
                    </>
                  )}
                </div>
              )}
              {estimatedSize && (
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                  <span>Est. size: {estimatedSize}</span>
                </div>
              )}

              <Separator />
              {showPanel && (
                <ExportRunPanel
                  exportMethod={exportMethod}
                  quantLevels={quantLevels}
                  checkpoint={selectedExportSource}
                  baseModelName={sourceBaseModelName}
                  isAdapter={sourceMode === "checkpoint" && isAdapter}
                  destination={destination}
                  onDestinationChange={setDestination}
                  saveDirectory={saveDirectory}
                  defaultSaveDirectory={defaultSaveDirectory}
                  saveDirectoryOverridden={!!customSaveDirectory}
                  onSaveDirectoryChange={setCustomSaveDirectory}
                  hfUsername={hfUsername}
                  onHfUsernameChange={setHfUsername}
                  modelName={modelName}
                  onModelNameChange={setModelName}
                  hfToken={hfToken}
                  onHfTokenChange={setHfToken}
                  privateRepo={privateRepo}
                  onPrivateRepoChange={setPrivateRepo}
                  onStart={handleStart}
                  onClose={handleClosePanel}
                />
              )}
              {showPanel && (
                <div
                  ref={panelEndRef}
                  aria-hidden="true"
                  className="h-px w-full"
                />
              )}
              {showPanel && !panelEndVisible && (
                <button
                  type="button"
                  onClick={() =>
                    panelEndRef.current?.scrollIntoView({
                      behavior: "smooth",
                      block: "end",
                    })
                  }
                  aria-label="Scroll to export output"
                  className="fixed bottom-6 right-6 z-30 flex size-10 items-center justify-center rounded-full border border-border/60 bg-background/90 text-foreground shadow-md backdrop-blur transition-colors hover:bg-muted"
                >
                  <HugeiconsIcon
                    icon={ChevronDownStandardIcon}
                    className="size-5"
                  />
                </button>
              )}
              {!showPanel && (
                <div className="flex items-center justify-end">
                  <Button
                    data-tour="export-cta"
                    disabled={!canExport}
                    onClick={handleOpenPanel}
                  >
                    Export Model
                  </Button>
                </div>
              )}
            </>
          )}
        </SectionCard>
      </main>
    </div>
  );
}