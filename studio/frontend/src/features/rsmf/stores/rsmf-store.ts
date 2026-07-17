/**
 * RSMF Zustand store — manages training state, spectrum data, and UI state.
 */
import { create } from 'zustand';
import type {
  RsmfModelConfig,
  RsmfTrainConfig,
  RsmfInitResult,
  RsmfTrainStepResult,
  RsmfBudgetResult,
  RsmfCoherenceReport,
} from '../api/rsmf';
import * as api from '../api/rsmf';

interface RsmfState {
  // Model state
  modelId: number | null;
  initResult: RsmfInitResult | null;
  isTraining: boolean;
  currentEpoch: number;
  currentBatch: number;

  // Training history
  lossHistory: number[];
  coherenceHistory: number[];
  spectrumSnapshot: number[][];

  // Budget & coherence
  budgetResult: RsmfBudgetResult | null;
  coherenceReports: RsmfCoherenceReport[];

  // Config
  modelConfig: RsmfModelConfig;
  trainConfig: RsmfTrainConfig;

  // Actions
  setModelConfig: (cfg: Partial<RsmfModelConfig>) => void;
  setTrainConfig: (cfg: Partial<RsmfTrainConfig>) => void;
  initializeModel: () => Promise<void>;
  runTrainStep: (input: number[], target: number[]) => Promise<void>;
  checkCoherence: () => Promise<void>;
  computeBudget: (numDevices: number, vramGb: number) => Promise<void>;
  reset: () => void;
}

const defaultModelConfig: RsmfModelConfig = {
  architecture: 'dense',
  hidden_dim: 64,
  num_layers: 4,
  num_heads: 4,
  intermediate_dim: 256,
  vocab_size: 1000,
};

const defaultTrainConfig: RsmfTrainConfig = {
  top_k: 8,
  lambda_spectral: 0.01,
  mu_coupling: 0.1,
  learning_rate: 0.001,
  coherence_threshold: 0.3,
  max_inner_iters: 10,
  loss_type: 'mse',
};

export const useRsmfStore = create<RsmfState>((set, get) => ({
  modelId: null,
  initResult: null,
  isTraining: false,
  currentEpoch: 0,
  currentBatch: 0,
  lossHistory: [],
  coherenceHistory: [],
  spectrumSnapshot: [],
  budgetResult: null,
  coherenceReports: [],
  modelConfig: defaultModelConfig,
  trainConfig: defaultTrainConfig,

  setModelConfig: (cfg) =>
    set((s) => ({ modelConfig: { ...s.modelConfig, ...cfg } })),

  setTrainConfig: (cfg) =>
    set((s) => ({ trainConfig: { ...s.trainConfig, ...cfg } })),

  initializeModel: async () => {
    const { modelConfig, trainConfig } = get();
    const result = await api.initModel(modelConfig, trainConfig);
    set({
      modelId: result.model_id,
      initResult: result,
      spectrumSnapshot: result.spectrum_snapshot,
      lossHistory: [],
      coherenceHistory: [],
      currentEpoch: 0,
      currentBatch: 0,
    });
  },

  runTrainStep: async (input, target) => {
    const { modelId, modelConfig, currentEpoch, currentBatch } = get();
    if (modelId === null) return;

    const result = await api.trainStep(
      modelId, input, target,
      input.length / modelConfig.hidden_dim,
      modelConfig.hidden_dim,
      currentEpoch, currentBatch,
    );

    set((s) => ({
      lossHistory: [...s.lossHistory, result.loss],
      coherenceHistory: [...s.coherenceHistory, result.min_coherence],
      spectrumSnapshot: result.spectrum_snapshot,
      currentBatch: s.currentBatch + 1,
    }));
  },

  checkCoherence: async () => {
    const { modelId } = get();
    if (modelId === null) return;
    const reports = await api.checkCoherence(modelId);
    set({ coherenceReports: reports });
  },

  computeBudget: async (numDevices, vramGb) => {
    const { modelConfig, trainConfig } = get();
    const result = await api.allocateBudget(modelConfig, numDevices, vramGb, trainConfig.top_k);
    set({ budgetResult: result });
  },

  reset: () =>
    set({
      modelId: null,
      initResult: null,
      isTraining: false,
      currentEpoch: 0,
      currentBatch: 0,
      lossHistory: [],
      coherenceHistory: [],
      spectrumSnapshot: [],
      budgetResult: null,
      coherenceReports: [],
    }),
}));
