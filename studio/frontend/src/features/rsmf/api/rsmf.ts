/**
 * RSMF API bridge — calls Tauri IPC commands for RSMF training engine.
 */
import { invoke } from '@tauri-apps/api/core';

export interface RsmfModelConfig {
  architecture: 'dense' | 'moe';
  hidden_dim: number;
  num_layers: number;
  num_heads: number;
  intermediate_dim: number;
  vocab_size: number;
  num_experts?: number;
  active_experts?: number;
}

export interface RsmfTrainConfig {
  top_k: number;
  lambda_spectral: number;
  mu_coupling: number;
  learning_rate: number;
  coherence_threshold: number;
  max_inner_iters: number;
  loss_type: 'mse' | 'cross_entropy';
}

export interface RsmfInitResult {
  model_id: number;
  total_parameters: number;
  compression_ratio: number;
  vram_per_stratum_bytes: number;
  spectrum_snapshot: number[][];
}

export interface RsmfTrainStepResult {
  loss: number;
  min_coherence: number;
  corrections_applied: number;
  avg_inner_iters: number;
  spectrum_snapshot: number[][];
}

export interface RsmfBudgetResult {
  feasible: boolean;
  allocations: {
    device_id: number;
    vram_budget_bytes: number;
    assigned_layers: number[];
    recommended_top_k: number;
    estimated_stratum_bytes: number;
    headroom_bytes: number;
  }[];
}

export interface RsmfCoherenceReport {
  intra_expert_ok: boolean[];
  inter_expert_diversity: number;
  inter_layer_coherence: number | null;
  overall_healthy: boolean;
  warnings: string[];
}

export async function initModel(
  modelConfig: RsmfModelConfig,
  trainConfig: RsmfTrainConfig,
): Promise<RsmfInitResult> {
  return invoke('rsmf_init_model', { modelConfig, trainConfig });
}

export async function trainStep(
  modelId: number,
  inputFlat: number[],
  targetFlat: number[],
  batchSize: number,
  hiddenDim: number,
  epoch: number,
  batchIdx: number,
): Promise<RsmfTrainStepResult> {
  return invoke('rsmf_train_step', {
    modelId, inputFlat, targetFlat, batchSize, hiddenDim, epoch, batchIdx,
  });
}

export async function checkCoherence(modelId: number): Promise<RsmfCoherenceReport[]> {
  return invoke('rsmf_check_coherence', { modelId });
}

export async function allocateBudget(
  modelConfig: RsmfModelConfig,
  numDevices: number,
  vramPerDeviceGb: number,
  topK: number,
): Promise<RsmfBudgetResult> {
  return invoke('rsmf_allocate_budget', { modelConfig, numDevices, vramPerDeviceGb, topK });
}

export async function getSpectrum(modelId: number): Promise<number[][]> {
  return invoke('rsmf_get_spectrum', { modelId });
}
