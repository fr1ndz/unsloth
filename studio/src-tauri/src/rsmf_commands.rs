//! RSMF Tauri commands — exposes the RSMF training engine to the Studio frontend.
//!
//! These commands bridge the Rust RSMF crate with the React/R3F UI via
//! Tauri's IPC mechanism. All heavy computation happens in Rust; the
//! frontend receives serializable snapshots for visualization.

use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Mutex;

use rsmf_core::SpectralConfig;
use rsmf_layers::RsmfModel;
use rsmf_train::{RsmfTrainer, ResonantLoss};
use rsmf_distributed::model_config::{ModelArchitecture, DenseConfig, MoeConfig, ShardStrategy};
use rsmf_distributed::budget::VramBudgetAllocator;
use rsmf_distributed::topology::DeviceTopology;
use rsmf_moe::MoeRsmfModel;
use ndarray::Array2;

// ── Serializable DTOs ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsmfModelConfig {
    pub architecture: String, // "dense" | "moe"
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    // MoE-specific (ignored for dense)
    pub num_experts: Option<usize>,
    pub active_experts: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsmfTrainConfig {
    pub top_k: usize,
    pub lambda_spectral: f64,
    pub mu_coupling: f64,
    pub learning_rate: f64,
    pub coherence_threshold: f64,
    pub max_inner_iters: usize,
    pub loss_type: String, // "mse" | "cross_entropy"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsmfInitResult {
    pub model_id: u64,
    pub total_parameters: u64,
    pub compression_ratio: f64,
    pub vram_per_stratum_bytes: usize,
    pub spectrum_snapshot: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsmfTrainStepResult {
    pub loss: f64,
    pub min_coherence: f64,
    pub corrections_applied: usize,
    pub avg_inner_iters: f64,
    pub spectrum_snapshot: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsmfBudgetResult {
    pub feasible: bool,
    pub allocations: Vec<RsmfDeviceAllocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsmfDeviceAllocation {
    pub device_id: usize,
    pub vram_budget_bytes: u64,
    pub assigned_layers: Vec<usize>,
    pub recommended_top_k: usize,
    pub estimated_stratum_bytes: u64,
    pub headroom_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsmfCoherenceReport {
    pub intra_expert_ok: Vec<bool>,
    pub inter_expert_diversity: f64,
    pub inter_layer_coherence: Option<f64>,
    pub overall_healthy: bool,
    pub warnings: Vec<String>,
}

// ── Managed State ──────────────────────────────────────────────────────────

pub struct RsmfState {
    inner: Mutex<RsmfStateInner>,
}

struct RsmfStateInner {
    next_id: u64,
    models: Vec<(u64, RsmfTrainer)>,
    moe_models: Vec<(u64, MoeRsmfModel)>,
}

impl RsmfState {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(RsmfStateInner {
                next_id: 1,
                models: Vec::new(),
                moe_models: Vec::new(),
            }),
        }
    }
}

// ── Tauri Commands ─────────────────────────────────────────────────────────

#[tauri::command]
pub fn rsmf_init_model(
    state: State<RsmfState>,
    model_config: RsmfModelConfig,
    train_config: RsmfTrainConfig,
) -> Result<RsmfInitResult, String> {
    let spectral = SpectralConfig {
        top_k: train_config.top_k,
        lambda_spectral: train_config.lambda_spectral,
        mu_coupling: train_config.mu_coupling,
        learning_rate: train_config.learning_rate,
        coherence_threshold: train_config.coherence_threshold,
        max_inner_iters: train_config.max_inner_iters,
        ..SpectralConfig::default()
    };

    let loss_fn = match train_config.loss_type.as_str() {
        "cross_entropy" => ResonantLoss::cross_entropy(),
        _ => ResonantLoss::mse(),
    };

    let mut inner = state.inner.lock().map_err(|e| e.to_string())?;
    let model_id = inner.next_id;
    inner.next_id += 1;

    match model_config.architecture.as_str() {
        "moe" => {
            let moe_cfg = MoeConfig {
                base: DenseConfig {
                    total_params: 0, // Will be computed
                    hidden_dim: model_config.hidden_dim,
                    num_layers: model_config.num_layers,
                    num_heads: model_config.num_heads,
                    intermediate_dim: model_config.intermediate_dim,
                    vocab_size: model_config.vocab_size,
                },
                num_experts: model_config.num_experts.unwrap_or(8),
                active_experts: model_config.active_experts.unwrap_or(2),
                moe_layer_indices: (0..model_config.num_layers).collect(),
                expert_intermediate_dim: model_config.intermediate_dim,
            };
            let moe_model = MoeRsmfModel::initialize(moe_cfg, spectral.clone());
            let total_params = moe_model.total_parameters();
            let active_vram = moe_model.active_vram_estimate();

            inner.moe_models.push((model_id, moe_model));

            Ok(RsmfInitResult {
                model_id,
                total_parameters: total_params,
                compression_ratio: 0.0, // MoE uses different metric
                vram_per_stratum_bytes: active_vram,
                spectrum_snapshot: vec![],
            })
        }
        _ => {
            // Dense model
            let model = RsmfModel::initialize(
                model_config.num_layers,
                model_config.hidden_dim,
                model_config.num_heads,
                spectral,
            );
            let total_params = model.total_parameters() as u64;
            let compression = model.compression_ratio();
            let vram = model.config.vram_per_stratum(model_config.hidden_dim);
            let spectrum: Vec<Vec<f64>> = model.spectrum_snapshot()
                .iter().map(|s| s.to_vec()).collect();

            let trainer = RsmfTrainer::new(model, loss_fn);
            inner.models.push((model_id, trainer));

            Ok(RsmfInitResult {
                model_id,
                total_parameters: total_params,
                compression_ratio: compression,
                vram_per_stratum_bytes: vram,
                spectrum_snapshot: spectrum,
            })
        }
    }
}

#[tauri::command]
pub fn rsmf_train_step(
    state: State<RsmfState>,
    model_id: u64,
    input_flat: Vec<f64>,
    target_flat: Vec<f64>,
    batch_size: usize,
    hidden_dim: usize,
    epoch: usize,
    batch_idx: usize,
) -> Result<RsmfTrainStepResult, String> {
    let mut inner = state.inner.lock().map_err(|e| e.to_string())?;

    let (_, trainer) = inner.models.iter_mut()
        .find(|(id, _)| *id == model_id)
        .ok_or_else(|| format!("Model {} not found", model_id))?;

    let input = Array2::from_shape_vec((batch_size, hidden_dim), input_flat)
        .map_err(|e| format!("Input shape error: {}", e))?;
    let target = Array2::from_shape_vec((batch_size, hidden_dim), target_flat)
        .map_err(|e| format!("Target shape error: {}", e))?;

    let stats = trainer.train_step(&input, &target, epoch, batch_idx);
    let spectrum: Vec<Vec<f64>> = trainer.model.spectrum_snapshot()
        .iter().map(|s| s.to_vec()).collect();

    Ok(RsmfTrainStepResult {
        loss: stats.loss,
        min_coherence: stats.min_coherence,
        corrections_applied: stats.corrections_applied,
        avg_inner_iters: stats.avg_inner_iters,
        spectrum_snapshot: spectrum,
    })
}

#[tauri::command]
pub fn rsmf_check_coherence(
    state: State<RsmfState>,
    model_id: u64,
) -> Result<Vec<RsmfCoherenceReport>, String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;

    let (_, moe) = inner.moe_models.iter()
        .find(|(id, _)| *id == model_id)
        .ok_or_else(|| format!("MoE model {} not found", model_id))?;

    let reports = moe.check_coherence();
    Ok(reports.into_iter().map(|r| RsmfCoherenceReport {
        intra_expert_ok: r.intra_expert_ok,
        inter_expert_diversity: r.inter_expert_diversity,
        inter_layer_coherence: r.inter_layer_coherence,
        overall_healthy: r.overall_healthy,
        warnings: r.warnings,
    }).collect())
}

#[tauri::command]
pub fn rsmf_allocate_budget(
    model_config: RsmfModelConfig,
    num_devices: usize,
    vram_per_device_gb: u64,
    top_k: usize,
) -> Result<RsmfBudgetResult, String> {
    let arch = match model_config.architecture.as_str() {
        "moe" => {
            let moe = MoeConfig {
                base: DenseConfig {
                    total_params: 0,
                    hidden_dim: model_config.hidden_dim,
                    num_layers: model_config.num_layers,
                    num_heads: model_config.num_heads,
                    intermediate_dim: model_config.intermediate_dim,
                    vocab_size: model_config.vocab_size,
                },
                num_experts: model_config.num_experts.unwrap_or(8),
                active_experts: model_config.active_experts.unwrap_or(2),
                moe_layer_indices: (0..model_config.num_layers).collect(),
                expert_intermediate_dim: model_config.intermediate_dim,
            };
            ModelArchitecture::Moe(moe)
        }
        _ => {
            ModelArchitecture::Dense(DenseConfig {
                total_params: 0,
                hidden_dim: model_config.hidden_dim,
                num_layers: model_config.num_layers,
                num_heads: model_config.num_heads,
                intermediate_dim: model_config.intermediate_dim,
                vocab_size: model_config.vocab_size,
            })
        }
    };

    let topo = DeviceTopology::uniform(
        num_devices,
        vram_per_device_gb * 1024 * 1024 * 1024,
        model_config.num_layers,
    );
    let spectral = SpectralConfig { top_k, ..SpectralConfig::default() };

    let feasible = VramBudgetAllocator::is_feasible(&arch, &topo, &spectral);
    let allocs = VramBudgetAllocator::allocate(&arch, &topo, &spectral);

    Ok(RsmfBudgetResult {
        feasible,
        allocations: allocs.into_iter().map(|a| RsmfDeviceAllocation {
            device_id: a.device_id,
            vram_budget_bytes: a.vram_budget_bytes,
            assigned_layers: a.assigned_layers,
            recommended_top_k: a.recommended_top_k,
            estimated_stratum_bytes: a.estimated_stratum_bytes,
            headroom_bytes: a.headroom_bytes,
        }).collect(),
    })
}

#[tauri::command]
pub fn rsmf_get_spectrum(
    state: State<RsmfState>,
    model_id: u64,
) -> Result<Vec<Vec<f64>>, String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;
    let (_, trainer) = inner.models.iter()
        .find(|(id, _)| *id == model_id)
        .ok_or_else(|| format!("Model {} not found", model_id))?;

    Ok(trainer.model.spectrum_snapshot()
        .iter().map(|s| s.to_vec()).collect())
}
