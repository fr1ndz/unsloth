use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig, Stratum};
use rsmf_distributed::model_config::MoeConfig;
use crate::expert::ExpertGroup;
use crate::routing::RoutingStratum;
use crate::hierarchical_coherence::{HierarchicalCoherence, CoherenceReport};

/// Complete MoE model for RSMF training.
///
/// Combines:
/// - Dense attention layers (standard RSMF strata)
/// - MoE FFN layers (expert groups + shared routing)
/// - Hierarchical coherence monitoring
pub struct MoeRsmfModel {
    /// Dense (non-MoE) layer strata.
    pub dense_layers: Vec<ResonantTensor>,
    /// MoE layers with expert groups.
    pub moe_layers: Vec<ExpertGroup>,
    /// Shared routing strata (one per MoE layer).
    pub routing_strata: Vec<RoutingStratum>,
    /// Which layer indices are MoE vs dense.
    pub moe_layer_indices: Vec<usize>,
    /// Model configuration.
    pub config: MoeConfig,
    /// Spectral configuration.
    pub spectral_config: SpectralConfig,
}

impl MoeRsmfModel {
    /// Initialize MoE model from config.
    pub fn initialize(moe_config: MoeConfig, spectral_config: SpectralConfig) -> Self {
        let hidden_dim = moe_config.base.hidden_dim;
        let num_layers = moe_config.base.num_layers;

        // Initialize dense layers
        let mut dense_layers = Vec::new();
        for l in 0..num_layers {
            if !moe_config.moe_layer_indices.contains(&l) {
                let k = spectral_config.top_k.min(hidden_dim);
                let sigma = Array1::from_vec(
                    (0..k).map(|i| 1.0 / (1.0 + i as f64)).collect()
                );
                let u = Array2::eye(hidden_dim).slice(ndarray::s![.., ..k]).to_owned();
                let v = Array2::eye(hidden_dim).slice(ndarray::s![.., ..k]).to_owned();
                let stratum = Stratum::new(sigma, u, v, l);
                dense_layers.push(ResonantTensor::from_stratum(stratum));
            }
        }

        // Initialize MoE layers
        let mut moe_layers = Vec::new();
        let mut routing_strata = Vec::new();
        for &l in &moe_config.moe_layer_indices {
            moe_layers.push(ExpertGroup::new(
                l,
                moe_config.num_experts,
                moe_config.active_experts,
                hidden_dim,
                &spectral_config,
            ));
            routing_strata.push(RoutingStratum::new(
                hidden_dim,
                moe_config.num_experts,
                &spectral_config,
            ));
        }

        Self {
            dense_layers,
            moe_layers,
            routing_strata,
            moe_layer_indices: moe_config.moe_layer_indices.clone(),
            config: moe_config,
            spectral_config,
        }
    }

    /// Total parameter count (all experts + dense).
    pub fn total_parameters(&self) -> u64 {
        let d = self.config.base.hidden_dim as u64;
        let dense_params = self.dense_layers.len() as u64 * d * d;
        let expert_params = self.moe_layers.iter().map(|g| {
            g.experts.len() as u64 * d * (self.config.expert_intermediate_dim as u64) * 3
        }).sum::<u64>();
        dense_params + expert_params
    }

    /// Active parameters per token (dense + top-k experts).
    pub fn active_parameters_per_token(&self) -> u64 {
        let d = self.config.base.hidden_dim as u64;
        let dense = self.dense_layers.len() as u64 * d * d;
        let active_experts = self.config.active_experts as u64;
        let expert_per_layer = active_experts * d * (self.config.expert_intermediate_dim as u64) * 3;
        dense + expert_per_layer * self.moe_layers.len() as u64
    }

    /// Run hierarchical coherence check on all MoE layers.
    pub fn check_coherence(&self) -> Vec<CoherenceReport> {
        let hc = HierarchicalCoherence::new(self.spectral_config.clone());
        let mut reports = Vec::new();
        for (i, group) in self.moe_layers.iter().enumerate() {
            let prev = if i > 0 { Some(&self.moe_layers[i - 1]) } else { None };
            reports.push(hc.check_layer(group, prev));
        }
        reports
    }

    /// Memory estimate for active components only (per-token VRAM).
    pub fn active_vram_estimate(&self) -> usize {
        let per_expert = self.moe_layers.first()
            .and_then(|g| g.experts.first())
            .map(|e| e.memory_bytes())
            .unwrap_or(0);
        let active_expert_mem = per_expert * self.config.active_experts * self.moe_layers.len();
        let routing_mem: usize = self.routing_strata.iter().map(|r| r.memory_bytes()).sum();
        let dense_mem: usize = self.dense_layers.iter().map(|t| t.memory_bytes()).sum();
        active_expert_mem + routing_mem + dense_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsmf_distributed::model_config::{DenseConfig, MoeConfig};

    fn test_moe_config() -> MoeConfig {
        MoeConfig {
            base: DenseConfig {
                total_params: 1_000_000,
                hidden_dim: 64,
                num_layers: 4,
                num_heads: 4,
                intermediate_dim: 256,
                vocab_size: 1000,
            },
            num_experts: 4,
            active_experts: 2,
            moe_layer_indices: vec![1, 3],
            expert_intermediate_dim: 256,
        }
    }

    #[test]
    fn moe_model_initialization() {
        let cfg = test_moe_config();
        let spec = SpectralConfig { top_k: 8, ..SpectralConfig::default() };
        let model = MoeRsmfModel::initialize(cfg, spec);

        assert_eq!(model.dense_layers.len(), 2); // Layers 0, 2
        assert_eq!(model.moe_layers.len(), 2);   // Layers 1, 3
        assert_eq!(model.routing_strata.len(), 2);
    }

    #[test]
    fn active_params_much_less_than_total() {
        let cfg = test_moe_config();
        let spec = SpectralConfig { top_k: 8, ..SpectralConfig::default() };
        let model = MoeRsmfModel::initialize(cfg, spec);

        let total = model.total_parameters();
        let active = model.active_parameters_per_token();
        // With 2/4 experts active, should be ~60-70% of total
        assert!(active < total, "Active {} should be < total {}", active, total);
        assert!(active > total / 3, "Active should be significant fraction");
    }

    #[test]
    fn coherence_check_runs() {
        let cfg = test_moe_config();
        let spec = SpectralConfig { top_k: 4, coherence_threshold: 0.0, ..SpectralConfig::default() };
        let model = MoeRsmfModel::initialize(cfg, spec);
        let reports = model.check_coherence();
        assert_eq!(reports.len(), 2);
    }
}
