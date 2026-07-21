use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use rsmf_core::{ResonantTensor, Stratum, SpectralConfig};

/// A single expert's weight representation in MoE RSMF.
///
/// Each expert maintains its own spectral stratum, updated independently
/// during training. Only active experts are loaded into VRAM.
#[derive(Debug, Clone)]
pub struct ExpertStratum {
    /// Expert index within the MoE layer.
    pub expert_id: usize,
    /// Parent MoE layer index.
    pub layer_id: usize,
    /// Resonant tensor for this expert's weights.
    pub tensor: ResonantTensor,
    /// Cumulative spectral energy (for load balancing).
    pub cumulative_energy: f64,
    /// Number of tokens routed to this expert in current batch.
    pub active_token_count: usize,
    /// Whether this expert is currently loaded in VRAM.
    pub is_loaded: bool,
}

impl ExpertStratum {
    /// Create a new expert stratum with initialized weights.
    pub fn new(expert_id: usize, layer_id: usize, hidden_dim: usize, config: &SpectralConfig) -> Self {
        let k = config.top_k.min(hidden_dim);
        // Initialize with decaying spectrum (stable start)
        let sigma = Array1::from_vec(
            (0..k).map(|i| 1.0 / (1.0 + i as f64 * 0.5)).collect()
        );
        let u = Array2::eye(hidden_dim).slice(ndarray::s![.., ..k]).to_owned();
        let v = Array2::eye(hidden_dim).slice(ndarray::s![.., ..k]).to_owned();
        let stratum = Stratum::new(sigma, u, v, layer_id);

        Self {
            expert_id,
            layer_id,
            tensor: ResonantTensor::from_stratum(stratum),
            cumulative_energy: 0.0,
            active_token_count: 0,
            is_loaded: false,
        }
    }

    /// Record that this expert processed N tokens.
    pub fn record_activity(&mut self, token_count: usize, energy: f64) {
        self.active_token_count += token_count;
        self.cumulative_energy += energy;
    }

    /// Reset per-batch counters.
    pub fn reset_batch_stats(&mut self) {
        self.active_token_count = 0;
    }

    /// Load balance score: deviation from uniform expert utilization.
    /// Returns 0.0 for perfectly balanced, higher for imbalanced.
    pub fn load_imbalance_score(&self, mean_energy: f64) -> f64 {
        if mean_energy < 1e-15 { return 0.0; }
        ((self.cumulative_energy - mean_energy) / mean_energy).abs()
    }

    /// Memory footprint when loaded.
    pub fn memory_bytes(&self) -> usize {
        self.tensor.memory_bytes()
    }
}

/// Collection of experts for one MoE layer.
#[derive(Debug, Clone)]
pub struct ExpertGroup {
    pub layer_id: usize,
    pub experts: Vec<ExpertStratum>,
    pub num_active: usize, // top-k routing
}

impl ExpertGroup {
    pub fn new(layer_id: usize, num_experts: usize, num_active: usize, hidden_dim: usize, config: &SpectralConfig) -> Self {
        let experts = (0..num_experts)
            .map(|id| ExpertStratum::new(id, layer_id, hidden_dim, config))
            .collect();
        Self { layer_id, experts, num_active }
    }

    /// Get indices of top-k experts by routing scores.
    pub fn select_experts(&self, routing_scores: &[f64]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = routing_scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(self.num_active).map(|(i, _)| i).collect()
    }

    /// Mean spectral energy across all experts.
    pub fn mean_energy(&self) -> f64 {
        let total: f64 = self.experts.iter().map(|e| e.cumulative_energy).sum();
        total / self.experts.len() as f64
    }

    /// Total memory if all experts loaded.
    pub fn total_memory_bytes(&self) -> usize {
        self.experts.iter().map(|e| e.memory_bytes()).sum()
    }

    /// Memory for only active experts.
    pub fn active_memory_bytes(&self) -> usize {
        // Estimate: num_active experts loaded at once
        let per_expert = self.experts.first().map(|e| e.memory_bytes()).unwrap_or(0);
        per_expert * self.num_active
    }

    /// Reset batch stats for all experts.
    pub fn reset_batch_stats(&mut self) {
        for expert in &mut self.experts {
            expert.reset_batch_stats();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_group_creation() {
        let config = SpectralConfig { top_k: 8, ..SpectralConfig::default() };
        let group = ExpertGroup::new(0, 8, 2, 64, &config);
        assert_eq!(group.experts.len(), 8);
        assert_eq!(group.num_active, 2);
    }

    #[test]
    fn select_top_k_experts() {
        let config = SpectralConfig { top_k: 4, ..SpectralConfig::default() };
        let group = ExpertGroup::new(0, 4, 2, 16, &config);
        let scores = vec![0.1, 0.9, 0.3, 0.7];
        let selected = group.select_experts(&scores);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&1)); // Highest score
        assert!(selected.contains(&3)); // Second highest
    }

    #[test]
    fn active_memory_much_less_than_total() {
        let config = SpectralConfig { top_k: 8, ..SpectralConfig::default() };
        let group = ExpertGroup::new(0, 64, 4, 256, &config);
        let active = group.active_memory_bytes();
        let total = group.total_memory_bytes();
        assert!(active < total / 10, "Active should be << total: {} vs {}", active, total);
    }
}
