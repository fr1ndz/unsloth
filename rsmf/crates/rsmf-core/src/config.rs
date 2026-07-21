use serde::{Deserialize, Serialize};

/// Hyperparameters governing stratified manifold dynamics.
///
/// Each field maps to a component of the RSMF objective:
///   𝔏ₗ = ‖A·Ψ - T‖² + λ·Ω(Ψ) + μ·Γ(Ψ, Ψ₋₁, Ψ₊₁)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    /// Number of top singular values/vectors to retain per stratum.
    /// Controls memory footprint: O(k·d) vs full O(d²).
    /// For 4GB VRAM with d=4096: k=32 uses ~512KB per stratum.
    pub top_k: usize,

    /// Spectral regularizer strength (λ in Ω term).
    /// Penalizes deviation of singular spectrum from target distribution.
    /// Higher → more stable but slower adaptation.
    pub lambda_spectral: f64,

    /// Inter-stratum coupling strength (μ in Γ term).
    /// Enforces geometric coherence between adjacent layers.
    /// Higher → stronger global consistency, lower → more local freedom.
    pub mu_coupling: f64,

    /// Epsilon for numerical stability in resonant transition Φ.
    /// Prevents division by zero in σ/(σ+ε).
    pub epsilon: f64,

    /// Learning rate for local stratum updates.
    pub learning_rate: f64,

    /// Minimum coherence threshold before triggering global correction.
    pub coherence_threshold: f64,

    /// Maximum number of inner iterations per stratum update.
    pub max_inner_iters: usize,

    /// Convergence tolerance for inner stratum optimization.
    pub convergence_tol: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            top_k: 32,
            lambda_spectral: 0.01,
            mu_coupling: 0.1,
            epsilon: 1e-8,
            learning_rate: 1e-3,
            coherence_threshold: 0.3,
            max_inner_iters: 10,
            convergence_tol: 1e-6,
        }
    }
}

impl SpectralConfig {
    /// Estimate VRAM footprint per stratum for given hidden dimension.
    /// Returns bytes needed for one stratum's working set.
    pub fn vram_per_stratum(&self, hidden_dim: usize) -> usize {
        let elem_size = 2; // FP16
        // Top-k singular vectors: k × d × 2 (U and V)
        let svd_storage = self.top_k * hidden_dim * 2 * elem_size;
        // Singular values: k
        let sigma_storage = self.top_k * 8; // f64
        // Activation cache: d × batch_size (estimated batch=8)
        let activation_cache = hidden_dim * 8 * elem_size;
        // Resonance signal: d
        let resonance_signal = hidden_dim * elem_size;

        svd_storage + sigma_storage + activation_cache + resonance_signal
    }

    /// Check if config fits within VRAM budget for given model dimensions.
    pub fn fits_budget(&self, hidden_dim: usize, _num_layers: usize, budget_bytes: usize) -> bool {
        // Only ONE stratum in memory at a time during layer cycling
        let per_stratum = self.vram_per_stratum(hidden_dim);
        // Plus overhead for current weights being reconstructed
        let weight_reconstruction = hidden_dim * hidden_dim * 2; // FP16
        let total = per_stratum + weight_reconstruction;
        total <= budget_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_fits_4gb() {
        let cfg = SpectralConfig::default();
        // GPT-2 small: d=768, should easily fit
        assert!(cfg.fits_budget(768, 12, 4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn vram_estimate_reasonable() {
        let cfg = SpectralConfig::default();
        let vram = cfg.vram_per_stratum(4096);
        // Should be under 1MB for k=32, d=4096
        assert!(vram < 1024 * 1024, "VRAM per stratum too high: {} bytes", vram);
    }
}
