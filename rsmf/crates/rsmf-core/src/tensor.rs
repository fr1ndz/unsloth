use ndarray::{Array1, Array2};
use crate::stratum::Stratum;
use crate::config::SpectralConfig;
use crate::svd;

/// Resonant Tensor Ψₗ = σ ⊗ U ⊗ Vᵀ with resonance metadata.
///
/// Extends [`Stratum`] with dynamic state needed during training:
/// - Resonance phase (for temporal coherence tracking)
/// - Local energy estimate 𝔼ₗ
/// - Coupling state with adjacent strata
#[derive(Debug, Clone)]
pub struct ResonantTensor {
    /// Underlying spectral stratum.
    pub stratum: Stratum,
    /// Resonance phase φ ∈ [0, 2π) — tracks oscillatory dynamics.
    pub phase: f64,
    /// Local activation energy 𝔼ₗ = ‖Aₗ‖_F² / batch_size.
    pub local_energy: f64,
    /// Spectral condition number: σ_max / σ_min.
    /// High values indicate ill-conditioned stratum.
    pub condition_number: f64,
    /// Whether this stratum has been updated in current epoch.
    pub is_updated: bool,
}

impl ResonantTensor {
    /// Wrap a stratum into a resonant tensor with initial metadata.
    pub fn from_stratum(stratum: Stratum) -> Self {
        let condition_number = compute_condition_number(&stratum.sigma);
        Self {
            stratum,
            phase: 0.0,
            local_energy: 0.0,
            condition_number,
            is_updated: false,
        }
    }

    /// Update local energy from activation matrix.
    /// 𝔼ₗ = ‖A‖_F² / batch_size
    pub fn update_energy(&mut self, activations: &Array2<f64>) {
        let batch_size = activations.nrows() as f64;
        self.local_energy = activations.mapv(|x| x * x).sum() / batch_size;
    }

    /// Advance resonance phase based on local energy and config.
    /// φ ← φ + ω·𝔼ₗ·dt, wrapped to [0, 2π)
    pub fn advance_phase(&mut self, dt: f64, config: &SpectralConfig) {
        let omega = config.learning_rate * self.local_energy.sqrt();
        self.phase += omega * dt;
        self.phase %= std::f64::consts::TAU;
    }

    /// Check if stratum is well-conditioned for stable updates.
    pub fn is_well_conditioned(&self, max_cond: f64) -> bool {
        self.condition_number < max_cond
    }

    /// Apply spectral regularization: shrink small singular values.
    /// σᵢ ← σᵢ · max(1 - λ/σᵢ², 0)
    pub fn regularize_spectrum(&mut self, lambda: f64) {
        for s in self.stratum.sigma.iter_mut() {
            let s2 = *s * *s;
            if s2 > lambda {
                *s *= (1.0 - lambda / s2).max(0.0);
            } else {
                *s = 0.0;
            }
        }
        self.condition_number = compute_condition_number(&self.stratum.sigma);
    }

    /// Memory footprint including metadata.
    pub fn memory_bytes(&self) -> usize {
        self.stratum.memory_bytes() + 4 * std::mem::size_of::<f64>() + std::mem::size_of::<bool>()
    }

    // ── RsLoRA Scaling ──────────────────────────────────────────────

    /// Compute RsLoRA scaling factor: α / √rank.
    ///
    /// Unlike linear LoRA scaling (α/rank), this preserves gradient magnitude
    /// across rank changes, following the RsLoRA formulation.
    pub fn rslora_scale(&self, alpha: f64) -> f64 {
        let rank = self.stratum.rank() as f64;
        if rank < 1.0 { return 0.0; }
        alpha / rank.sqrt()
    }

    // ── Stable Rank Monitoring ──────────────────────────────────────

    /// Nuclear norm: ‖W‖_* = Σ σᵢ (sum of singular values).
    pub fn nuclear_norm(&self) -> f64 {
        svd::nuclear_norm(&self.stratum.sigma)
    }

    /// Spectral norm: ‖W‖₂ = σ_max (largest singular value).
    pub fn spectral_norm(&self) -> f64 {
        svd::spectral_norm(&self.stratum.sigma)
    }

    /// Stable rank: sr(W) = (‖W‖_* / ‖W‖₂)².
    ///
    /// Measures effective dimensionality of the weight spectrum.
    /// - sr ≈ 1: rank-1 dominant (collapsed spectrum)
    /// - sr ≈ k: uniform spectrum (well-distributed energy)
    /// Useful for detecting spectral collapse during training.
    pub fn stable_rank(&self) -> f64 {
        svd::stable_rank(&self.stratum.sigma)
    }

    // ── Adaptive Rank ───────────────────────────────────────────────

    /// Adaptively adjust rank based on energy ratio threshold from config.
    ///
    /// If captured energy / total_energy > threshold, the current rank
    /// is sufficient. If below, more components are needed.
    ///
    /// Returns the recommended new rank k' ∈ [1, max_rank].
    /// The caller is responsible for actually resizing the stratum
    /// (re-running SVD or truncating).
    ///
    /// # Arguments
    /// * `total_energy` - Total spectral energy (Σ σᵢ² of full matrix)
    /// * `energy_threshold` - Minimum acceptable energy ratio (e.g., 0.95)
    /// * `max_rank` - Upper bound on rank (typically hidden_dim)
    pub fn adapt_rank(&self, total_energy: f64, energy_threshold: f64, max_rank: usize) -> usize {
        let current_ratio = self.stratum.energy_ratio(total_energy);
        let current_k = self.stratum.rank();

        if current_ratio >= energy_threshold {
            // Energy captured is sufficient — potentially shrink
            // Find minimal k that still captures enough energy
            let mut cumulative = 0.0;
            for (i, &s) in self.stratum.sigma.iter().enumerate() {
                cumulative += s * s;
                if cumulative / total_energy >= energy_threshold {
                    return (i + 1).max(1);
                }
            }
            current_k
        } else {
            // Not enough energy captured — grow rank
            // Estimate how many more components we need
            // Assume remaining spectrum decays similarly to observed tail
            let deficit = energy_threshold - current_ratio;
            let avg_remaining_per_component = if current_k > 0 {
                let last_sigma = self.stratum.sigma[current_k - 1];
                (last_sigma * last_sigma).max(1e-30)
            } else {
                total_energy / max_rank as f64
            };
            let extra_needed = (deficit * total_energy / avg_remaining_per_component).ceil() as usize;
            (current_k + extra_needed.max(1)).min(max_rank)
        }
    }
}

fn compute_condition_number(sigma: &Array1<f64>) -> f64 {
    let max = sigma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = sigma.iter().cloned().filter(|&s| s > 1e-15).fold(f64::INFINITY, f64::min);
    if min < 1e-15 { f64::INFINITY } else { max / min }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn condition_number_correct() {
        let sigma = array![4.0, 2.0, 1.0];
        let u = Array2::eye(3);
        let v = Array2::eye(3);
        let stratum = Stratum::new(sigma, u, v, 0);
        let rt = ResonantTensor::from_stratum(stratum);
        assert!((rt.condition_number - 4.0).abs() < 1e-10);
    }

    #[test]
    fn regularization_shrinks_small_singular_values() {
        let sigma = array![10.0, 1.0, 0.1];
        let u = Array2::eye(3);
        let v = Array2::eye(3);
        let stratum = Stratum::new(sigma, u, v, 0);
        let mut rt = ResonantTensor::from_stratum(stratum);
        rt.regularize_spectrum(0.5);
        // σ=10: 10*(1-0.5/100)=9.95, σ=1: 1*(1-0.5/1)=0.5, σ=0.1: 0 (0.01<0.5)
        assert!((rt.stratum.sigma[0] - 9.95).abs() < 1e-10);
        assert!((rt.stratum.sigma[1] - 0.5).abs() < 1e-10);
        assert!(rt.stratum.sigma[2].abs() < 1e-15);
    }
}
