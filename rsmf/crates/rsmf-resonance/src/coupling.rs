use ndarray::Array1;
use rsmf_core::{CoherenceMetric, ResonantTensor, SpectralConfig};

/// Inter-stratum coupling operator Γ.
///
/// Enforces geometric coherence between adjacent layers during training.
/// Γ(Ψₗ, Ψₗ₋₁, Ψₗ₊₁) = α·‖spec(Ψₗ) - spec(Ψₗ₋₁)‖² + β·subspace_overlap(Uₗ, Uₗ₊₁)
pub struct InterStratumCoupling<'a> {
    config: &'a SpectralConfig,
}

impl<'a> InterStratumCoupling<'a> {
    pub fn new(config: &'a SpectralConfig) -> Self {
        Self { config }
    }

    /// Compute coupling loss between two adjacent strata.
    pub fn coupling_loss(&self, a: &ResonantTensor, b: &ResonantTensor) -> f64 {
        let metric = CoherenceMetric::between(a, b);
        // Loss = μ · (1 - coherence_score)²
        self.config.mu_coupling * (1.0 - metric.score).powi(2)
    }

    /// Compute coupling gradient for spectral values.
    /// ∂Γ/∂σᵢ = 2μ·(σᵢ - σ_adjacent_i) for aligned indices.
    pub fn spectral_gradient(
        &self,
        current: &Array1<f64>,
        adjacent: &Array1<f64>,
    ) -> Array1<f64> {
        let k = current.len();
        let mut grad = Array1::zeros(k);
        let adj_len = adjacent.len();
        for i in 0..k {
            let adj_val = if i < adj_len { adjacent[i] } else { 0.0 };
            grad[i] = 2.0 * self.config.mu_coupling * (current[i] - adj_val);
        }
        grad
    }

    /// Check if entire chain of strata has safe coherence.
    pub fn chain_is_coherent(&self, tensors: &[ResonantTensor]) -> bool {
        if tensors.len() < 2 {
            return true;
        }
        for window in tensors.windows(2) {
            let metric = CoherenceMetric::between(&window[0], &window[1]);
            if !metric.is_safe(self.config.coherence_threshold) {
                return false;
            }
        }
        true
    }

    /// Find the weakest link in the stratum chain.
    pub fn weakest_link(&self, tensors: &[ResonantTensor]) -> Option<(usize, f64)> {
        if tensors.len() < 2 {
            return None;
        }
        let mut min_score = f64::INFINITY;
        let mut min_idx = 0;
        for (i, window) in tensors.windows(2).enumerate() {
            let metric = CoherenceMetric::between(&window[0], &window[1]);
            if metric.score < min_score {
                min_score = metric.score;
                min_idx = i;
            }
        }
        Some((min_idx, min_score))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rsmf_core::Stratum;
    use ndarray::Array2;

    fn make_tensor(sigma: &[f64], id: usize) -> ResonantTensor {
        let k = sigma.len();
        let s = Array1::from_vec(sigma.to_vec());
        let u = Array2::eye(k);
        let v = Array2::eye(k);
        ResonantTensor::from_stratum(Stratum::new(s, u, v, id))
    }

    #[test]
    fn identical_chain_is_coherent() {
        let config = SpectralConfig::default();
        let coupling = InterStratumCoupling::new(&config);
        let tensors: Vec<_> = (0..5).map(|i| make_tensor(&[3.0, 2.0, 1.0], i)).collect();
        assert!(coupling.chain_is_coherent(&tensors));
    }

    #[test]
    fn divergent_chain_detected() {
        let config = SpectralConfig { coherence_threshold: 0.9, ..SpectralConfig::default() };
        let coupling = InterStratumCoupling::new(&config);
        let tensors = vec![
            make_tensor(&[10.0, 1.0, 0.1], 0),
            make_tensor(&[0.1, 1.0, 10.0], 1), // Inverted spectrum
        ];
        assert!(!coupling.chain_is_coherent(&tensors));
    }
}
