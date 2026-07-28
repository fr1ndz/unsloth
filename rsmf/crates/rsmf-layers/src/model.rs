use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig, Stratum, randomized_svd};
use crate::forward::StratifiedForward;
use crate::activation_cache::ActivationCache;

/// Complete RSMF model: collection of resonant tensors representing all layers.
///
/// Manages the lifecycle of strata across training steps:
/// - Initialization from random or pretrained weights
/// - Sequential forward/backward with memory cycling
/// - Checkpoint save/load for fault tolerance
pub struct RsmfModel {
    /// All layer resonant tensors.
    pub layers: Vec<ResonantTensor>,
    /// Model configuration.
    pub config: SpectralConfig,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads (for multi-head architectures).
    pub num_heads: usize,
}

impl RsmfModel {
    /// Initialize model with random orthogonal weights.
    ///
    /// Each layer gets a random weight matrix decomposed via truncated SVD.
    pub fn initialize(
        num_layers: usize,
        hidden_dim: usize,
        num_heads: usize,
        config: SpectralConfig,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let k = config.top_k.min(hidden_dim);

            // Generate a random weight matrix with Xavier-like scaling,
            // then decompose via randomized truncated SVD.
            let scale = (2.0 / hidden_dim as f64).sqrt();
            let mut rng = rand::thread_rng();
            let w = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                use rand_distr::{Distribution, StandardNormal};
                let z: f64 = StandardNormal.sample(&mut rng);
                z * scale
            });

            // Randomized SVD: rank-k approximation with oversampling + power iteration
            let n_oversamples = 5.min(hidden_dim - k);
            let svd = randomized_svd(&w, k, n_oversamples, 2);

            let stratum = Stratum::new(svd.sigma, svd.u, svd.v, l);
            layers.push(ResonantTensor::from_stratum(stratum));
        }

        Self { layers, config, hidden_dim, num_heads }
    }

    /// Total parameter count (reconstructed).
    pub fn total_parameters(&self) -> usize {
        self.layers.len() * self.hidden_dim * self.hidden_dim
    }

    /// Active parameters in spectral representation.
    pub fn active_spectral_parameters(&self) -> usize {
        self.layers.iter().map(|t| {
            let k = t.stratum.rank();
            k + 2 * k * self.hidden_dim // σ + U + V
        }).sum()
    }

    /// Memory compression ratio vs full parameter storage.
    pub fn compression_ratio(&self) -> f64 {
        let full = self.total_parameters() * std::mem::size_of::<f64>();
        let spectral: usize = self.layers.iter().map(|t| t.memory_bytes()).sum();
        if spectral == 0 { return 1.0; }
        full as f64 / spectral as f64
    }

    /// Run full forward pass.
    pub fn forward(
        &mut self,
        input: &Array2<f64>,
    ) -> (Array2<f64>, Vec<ActivationCache>) {
        let fwd = StratifiedForward::new(&self.config);
        fwd.forward_all(input, &mut self.layers)
    }

    /// Get spectrum snapshot for monitoring.
    pub fn spectrum_snapshot(&self) -> Vec<Array1<f64>> {
        self.layers.iter()
            .map(|t| t.stratum.sigma.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_initialization_correct() {
        let config = SpectralConfig { top_k: 8, ..SpectralConfig::default() };
        let model = RsmfModel::initialize(4, 64, 4, config);

        assert_eq!(model.layers.len(), 4);
        assert_eq!(model.total_parameters(), 4 * 64 * 64);
        assert!(model.compression_ratio() > 1.0);
    }

    #[test]
    fn forward_runs_without_panic() {
        let config = SpectralConfig { top_k: 4, ..SpectralConfig::default() };
        let mut model = RsmfModel::initialize(2, 16, 2, config);

        let input = Array2::ones((2, 16));
        let (output, caches) = model.forward(&input);

        assert_eq!(output.shape(), &[2, 16]);
        assert_eq!(caches.len(), 2);
    }
}
