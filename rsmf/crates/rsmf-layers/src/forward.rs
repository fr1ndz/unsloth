use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig, Stratum};
use crate::activation_cache::ActivationCache;

/// Stratified forward pass with per-layer recording.
///
/// Processes one layer at a time, producing:
/// 1. Output activations for next layer
/// 2. Compressed activation cache for backward resonance
/// 3. Updated local energy in resonant tensor
///
/// Memory invariant: only current layer's weights + activations in VRAM.
pub struct StratifiedForward<'a> {
    config: &'a SpectralConfig,
}

impl<'a> StratifiedForward<'a> {
    pub fn new(config: &'a SpectralConfig) -> Self {
        Self { config }
    }

    /// Execute forward pass through a single layer.
    ///
    /// # Arguments
    /// * `input` — activations from previous layer, shape (batch, d_in)
    /// * `tensor` — resonant tensor containing weight decomposition
    ///
    /// # Returns
    /// * `output` — activations for next layer, shape (batch, d_out)
    /// * `cache` — compressed activation record for backward pass
    pub fn forward_layer(
        &self,
        input: &Array2<f64>,
        tensor: &mut ResonantTensor,
    ) -> (Array2<f64>, ActivationCache) {
        // Reconstruct weights on-demand (temporary allocation)
        let weights = tensor.stratum.reconstruct();

        // Linear transform: z = input · Wᵀ
        let pre_activation = input.dot(&weights.t());

        // Apply activation function (GELU approximation)
        let output = gelu_forward(&pre_activation);

        // Update local energy in resonant tensor
        tensor.update_energy(input);

        // Create compressed cache BEFORE dropping weights
        let cache = ActivationCache::from_activations(&output, &pre_activation, &tensor.stratum);

        // Weights are dropped here — freeing O(d²) memory
        drop(weights);

        (output, cache)
    }

    /// Execute full forward pass through all layers sequentially.
    ///
    /// Returns final output and vector of per-layer caches.
    /// Caches are stored in host RAM (not VRAM) — only current layer uses VRAM.
    pub fn forward_all(
        &self,
        input: &Array2<f64>,
        tensors: &mut [ResonantTensor],
    ) -> (Array2<f64>, Vec<ActivationCache>) {
        let mut current_input = input.clone();
        let mut caches = Vec::with_capacity(tensors.len());

        for tensor in tensors.iter_mut() {
            let (output, cache) = self.forward_layer(&current_input, tensor);
            caches.push(cache);
            current_input = output;
        }

        (current_input, caches)
    }
}

/// GELU activation forward pass.
fn gelu_forward(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| {
        let cdf = 0.5 * (1.0 + ((0.7978845608 * (v + 0.044715 * v.powi(3))).tanh()));
        v * cdf
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn forward_produces_correct_shapes() {
        let config = SpectralConfig::default();
        let fwd = StratifiedForward::new(&config);

        let sigma = array![2.0, 1.0];
        let u = Array2::eye(4).slice(ndarray::s![.., ..2]).to_owned();
        let v = Array2::eye(4).slice(ndarray::s![.., ..2]).to_owned();
        let stratum = Stratum::new(sigma, u, v, 0);
        let mut tensor = ResonantTensor::from_stratum(stratum);

        let input = Array2::ones((3, 4));
        let (output, cache) = fwd.forward_layer(&input, &mut tensor);

        assert_eq!(output.shape(), &[3, 4]);
        assert_eq!(cache.batch_size, 3);
        assert_eq!(cache.input_dim, 4);
    }

    #[test]
    fn gelu_approximation_reasonable() {
        let x = array![[0.0, 1.0, -1.0, 2.0]];
        let y = gelu_forward(&x);
        // GELU(0) ≈ 0, GELU(1) ≈ 0.84, GELU(-1) ≈ -0.16, GELU(2) ≈ 1.95
        assert!(y[[0, 0]].abs() < 0.01);
        assert!((y[[0, 1]] - 0.84).abs() < 0.05);
        assert!((y[[0, 2]] + 0.16).abs() < 0.05);
        assert!((y[[0, 3]] - 1.95).abs() < 0.1);
    }
}
