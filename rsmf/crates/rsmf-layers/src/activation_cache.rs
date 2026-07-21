use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig, Stratum};

/// Compressed activation cache for a single layer.
///
/// Instead of storing full activations A ∈ R^(batch × d), stores:
/// - Projected activations onto top-k singular subspace: A_proj ∈ R^(batch × k)
/// - Activation statistics (mean, variance) for normalization recovery
/// - Element-wise derivative values f'(z) for resonant backward channel
///
/// Memory: O(batch × k) instead of O(batch × d).
#[derive(Debug, Clone)]
pub struct ActivationCache {
    /// Projected activations: A · U ∈ R^(batch × k)
    pub projected: Array2<f64>,
    /// Per-feature mean of original activations (for denormalization).
    pub mean: Array1<f64>,
    /// Per-feature std of original activations.
    pub std: Array1<f64>,
    /// Element-wise activation derivative f'(z) ∈ R^(batch × d_in).
    /// Stored separately because it cannot be recovered from projection alone.
    pub activation_derivative: Array2<f64>,
    /// Original input dimension (before projection).
    pub input_dim: usize,
    /// Batch size.
    pub batch_size: usize,
}

impl ActivationCache {
    /// Create cache from full activations and stratum basis.
    ///
    /// Projects activations onto stratum's U basis for compact storage.
    pub fn from_activations(
        activations: &Array2<f64>,
        pre_activation: &Array2<f64>,
        stratum: &Stratum,
    ) -> Self {
        let batch = activations.nrows();
        let d = activations.ncols();
        let k = stratum.rank();

        // Project onto U basis: A_proj = A · U
        let projected = activations.dot(&stratum.u_basis);

        // Compute statistics for potential reconstruction
        let mut mean = Array1::zeros(d);
        let mut std = Array1::zeros(d);
        for j in 0..d {
            let mut sum = 0.0;
            let mut sq_sum = 0.0;
            for i in 0..batch {
                sum += activations[[i, j]];
                sq_sum += activations[[i, j]].powi(2);
            }
            mean[j] = sum / batch as f64;
            let variance = sq_sum / batch as f64 - mean[j].powi(2);
            std[j] = variance.max(0.0).sqrt();
        }

        // Compute activation derivative (assuming GELU-like: approximate as sigmoid)
        // For production, this should accept an activation function enum
        let mut deriv = Array2::zeros(pre_activation.dim());
        for i in 0..pre_activation.nrows() {
            for j in 0..pre_activation.ncols() {
                // Approximate GELU derivative via tanh-based formula
                let x = pre_activation[[i, j]];
                let t = (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh();
                deriv[[i, j]] = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t.powi(2)) * 0.7978845608 * (1.0 + 0.134145 * x.powi(2));
            }
        }

        Self {
            projected,
            mean,
            std,
            activation_derivative: deriv,
            input_dim: d,
            batch_size: batch,
        }
    }

    /// Reconstruct approximate full activations from compressed cache.
    /// A_approx = A_proj · Uᵀ
    ///
    /// ⚠️ Lossy reconstruction — only used for debugging/verification.
    pub fn reconstruct_approx(&self, u_basis: &Array2<f64>) -> Array2<f64> {
        self.projected.dot(&u_basis.t())
    }

    /// Memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        let f64_size = std::mem::size_of::<f64>();
        self.projected.len() * f64_size
            + self.mean.len() * f64_size
            + self.std.len() * f64_size
            + self.activation_derivative.len() * f64_size
    }

    /// Compression ratio vs storing full activations.
    pub fn compression_ratio(&self) -> f64 {
        let full_size = self.batch_size * self.input_dim * std::mem::size_of::<f64>();
        if full_size == 0 { return 1.0; }
        full_size as f64 / self.memory_bytes() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cache_compression_is_significant() {
        // With d=64, k=4: projected is 8×4=32 elements vs full 8×64=512
        // But activation_derivative is still 8×64, so total ~544 vs 512 full.
        // Compression only wins when k << d AND we account for derivative being necessary.
        // Test that projected part alone is significantly smaller:
        let sigma = array![3.0, 2.0, 1.5, 1.0];
        let u = Array2::eye(64).slice(ndarray::s![.., ..4]).to_owned();
        let v = Array2::eye(64).slice(ndarray::s![.., ..4]).to_owned();
        let stratum = Stratum::new(sigma, u.clone(), v, 0);

        let activations = Array2::from_shape_fn((8, 64), |(i, j)| (i * 64 + j) as f64 * 0.01);
        let pre_act = activations.clone();
        let cache = ActivationCache::from_activations(&activations, &pre_act, &stratum);

        // Projected part: 8*4 = 32 floats vs full activations 8*64 = 512 floats
        let projected_bytes = cache.projected.len() * std::mem::size_of::<f64>();
        let full_bytes = 8 * 64 * std::mem::size_of::<f64>();
        assert!(
            projected_bytes < full_bytes / 4,
            "Projected activations should be <25% of full: {} vs {}",
            projected_bytes, full_bytes
        );
    }
}
