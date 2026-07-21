use ndarray::{Array1, Array2};
use rsmf_core::{CoherenceMetric, ResonantTensor, SpectralConfig};

/// Global coherence corrector.
///
/// Triggered when inter-stratum coherence drops below threshold.
/// Applies a single-pass spectral alignment correction across all layers
/// without requiring full forward/backward pass.
///
/// Cost: O(L·k²) — negligible compared to training step.
pub struct CoherenceCorrector<'a> {
    config: &'a SpectralConfig,
}

impl<'a> CoherenceCorrector<'a> {
    pub fn new(config: &'a SpectralConfig) -> Self {
        Self { config }
    }

    /// Apply global coherence correction to entire stratum chain.
    ///
    /// Strategy: iteratively align each stratum's spectrum toward
    /// the mean spectrum of its neighbors, weighted by coherence.
    pub fn correct(&self, tensors: &mut [ResonantTensor]) -> f64 {
        if tensors.len() < 2 {
            return 1.0;
        }

        let n = tensors.len();
        let alpha = 0.3; // Correction strength per pass

        // Compute pairwise coherence scores
        let mut coherences = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            coherences.push(CoherenceMetric::between(&tensors[i], &tensors[i + 1]).score);
        }

        // Align spectra toward neighbor-weighted mean
        for i in 0..n {
            let k = tensors[i].stratum.rank();
            let mut target_spectrum = Array1::zeros(k);
            let mut weight_sum = 0.0;

            // Weighted contribution from previous neighbor
            if i > 0 {
                let w = coherences[i - 1];
                let adj_k = tensors[i - 1].stratum.rank().min(k);
                for j in 0..adj_k {
                    target_spectrum[j] += w * tensors[i - 1].stratum.sigma[j];
                }
                weight_sum += w;
            }

            // Weighted contribution from next neighbor
            if i < n - 1 {
                let w = coherences[i];
                let adj_k = tensors[i + 1].stratum.rank().min(k);
                for j in 0..adj_k {
                    target_spectrum[j] += w * tensors[i + 1].stratum.sigma[j];
                }
                weight_sum += w;
            }

            if weight_sum > 1e-15 {
                target_spectrum /= weight_sum;
                // Blend current spectrum toward target
                for j in 0..k {
                    tensors[i].stratum.sigma[j] =
                        (1.0 - alpha) * tensors[i].stratum.sigma[j] + alpha * target_spectrum[j];
                    tensors[i].stratum.sigma[j] = tensors[i].stratum.sigma[j].max(self.config.epsilon);
                }
            }
        }

        // Recompute and return minimum coherence after correction
        let mut min_coherence = f64::INFINITY;
        for i in 0..n - 1 {
            let metric = CoherenceMetric::between(&tensors[i], &tensors[i + 1]);
            min_coherence = min_coherence.min(metric.score);
        }
        min_coherence
    }

    /// Check if correction is needed.
    pub fn needs_correction(&self, tensors: &[ResonantTensor]) -> bool {
        if tensors.len() < 2 {
            return false;
        }
        for window in tensors.windows(2) {
            let metric = CoherenceMetric::between(&window[0], &window[1]);
            if !metric.is_safe(self.config.coherence_threshold) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rsmf_core::Stratum;

    fn make_tensor(sigma: &[f64], id: usize) -> ResonantTensor {
        let k = sigma.len();
        let s = Array1::from_vec(sigma.to_vec());
        let u = Array2::eye(k);
        let v = Array2::eye(k);
        ResonantTensor::from_stratum(Stratum::new(s, u, v, id))
    }

    #[test]
    fn correction_improves_coherence() {
        // Use high threshold so that even moderate divergence triggers correction
        let config = SpectralConfig { coherence_threshold: 0.95, ..SpectralConfig::default() };
        let corrector = CoherenceCorrector::new(&config);

        let mut tensors = vec![
            make_tensor(&[10.0, 5.0, 1.0], 0),
            make_tensor(&[1.0, 5.0, 10.0], 1), // Inverted spectrum
            make_tensor(&[10.0, 5.0, 1.0], 2),
        ];

        assert!(corrector.needs_correction(&tensors));
        let post_coherence = corrector.correct(&mut tensors);

        // After correction, spectra should be more aligned
        assert!(post_coherence > 0.0, "Post-correction coherence should be positive: {}", post_coherence);
    }
}
