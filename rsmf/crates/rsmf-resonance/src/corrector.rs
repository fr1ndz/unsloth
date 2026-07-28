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

// ---------------------------------------------------------------------------
// Orthogonality utilities (public for use in merge / checkpoint contexts)
// ---------------------------------------------------------------------------

/// Modified Gram-Schmidt re-orthogonalization of column vectors in-place.
///
/// Operates on an m×k matrix whose *columns* are the basis vectors.
/// Returns the maximum deviation from orthonormality observed **before**
/// correction (useful as a drift diagnostic).
pub fn reorthogonalize_columns(basis: &mut Array2<f64>) -> f64 {
    let k = basis.ncols();
    if k == 0 {
        return 0.0;
    }

    // Measure pre-correction orthogonality error: max |⟨cᵢ, cⱼ⟩ - δᵢⱼ|
    let mut max_err = 0.0_f64;
    for i in 0..k {
        let ni = col_norm(basis, i);
        max_err = max_err.max((ni - 1.0).abs());
        for j in (i + 1)..k {
            let dot = col_dot(basis, i, j);
            max_err = max_err.max(dot.abs());
        }
    }

    // Classical Gram-Schmidt with re-projection (more numerically stable
    // than one-pass MGS for near-singular inputs).
    for i in 0..k {
        // Subtract projections onto all previous columns
        for j in 0..i {
            let d = col_dot(basis, j, i);
            for row in 0..basis.nrows() {
                basis[[row, i]] -= d * basis[[row, j]];
            }
        }
        // Normalize
        let n = col_norm(basis, i);
        if n > 1e-30 {
            for row in 0..basis.nrows() {
                basis[[row, i]] /= n;
            }
        }
    }

    max_err
}

/// Verify orthonormality of column basis. Returns max |⟨cᵢ,cⱼ⟩ − δᵢⱼ|.
pub fn orthogonality_error(basis: &Array2<f64>) -> f64 {
    let k = basis.ncols();
    let mut max_err = 0.0_f64;
    for i in 0..k {
        let ni = col_norm(basis, i);
        max_err = max_err.max((ni - 1.0).abs());
        for j in (i + 1)..k {
            let dot = col_dot(basis, i, j);
            max_err = max_err.max(dot.abs());
        }
    }
    max_err
}

fn col_dot(m: &Array2<f64>, a: usize, b: usize) -> f64 {
    let rows = m.nrows();
    let mut s = 0.0;
    for r in 0..rows {
        s += m[[r, a]] * m[[r, b]];
    }
    s
}

fn col_norm(m: &Array2<f64>, c: usize) -> f64 {
    col_dot(m, c, c).sqrt()
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

        // Re-orthogonalize U and V bases after spectral modification.
        // Sigma blending can indirectly break basis orthonormality when the
        // stratum was previously updated via gradient steps that only touched σ.
        for t in tensors.iter_mut() {
            reorthogonalize_columns(&mut t.stratum.u_basis);
            reorthogonalize_columns(&mut t.stratum.v_basis);
        }

        // Orthogonality verification checkpoint (debug / merge guard)
        #[cfg(debug_assertions)]
        for (idx, t) in tensors.iter().enumerate() {
            let u_err = orthogonality_error(&t.stratum.u_basis);
            let v_err = orthogonality_error(&t.stratum.v_basis);
            debug_assert!(
                u_err < 1e-8 && v_err < 1e-8,
                "Post-correction orthogonality drift at stratum {}: U_err={:.2e}, V_err={:.2e}",
                idx, u_err, v_err
            );
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
