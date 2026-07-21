use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig};

/// Resonant backward channel Φ — replaces backpropagation.
///
/// Instead of δₗ = (Wₗ₊₁ᵀ · δₗ₊₁) ⊙ f'(zₗ), computes:
///   δₗ = Φ(δₗ₊₁, Ψₗ₊₁, Ψₗ) · κₗ
///
/// where Φ is the resonant transition functor and κₗ is the
/// adaptive coupling coefficient based on spectral coherence.
///
/// Memory: O(d) per layer instead of O(d²) for gradient matrices.
pub struct ResonantBackward<'a> {
    config: &'a SpectralConfig,
}

impl<'a> ResonantBackward<'a> {
    pub fn new(config: &'a SpectralConfig) -> Self {
        Self { config }
    }

    /// Compute resonant backward signal for one layer transition.
    ///
    /// # Arguments
    /// * `delta_next` — resonance signal from layer l+1, shape (batch, d_out)
    /// * `current` — resonant tensor of current layer l
    /// * `next` — resonant tensor of next layer l+1
    /// * `activation_derivative` — element-wise f'(zₗ), shape (batch, d_in)
    ///
    /// # Returns
    /// Resonance signal δₗ for current layer, shape (batch, d_in)
    pub fn propagate(
        &self,
        delta_next: &Array2<f64>,
        current: &ResonantTensor,
        next: &ResonantTensor,
        activation_derivative: &Array2<f64>,
    ) -> Array2<f64> {
        let batch = delta_next.nrows();
        let d_in = current.stratum.shape.0;

        // Step 1: Compute coupling coefficient κₗ
        let kappa = self.compute_coupling_coefficient(current, next);

        // Step 2: Apply resonant transition functor Φ
        // Φ(δ, Ψ', Ψ) = Uₗ · diag(σₗ/(σₗ+ε)) · Vₗᵀ · δ · (U'ₗ · V'ₗᵀ)
        // Simplified: project δ through current stratum's spectral filter

        // Project delta_next through next layer's V basis: δ_proj = δ · V_next ∈ R^(batch × k)
        let k_next = next.stratum.rank();
        let delta_proj = delta_next.dot(&next.stratum.v_basis);

        // Scale by spectral filter: δ_filtered[i] = δ_proj[i] · σ_next[i] / (σ_next[i] + ε)
        let mut delta_filtered = delta_proj.clone();
        for i in 0..k_next {
            let s = next.stratum.sigma[i];
            let scale = s / (s + self.config.epsilon);
            for j in 0..batch {
                delta_filtered[[j, i]] *= scale;
            }
        }

        // Map back through next layer's U basis: δ_mapped = δ_filtered · U_nextᵀ ∈ R^(batch × d_out)
        let delta_mapped = delta_filtered.dot(&next.stratum.u_basis.t());

        // Step 3: Cross-layer projection via current stratum
        // Project through current U: δ_curr_proj = δ_mapped · U_curr ∈ R^(batch × k_curr)
        let k_curr = current.stratum.rank();
        let d_out = delta_mapped.ncols();

        // Handle dimension mismatch via truncated projection
        let min_d = d_out.min(d_in);
        let mut result = Array2::zeros((batch, d_in));

        for b in 0..batch {
            for i in 0..min_d.min(k_curr) {
                let mut val = 0.0;
                for j in 0..d_out.min(current.stratum.u_basis.nrows()) {
                    val += delta_mapped[[b, j]] * current.stratum.u_basis[[j, i]];
                }
                // Apply spectral filter of current layer
                let s = current.stratum.sigma[i];
                val *= s / (s + self.config.epsilon);
                // Map to output dimension via V basis
                for o in 0..d_in.min(current.stratum.v_basis.nrows()) {
                    result[[b, o]] += val * current.stratum.v_basis[[o, i]];
                }
            }
        }

        // Step 4: Apply coupling coefficient and activation derivative
        for b in 0..batch {
            for i in 0..d_in.min(activation_derivative.ncols()) {
                result[[b, i]] *= kappa * activation_derivative[[b, i]];
            }
        }

        result
    }

    /// Compute adaptive coupling coefficient κₗ based on spectral coherence.
    /// κ = softmax(cos_angle(spec(Ψₗ), spec(Ψₗ₊₁)))
    pub fn compute_coupling_coefficient(
        &self,
        current: &ResonantTensor,
        next: &ResonantTensor,
    ) -> f64 {
        let cos_sim = spectral_cosine_simple(&current.stratum.sigma, &next.stratum.sigma);
        // Map [-1, 1] → [0.1, 2.0] via sigmoid-like transform
        let kappa = 0.1 + 1.9 / (1.0 + (-4.0 * (cos_sim - 0.5)).exp());
        kappa
    }
}

/// Simple cosine similarity between two spectra (padded with zeros).
fn spectral_cosine_simple(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let max_len = a.len().max(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..max_len {
        let va = if i < a.len() { a[i] } else { 0.0 };
        let vb = if i < b.len() { b[i] } else { 0.0 };
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-15 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rsmf_core::Stratum;

    #[test]
    fn coupling_coefficient_range() {
        let config = SpectralConfig::default();
        let rb = ResonantBackward::new(&config);

        let make_tensor = |sigma_vals: &[f64]| {
            let k = sigma_vals.len();
            let sigma = Array1::from_vec(sigma_vals.to_vec());
            let u = Array2::eye(k);
            let v = Array2::eye(k);
            ResonantTensor::from_stratum(Stratum::new(sigma, u, v, 0))
        };

        let identical_a = make_tensor(&[3.0, 2.0, 1.0]);
        let identical_b = make_tensor(&[3.0, 2.0, 1.0]);
        let kappa_same = rb.compute_coupling_coefficient(&identical_a, &identical_b);
        assert!(kappa_same > 1.0, "Identical spectra should have high coupling: {}", kappa_same);

        let different = make_tensor(&[1.0, 0.5, 0.1]);
        let kappa_diff = rb.compute_coupling_coefficient(&identical_a, &different);
        assert!(kappa_diff < kappa_same, "Different spectra should have lower coupling");
    }
}
