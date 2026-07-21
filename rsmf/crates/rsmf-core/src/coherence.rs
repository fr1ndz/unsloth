use ndarray::Array1;
use crate::tensor::ResonantTensor;

/// Measures geometric alignment between adjacent strata.
///
/// Γ(Ψₗ, Ψₗ₋₁, Ψ₊₁) quantifies how well the spectral subspaces
/// of neighboring layers align. Low coherence indicates potential
/// information bottleneck or gradient vanishing zone.
#[derive(Debug, Clone, Copy)]
pub struct CoherenceMetric {
    /// Cosine similarity between singular spectra of adjacent layers.
    pub spectral_alignment: f64,
    /// Subspace overlap (principal angles) between U bases.
    pub subspace_overlap: f64,
    /// Combined coherence score ∈ [0, 1].
    pub score: f64,
}

impl CoherenceMetric {
    /// Compute coherence between two adjacent resonant tensors.
    pub fn between(a: &ResonantTensor, b: &ResonantTensor) -> Self {
        let spectral_alignment = spectral_cosine(&a.stratum.sigma, &b.stratum.sigma);
        let subspace_overlap = subspace_principal_angle(&a.stratum.u_basis, &b.stratum.u_basis);
        let score = 0.6 * spectral_alignment + 0.4 * subspace_overlap;
        Self { spectral_alignment, subspace_overlap, score }
    }

    /// Check if coherence is above safety threshold.
    pub fn is_safe(&self, threshold: f64) -> bool {
        self.score >= threshold
    }
}

/// Cosine similarity between two singular value vectors.
/// Pads shorter vector with zeros for comparison.
fn spectral_cosine(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let max_len = a.len().max(b.len());
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..max_len {
        let va = if i < a.len() { a[i] } else { 0.0 };
        let vb = if i < b.len() { b[i] } else { 0.0 };
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-15 { 0.0 } else { dot / denom }
}

/// Approximate subspace overlap via trace(Uₐᵀ · U_b · U_bᵀ · Uₐ) / k.
/// Returns value in [0, 1] where 1 = identical subspaces.
fn subspace_principal_angle(u_a: &ndarray::Array2<f64>, u_b: &ndarray::Array2<f64>) -> f64 {
    let k = u_a.ncols().min(u_b.ncols());
    if k == 0 { return 0.0; }
    // C = Uₐᵀ · U_b ∈ R^(kₐ × k_b)
    let c = u_a.t().dot(u_b);
    // Overlap = ‖C‖_F² / k
    let frob_sq = c.mapv(|x| x * x).sum();
    (frob_sq / k as f64).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn identical_spectra_have_unit_coherence() {
        let sigma = array![3.0, 2.0, 1.0];
        let cos = spectral_cosine(&sigma, &sigma);
        assert!((cos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn orthogonal_spectra_have_zero_coherence() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        let cos = spectral_cosine(&a, &b);
        assert!(cos.abs() < 1e-10);
    }

    #[test]
    fn identical_subspaces_have_unit_overlap() {
        let u = Array2::eye(4);
        let overlap = subspace_principal_angle(&u, &u);
        assert!((overlap - 1.0).abs() < 1e-10);
    }
}
