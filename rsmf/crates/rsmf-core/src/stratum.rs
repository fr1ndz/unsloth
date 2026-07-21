use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// A single stratum in the stratified weight manifold.
///
/// Represents the spectral decomposition of a weight matrix W ∈ R^(m×n):
///   W ≈ U · diag(σ) · Vᵀ
///
/// Only top-k components are stored for memory efficiency.
/// The full weight matrix can be reconstructed on demand and freed immediately.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stratum {
    /// Top-k singular values (descending order).
    pub sigma: Array1<f64>,
    /// Left singular vectors U ∈ R^(m × k), columns are basis vectors.
    pub u_basis: Array2<f64>,
    /// Right singular vectors V ∈ R^(n × k), columns are basis vectors.
    pub v_basis: Array2<f64>,
    /// Layer index this stratum belongs to.
    pub layer_id: usize,
    /// Original weight matrix dimensions (m, n).
    pub shape: (usize, usize),
}

impl Stratum {
    /// Create a new stratum from pre-computed SVD components.
    ///
    /// # Panics
    /// Panics if dimensions are inconsistent.
    pub fn new(
        sigma: Array1<f64>,
        u_basis: Array2<f64>,
        v_basis: Array2<f64>,
        layer_id: usize,
    ) -> Self {
        let k = sigma.len();
        assert_eq!(u_basis.ncols(), k, "U columns must match sigma length");
        assert_eq!(v_basis.ncols(), k, "V columns must match sigma length");
        let shape = (u_basis.nrows(), v_basis.nrows());
        Self { sigma, u_basis, v_basis, layer_id, shape }
    }

    /// Reconstruct the full weight matrix W = U · diag(σ) · Vᵀ.
    ///
    /// ⚠️ MEMORY WARNING: This allocates O(m×n). Call only when needed,
    /// and drop the result as soon as the update is applied.
    pub fn reconstruct(&self) -> Array2<f64> {
        // W = U · diag(σ) · Vᵀ
        // Step 1: Scale U columns by σ → U_scaled
        let mut u_scaled = self.u_basis.clone();
        for (j, &s) in self.sigma.iter().enumerate() {
            for i in 0..u_scaled.nrows() {
                u_scaled[[i, j]] *= s;
            }
        }
        // Step 2: U_scaled · Vᵀ
        u_scaled.dot(&self.v_basis.t())
    }

    /// Compute the Frobenius norm of the reconstructed weight matrix
    /// WITHOUT materializing it. Uses ‖W‖_F = √(Σ σᵢ²) for exact SVD.
    pub fn frobenius_norm(&self) -> f64 {
        self.sigma.mapv(|s| s * s).sum().sqrt()
    }

    /// Number of retained singular components.
    pub fn rank(&self) -> usize {
        self.sigma.len()
    }

    /// Spectral energy ratio: how much variance is captured by top-k.
    /// Returns value in [0, 1]. Requires knowledge of total energy.
    pub fn energy_ratio(&self, total_energy: f64) -> f64 {
        if total_energy < 1e-15 { return 0.0; }
        let captured = self.sigma.mapv(|s| s * s).sum();
        captured / total_energy
    }

    /// Memory footprint in bytes (FP64 storage).
    pub fn memory_bytes(&self) -> usize {
        let elem = std::mem::size_of::<f64>();
        self.sigma.len() * elem                          // sigma
        + self.u_basis.len() * elem                      // U
        + self.v_basis.len() * elem                      // V
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn reconstruct_identity() {
        // Identity-like: σ=[1,1], U=I₂, V=I₂
        let sigma = array![1.0, 1.0];
        let u = Array2::eye(2);
        let v = Array2::eye(2);
        let s = Stratum::new(sigma, u, v, 0);
        let w = s.reconstruct();
        let expected: Array2<f64> = Array2::eye(2);
        assert!((w - expected).mapv(|x| x.abs()).sum() < 1e-10);
    }

    #[test]
    fn frobenius_without_reconstruction() {
        let sigma = array![3.0, 4.0];
        let u = Array2::eye(2);
        let v = Array2::eye(2);
        let s = Stratum::new(sigma, u, v, 0);
        // ‖W‖_F = √(9+16) = 5
        assert!((s.frobenius_norm() - 5.0).abs() < 1e-10);
    }
}
