//! Randomized truncated SVD via subspace iteration (Halko, Martinsson, Tropp 2011).
//!
//! Provides LAPACK-free approximate SVD suitable for initialization and
//! adaptive rank updates in the RSMF framework. Uses power iteration with
//! QR orthogonalization for numerical stability.

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, StandardNormal};

/// Result of a truncated SVD: W ≈ U · diag(σ) · Vᵀ
#[derive(Debug, Clone)]
pub struct TruncatedSvd {
    /// Singular values in descending order, length k.
    pub sigma: Array1<f64>,
    /// Left singular vectors U ∈ R^(m × k).
    pub u: Array2<f64>,
    /// Right singular vectors V ∈ R^(n × k).
    pub v: Array2<f64>,
}

/// Compute a rank-k truncated SVD of matrix `a` using randomized subspace iteration.
///
/// Algorithm (Halko et al., 2011, Algorithm 4.4 with power iteration):
/// 1. Draw random Gaussian test matrix Ω ∈ R^(n × (k+p))
/// 2. Form Y = A · Ω
/// 3. Power iterate: Y ← A·(Aᵀ·Y) with QR at each step for stability
/// 4. Thin QR: Y = Q·R, Q ∈ R^(m × (k+p))
/// 5. Form B = Qᵀ·A ∈ R^((k+p) × n)
/// 6. Exact SVD of small matrix B = Û·Σ·Vᵀ (via eigendecomposition of B·Bᵀ)
/// 7. U = Q·Û, truncate to top k
///
/// # Arguments
/// * `a` - Input matrix of shape (m, n)
/// * `k` - Target rank
/// * `n_oversamples` - Oversampling parameter p (typically 5-10)
/// * `n_power_iters` - Number of power iterations (2-3 is usually sufficient)
///
/// # Panics
/// Panics if k + n_oversamples > min(m, n).
pub fn randomized_svd(
    a: &Array2<f64>,
    k: usize,
    n_oversamples: usize,
    n_power_iters: usize,
) -> TruncatedSvd {
    let m = a.nrows();
    let n = a.ncols();
    let target = k + n_oversamples;
    assert!(
        target <= m.min(n),
        "k + oversamples ({}) must be <= min(m,n) ({})",
        target,
        m.min(n)
    );

    let mut rng = rand::thread_rng();

    // Step 1: Random Gaussian test matrix Ω ∈ R^(n × target)
    let omega: Array2<f64> = Array2::from_shape_fn((n, target), |_| {
        StandardNormal.sample(&mut rng)
    });

    // Step 2: Y = A · Ω
    let mut y = a.dot(&omega);

    // Step 3: Power iteration with QR stabilization
    for _ in 0..n_power_iters {
        // Y = A · (Aᵀ · Y)
        let at_y = a.t().dot(&y);
        y = a.dot(&at_y);
        // QR orthogonalization for numerical stability
        y = qr_q(&y);
    }

    // Step 4: Thin QR factorization Y = Q·R
    let q = qr_q(&y);

    // Step 5: B = Qᵀ · A
    let b = q.t().dot(a);

    // Step 6: Eigendecomposition of B·Bᵀ to get singular values/vectors
    // B·Bᵀ = Û·Σ²·Ûᵀ
    let bbt = b.dot(&b.t());
    let (eigenvalues, u_hat) = symmetric_eigendecomp(&bbt);

    // Singular values = sqrt(eigenvalues), clamp negatives from numerical noise
    let sigma_full: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
        .collect();

    // Sort by descending singular value
    let mut indices: Vec<usize> = (0..sigma_full.len()).collect();
    indices.sort_by(|&i, &j| sigma_full[j].partial_cmp(&sigma_full[i]).unwrap());

    // Truncate to k
    let actual_k = k.min(sigma_full.len());
    let sigma = Array1::from_vec(indices[..actual_k].iter().map(|&i| sigma_full[i]).collect());

    // U = Q · Û[:, sorted_indices[:k]]
    let u_cols: Vec<usize> = indices[..actual_k].to_vec();
    let u_hat_trunc = select_columns(&u_hat, &u_cols);
    let u = q.dot(&u_hat_trunc);

    // V = Bᵀ · Û · Σ⁻¹ (or equivalently, compute from Aᵀ·U·Σ⁻¹)
    // More stable: V_j = (1/σ_j) · Aᵀ · U_j
    let mut v = Array2::zeros((n, actual_k));
    for j in 0..actual_k {
        if sigma[j] > 1e-15 {
            let u_col = u.column(j);
            let av = a.t().dot(&u_col.to_owned().insert_axis(Axis(1)));
            let v_col = av.mapv(|x| x / sigma[j]);
            for i in 0..n {
                v[[i, j]] = v_col[[i, 0]];
            }
        }
    }

    TruncatedSvd { sigma, u, v }
}

/// Compute thin QR Q-factor using modified Gram-Schmidt.
/// Returns Q ∈ R^(m × n) with orthonormal columns.
fn qr_q(a: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut q = a.clone();

    for j in 0..n {
        // Subtract projections onto previous columns
        for i in 0..j {
            let qi = q.column(i).to_owned();
            let dot = q.column(j).dot(&qi);
            for row in 0..m {
                q[[row, j]] -= dot * qi[row];
            }
        }
        // Normalize
        let norm = q.column(j).mapv(|x| x * x).sum().sqrt();
        if norm > 1e-15 {
            for row in 0..m {
                q[[row, j]] /= norm;
            }
        }
    }
    q
}

/// Simple eigendecomposition of a symmetric positive semi-definite matrix
/// via power iteration with deflation. Returns (eigenvalues, eigenvectors)
/// where eigenvectors are columns.
///
/// This is intentionally simple — adequate for the small matrices
/// (k+p)×(k+p) that arise in randomized SVD.
fn symmetric_eigendecomp(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "Matrix must be square");

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Array2::zeros((n, n));
    let mut residual = a.clone();

    for k in 0..n {
        // Power iteration on residual matrix
        let mut v = Array1::from_shape_fn(n, |i| if i == k % n { 1.0 } else { 0.0 });

        for _ in 0..100 {
            let w = residual.dot(&v.to_owned().insert_axis(Axis(1)));
            let w = w.column(0).to_owned();
            let norm = w.mapv(|x| x * x).sum().sqrt();
            if norm < 1e-15 {
                break;
            }
            v = w.mapv(|x| x / norm);
        }

        let mv = residual.dot(&v.to_owned().insert_axis(Axis(1)));
        let lambda = v.dot(&mv.column(0));

        eigenvalues.push(lambda.max(0.0));
        for i in 0..n {
            eigenvectors[[i, k]] = v[i];
        }

        // Deflate: residual -= λ · v · vᵀ
        for i in 0..n {
            for j in 0..n {
                residual[[i, j]] -= lambda * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

/// Select specific columns from a matrix.
fn select_columns(a: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let m = a.nrows();
    let mut out = Array2::zeros((m, cols.len()));
    for (j, &col) in cols.iter().enumerate() {
        for i in 0..m {
            out[[i, j]] = a[[i, col]];
        }
    }
    out
}

/// Compute nuclear norm (sum of singular values) from a spectrum.
pub fn nuclear_norm(sigma: &Array1<f64>) -> f64 {
    sigma.sum()
}

/// Compute spectral norm (largest singular value) from a spectrum.
pub fn spectral_norm(sigma: &Array1<f64>) -> f64 {
    sigma.iter().cloned().fold(0.0_f64, f64::max)
}

/// Stable rank = (nuclear_norm / spectral_norm)².
/// Measures effective dimensionality; higher = more uniform spectrum.
pub fn stable_rank(sigma: &Array1<f64>) -> f64 {
    let sn = spectral_norm(sigma);
    if sn < 1e-15 {
        return 0.0;
    }
    let nn = nuclear_norm(sigma);
    (nn / sn).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn svd_reconstructs_low_rank_matrix() {
        // Create a known rank-2 matrix using orthonormal U, V
        // U ∈ R^(4×2) orthonormal columns via Gram-Schmidt
        let u = Array2::from_shape_vec((4, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ]).unwrap();
        let sigma_true = array![3.0, 1.0];
        // V ∈ R^(3×2) orthonormal columns
        let v = Array2::from_shape_vec((3, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]).unwrap();

        // Build A = U·diag(σ)·Vᵀ (exact rank-2 with known singular values)
        let mut u_scaled = u.clone();
        for j in 0..2 {
            for i in 0..4 {
                u_scaled[[i, j]] *= sigma_true[j];
            }
        }
        let a = u_scaled.dot(&v.t());

        let svd = randomized_svd(&a, 2, 1, 3);

        // Check singular values are close
        assert_relative_eq!(svd.sigma[0], sigma_true[0], epsilon = 0.1);
        assert_relative_eq!(svd.sigma[1], sigma_true[1], epsilon = 0.1);

        // Check reconstruction error
        let mut recon: Array2<f64> = Array2::zeros((4, 3));
        for j in 0..2 {
            for i in 0..4 {
                for l in 0..3 {
                    recon[[i, l]] += svd.u[[i, j]] * svd.sigma[j] * svd.v[[l, j]];
                }
            }
        }
        let err = (&a - &recon).mapv(|x| x * x).sum().sqrt();
        assert!(err < 0.5, "Reconstruction error too large: {}", err);
    }

    #[test]
    fn stable_rank_identity() {
        // Identity spectrum: all equal → stable rank = k
        let sigma = Array1::from_vec(vec![1.0; 5]);
        assert_relative_eq!(stable_rank(&sigma), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn stable_rank_single_direction() {
        // One dominant direction → stable rank ≈ 1
        let sigma = Array1::from_vec(vec![10.0, 0.01, 0.01, 0.01]);
        let sr = stable_rank(&sigma);
        assert!(sr < 2.0, "Expected ~1, got {}", sr);
    }
}
