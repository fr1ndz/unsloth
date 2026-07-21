use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig, Stratum};

/// Local resonance operator for single-stratum updates.
///
/// Implements the core RSMF update rule:
///   Ψₗ ← argmin_Ψ ‖A·Ψ - T‖² + λ·Ω(Ψ) + μ·Γ(Ψ, Ψₗ₋₁, Ψₗ₊₁)
///
/// This operates entirely within O(k·d) memory — no full weight matrix needed.
pub struct LocalResonance<'a> {
    config: &'a SpectralConfig,
}

impl<'a> LocalResonance<'a> {
    pub fn new(config: &'a SpectralConfig) -> Self {
        Self { config }
    }

    /// Perform one local stratum update step.
    ///
    /// # Arguments
    /// * `tensor` — mutable resonant tensor to update
    /// * `activations` — input activations A ∈ R^(batch × d_in)
    /// * `target` — resonance target T ∈ R^(batch × d_out)
    /// * `prev_spectrum` — singular values of previous layer (for Γ coupling)
    /// * `next_spectrum` — singular values of next layer (for Γ coupling)
    ///
    /// # Returns
    /// Residual norm after update (for convergence monitoring).
    pub fn update_step(
        &self,
        tensor: &mut ResonantTensor,
        activations: &Array2<f64>,
        target: &Array2<f64>,
        prev_spectrum: Option<&Array1<f64>>,
        next_spectrum: Option<&Array1<f64>>,
    ) -> f64 {
        let k = tensor.stratum.rank();
        let batch = activations.nrows();

        // Project activations onto stratum subspace: A_proj = A · U ∈ R^(batch × k)
        let a_proj = activations.dot(&tensor.stratum.u_basis);

        // Project target onto output subspace: T_proj = T · V ∈ R^(batch × k)
        let t_proj = target.dot(&tensor.stratum.v_basis);

        // Compute gradient in spectral space:
        // ∂/∂σᵢ = 2·Σⱼ (A_proj[j,i]·σᵢ - T_proj[j,i]) · A_proj[j,i]
        let mut sigma_grad = Array1::zeros(k);
        for i in 0..k {
            let mut grad_i = 0.0;
            for j in 0..batch {
                let residual = a_proj[[j, i]] * tensor.stratum.sigma[i] - t_proj[[j, i]];
                grad_i += 2.0 * residual * a_proj[[j, i]];
            }
            sigma_grad[i] = grad_i / batch as f64;
        }

        // Add spectral regularization gradient: ∂Ω/∂σᵢ = -2λ/σᵢ³ (soft thresholding)
        for i in 0..k {
            let s = tensor.stratum.sigma[i];
            if s.abs() > self.config.epsilon {
                sigma_grad[i] += self.config.lambda_spectral * (-2.0 / (s * s * s));
            }
        }

        // Add inter-stratum coupling gradient (simplified Γ)
        if let Some(prev) = prev_spectrum {
            add_coupling_gradient(&mut sigma_grad, &tensor.stratum.sigma, prev, self.config.mu_coupling, true);
        }
        if let Some(next) = next_spectrum {
            add_coupling_gradient(&mut sigma_grad, &tensor.stratum.sigma, next, self.config.mu_coupling, false);
        }

        // Apply gradient descent in spectral space
        let lr = self.config.learning_rate;
        let mut residual_norm = 0.0;
        for i in 0..k {
            let delta = lr * sigma_grad[i];
            tensor.stratum.sigma[i] -= delta;
            // Ensure non-negative singular values
            tensor.stratum.sigma[i] = tensor.stratum.sigma[i].max(self.config.epsilon);
            residual_norm += delta * delta;
        }

        // Update metadata
        tensor.is_updated = true;
        tensor.update_energy(activations);
        tensor.advance_phase(1.0, self.config);

        residual_norm.sqrt()
    }

    /// Run full inner optimization loop until convergence or max iterations.
    pub fn optimize(
        &self,
        tensor: &mut ResonantTensor,
        activations: &Array2<f64>,
        target: &Array2<f64>,
        prev_spectrum: Option<&Array1<f64>>,
        next_spectrum: Option<&Array1<f64>>,
    ) -> usize {
        let mut iters = 0;
        for _ in 0..self.config.max_inner_iters {
            let residual = self.update_step(tensor, activations, target, prev_spectrum, next_spectrum);
            iters += 1;
            if residual < self.config.convergence_tol {
                break;
            }
        }
        iters
    }
}

/// Add coupling gradient contribution from adjacent stratum spectrum.
/// Encourages spectral alignment: penalizes large differences in σ distributions.
fn add_coupling_gradient(
    grad: &mut Array1<f64>,
    current_sigma: &Array1<f64>,
    adjacent_sigma: &Array1<f64>,
    mu: f64,
    is_prev: bool,
) {
    let k = current_sigma.len().min(adjacent_sigma.len());
    let sign = if is_prev { -1.0 } else { 1.0 };
    for i in 0..k {
        let diff = current_sigma[i] - adjacent_sigma[i];
        grad[i] += mu * sign * 2.0 * diff;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn update_reduces_residual() {
        let config = SpectralConfig {
            top_k: 2,
            lambda_spectral: 0.0,
            mu_coupling: 0.0,
            epsilon: 1e-8,
            learning_rate: 0.1,
            coherence_threshold: 0.3,
            max_inner_iters: 50,
            convergence_tol: 1e-8,
        };

        let sigma = array![1.0, 1.0];
        let u = Array2::eye(2);
        let v = Array2::eye(2);
        let stratum = Stratum::new(sigma, u, v, 0);
        let mut tensor = ResonantTensor::from_stratum(stratum);

        // Simple target: identity mapping
        let activations = Array2::eye(2);
        let target = Array2::eye(2) * 2.0; // Want W ≈ 2I

        let lr = LocalResonance::new(&config);
        let initial_residual = lr.update_step(&mut tensor, &activations, &target, None, None);

        // After several steps, residual should decrease
        for _ in 0..10 {
            lr.update_step(&mut tensor, &activations, &target, None, None);
        }
        let final_residual = lr.update_step(&mut tensor, &activations, &target, None, None);
        assert!(final_residual < initial_residual);
    }
}
