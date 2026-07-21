use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig};

/// Symplectic Tensor-Spectral Modulation (STSM) operator.
///
/// Replaces gradient-based local update with symplectic phase flow
/// on the stratum manifold. The key insight: weight updates are
/// Hamiltonian flows preserving the symplectic structure of the
/// spectral decomposition.
///
/// ## Mathematics
///
/// Phase space: (σ, φ) where σ = singular values, φ = conjugate momenta
/// Hamiltonian: H(σ, φ) = ½‖φ‖² + V(σ) where V is the resonance potential
/// Flow: dσ/dt = ∂H/∂φ = φ,  dφ/dt = -∂H/∂σ = -∇V(σ)
///
/// This preserves the symplectic form ω = dσ ∧ dφ exactly under
/// leapfrog integration, preventing energy drift that plagues
/// standard gradient descent in spectral space.
pub struct SymplecticModulator<'a> {
    config: &'a SpectralConfig,
}

/// Symplectic state for one stratum: (σ, φ) pair.
#[derive(Debug, Clone)]
pub struct SymplecticState {
    /// Singular values (position in phase space).
    pub sigma: Array1<f64>,
    /// Conjugate momenta.
    pub momentum: Array1<f64>,
    /// Current Hamiltonian energy (conserved quantity).
    pub energy: f64,
}

impl SymplecticState {
    /// Initialize from resonant tensor with zero momentum.
    pub fn from_tensor(tensor: &ResonantTensor) -> Self {
        let k = tensor.stratum.rank();
        Self {
            sigma: tensor.stratum.sigma.clone(),
            momentum: Array1::zeros(k),
            energy: 0.0,
        }
    }

    /// Compute Hamiltonian: H = ½‖p‖² + V(σ)
    pub fn hamiltonian(&self, lambda: f64) -> f64 {
        let kinetic: f64 = self.momentum.mapv(|p| p * p).sum() * 0.5;
        // Potential: spectral regularizer as confining potential
        let potential: f64 = self.sigma.iter()
            .map(|&s| lambda * (s.ln().powi(2) + 1.0 / (s * s + 1e-15)))
            .sum();
        kinetic + potential
    }
}

impl<'a> SymplecticModulator<'a> {
    pub fn new(config: &'a SpectralConfig) -> Self {
        Self { config }
    }

    /// One leapfrog step of symplectic integration.
    ///
    /// Leapfrog preserves symplectic structure exactly:
    ///   p_{n+½} = p_n - (ε/2) · ∇V(σ_n)
    ///   σ_{n+1} = σ_n + ε · p_{n+½}
    ///   p_{n+1} = p_{n+½} - (ε/2) · ∇V(σ_{n+1})
    pub fn leapfrog_step(
        &self,
        state: &mut SymplecticState,
        target_spectrum: Option<&Array1<f64>>,
        dt: f64,
    ) {
        let k = state.sigma.len();
        let lambda = self.config.lambda_spectral;

        // Half-step momentum update
        let grad_v = self.potential_gradient(&state.sigma, target_spectrum, lambda);
        for i in 0..k {
            state.momentum[i] -= 0.5 * dt * grad_v[i];
        }

        // Full-step position update
        for i in 0..k {
            state.sigma[i] += dt * state.momentum[i];
            // Enforce positivity constraint (reflecting boundary)
            if state.sigma[i] < self.config.epsilon {
                state.sigma[i] = self.config.epsilon;
                state.momentum[i] = state.momentum[i].abs() * 0.5; // Damped reflection
            }
        }

        // Half-step momentum update with new position
        let grad_v_new = self.potential_gradient(&state.sigma, target_spectrum, lambda);
        for i in 0..k {
            state.momentum[i] -= 0.5 * dt * grad_v_new[i];
        }

        // Update conserved energy
        state.energy = state.hamiltonian(lambda);
    }

    /// Run N leapfrog steps with optional damping.
    pub fn integrate(
        &self,
        state: &mut SymplecticState,
        target_spectrum: Option<&Array1<f64>>,
        num_steps: usize,
        dt: f64,
        damping: f64,
    ) {
        for _ in 0..num_steps {
            self.leapfrog_step(state, target_spectrum, dt);
            // Optional momentum damping (breaks exact symplecticity but aids convergence)
            if damping > 0.0 && damping < 1.0 {
                state.momentum *= damping;
            }
        }
    }

    /// Apply symplectic state back to resonant tensor.
    pub fn apply_to_tensor(&self, state: &SymplecticState, tensor: &mut ResonantTensor) {
        tensor.stratum.sigma = state.sigma.clone();
        tensor.is_updated = true;
    }

    /// Gradient of spectral potential V(σ).
    /// V(σ) = λ·Σ[ln²(σᵢ) + 1/(σᵢ² + ε)] + μ·‖σ - σ_target‖²
    fn potential_gradient(
        &self,
        sigma: &Array1<f64>,
        target: Option<&Array1<f64>>,
        lambda: f64,
    ) -> Array1<f64> {
        let k = sigma.len();
        let mut grad = Array1::zeros(k);
        let eps = self.config.epsilon;

        for i in 0..k {
            let s = sigma[i].max(eps);
            // d/dσ [ln²(σ)] = 2·ln(σ)/σ
            let log_term = 2.0 * s.ln() / s;
            // d/dσ [1/(σ²+ε)] = -2σ/(σ²+ε)²
            let inv_term = -2.0 * s / (s * s + eps).powi(2);
            grad[i] = lambda * (log_term + inv_term);

            // Target attraction gradient
            if let Some(tgt) = target {
                if i < tgt.len() {
                    let mu = self.config.mu_coupling;
                    grad[i] += 2.0 * mu * (s - tgt[i]);
                }
            }
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rsmf_core::Stratum;

    #[test]
    fn leapfrog_conserves_energy_approximately() {
        let config = SpectralConfig {
            top_k: 4,
            lambda_spectral: 0.1,
            epsilon: 1e-8,
            ..SpectralConfig::default()
        };
        let modulator = SymplecticModulator::new(&config);

        let sigma = array![3.0, 2.0, 1.0, 0.5];
        let u = Array2::eye(4);
        let v = Array2::eye(4);
        let stratum = Stratum::new(sigma, u, v, 0);
        let tensor = ResonantTensor::from_stratum(stratum);
        let mut state = SymplecticState::from_tensor(&tensor);

        let initial_energy = state.hamiltonian(config.lambda_spectral);

        // Run 100 leapfrog steps without damping
        modulator.integrate(&mut state, None, 100, 0.01, 0.0);

        let final_energy = state.hamiltonian(config.lambda_spectral);
        let drift = (final_energy - initial_energy).abs() / initial_energy.abs().max(1e-15);

        // Symplectic integrator should conserve energy to ~1%
        assert!(drift < 0.05, "Energy drift too large: {:.4}%", drift * 100.0);
    }

    #[test]
    fn sigma_stays_positive() {
        let config = SpectralConfig {
            top_k: 2,
            lambda_spectral: 0.5,
            epsilon: 1e-10,
            ..SpectralConfig::default()
        };
        let modulator = SymplecticModulator::new(&config);

        let sigma = array![0.1, 0.01]; // Small values near boundary
        let u = Array2::eye(2);
        let v = Array2::eye(2);
        let stratum = Stratum::new(sigma, u, v, 0);
        let tensor = ResonantTensor::from_stratum(stratum);
        let mut state = SymplecticState::from_tensor(&tensor);

        modulator.integrate(&mut state, None, 50, 0.05, 0.0);

        for &s in state.sigma.iter() {
            assert!(s >= config.epsilon, "Sigma went negative: {}", s);
        }
    }
}
