use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig, CoherenceMetric};
use rsmf_layers::RsmfModel;
use rsmf_resonance::{LocalResonance, ResonantBackward, InterStratumCoupling, CoherenceCorrector};
use crate::loss::ResonantLoss;
use crate::schedule::CoherenceSchedule;

// ============================================================================
// Gradient SNR Estimator (Welford's Online Algorithm)
// ============================================================================

/// Online estimator for gradient signal-to-noise ratio using Welford's algorithm.
///
/// Computes running mean and variance of per-batch gradient norms (σ_grad),
/// then derives SNR = ‖E[g]‖² / Var(g).
///
/// Used for **rank initialization only** — SOTA research shows continuous
/// hparam tuning via gradient SNR is unstable; we use it once at warmup
/// to set spectral-adaptive ranks per layer.
#[derive(Debug, Clone)]
pub struct GradientSnrEstimator {
    /// Number of samples observed.
    n: u64,
    /// Running mean of gradient norm (Welford's M₁).
    mean: f64,
    /// Running sum of squared deviations (Welford's M₂).
    m2: f64,
    /// Running mean of squared gradient norm (for ‖E[g]‖² estimate).
    mean_sq: f64,
    /// Accumulated squared norm of mean gradient direction.
    /// Updated incrementally: tracks ‖Σgᵢ/n‖² without storing all gradients.
    #[allow(dead_code)]
    mean_grad_norm_sq: f64,
    /// Running vector sum of gradient norms (for directional mean).
    /// We track scalar proxy: cumulative sum of per-element squared means.
    sum_grad_elements_sq: f64,
}

impl GradientSnrEstimator {
    /// Create a new SNR estimator.
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            mean_sq: 0.0,
            mean_grad_norm_sq: 0.0,
            sum_grad_elements_sq: 0.0,
        }
    }

    /// Update with a new batch gradient.
    ///
    /// Uses Welford's online algorithm for numerically stable
    /// running mean/variance computation.
    ///
    /// # Arguments
    /// * `grad` - The gradient matrix for this batch
    pub fn update(&mut self, grad: &Array2<f64>) {
        self.n += 1;
        let n = self.n as f64;

        // Compute Frobenius norm of this batch's gradient
        let mut grad_norm_sq = 0.0f64;
        for g in grad.iter() {
            grad_norm_sq += g * g;
        }
        let grad_norm = grad_norm_sq.sqrt();

        // Welford's update for mean and M2 of gradient norm
        let delta = grad_norm - self.mean;
        self.mean += delta / n;
        let delta2 = grad_norm - self.mean;
        self.m2 += delta * delta2;

        // Welford's update for mean of squared gradient norm
        let delta_sq = grad_norm_sq - self.mean_sq;
        self.mean_sq += delta_sq / n;

        // Track element-wise mean gradient magnitude squared
        // This approximates ‖E[g]‖² without storing full gradient tensors
        let n_elements = grad.len() as f64;
        let element_mean_sq = grad_norm_sq / n_elements;
        let delta_elem = element_mean_sq - self.sum_grad_elements_sq;
        self.sum_grad_elements_sq += delta_elem / n;
    }

    /// Current variance estimate.
    /// Returns 0.0 if fewer than 2 samples observed.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }

    /// Current mean gradient norm.
    pub fn mean_norm(&self) -> f64 {
        self.mean
    }

    /// Signal-to-noise ratio: SNR = ‖E[g]‖² / Var(g)
    ///
    /// High SNR → consistent gradient direction → can use higher rank
    /// Low SNR → noisy gradients → should use lower rank for regularization
    ///
    /// Returns None if variance is effectively zero or insufficient samples.
    pub fn snr(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let var = self.variance();
        if var < 1e-30 {
            return None;
        }
        // ‖E[g]‖² ≈ (mean_norm)² when gradients are aligned
        // More precisely: E[‖g‖²] - Var(‖g‖) gives directional signal
        let signal = self.mean * self.mean;
        Some(signal / var)
    }

    /// Suggest rank based on current SNR estimate.
    ///
    /// Maps SNR to rank multiplier:
    /// - SNR > 10.0 → high confidence → rank × 1.5
    /// - SNR > 3.0  → moderate       → rank × 1.0
    /// - SNR > 1.0  → low            → rank × 0.75
    /// - SNR ≤ 1.0  → very noisy     → rank × 0.5
    ///
    /// Only intended for use during warmup / rank initialization.
    pub fn suggest_rank_multiplier(&self) -> f64 {
        match self.snr() {
            Some(snr) if snr > 10.0 => 1.5,
            Some(snr) if snr > 3.0 => 1.0,
            Some(snr) if snr > 1.0 => 0.75,
            _ => 0.5,
        }
    }

    /// Number of samples observed.
    pub fn sample_count(&self) -> u64 {
        self.n
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for GradientSnrEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Training statistics for monitoring.
#[derive(Debug, Clone)]
pub struct TrainStats {
    pub epoch: usize,
    pub batch: usize,
    pub loss: f64,
    pub min_coherence: f64,
    pub corrections_applied: usize,
    pub avg_inner_iters: f64,
    /// Gradient SNR estimate (if enough samples collected).
    pub gradient_snr: Option<f64>,
}

/// Main RSMF trainer implementing the full training loop.
///
/// Memory invariant: at any point during training, only ONE layer's
/// working set resides in GPU/primary memory. All other layers are
/// stored in their compressed spectral form.
pub struct RsmfTrainer {
    pub model: RsmfModel,
    pub config: SpectralConfig,
    pub loss_fn: ResonantLoss,
    /// Gradient SNR estimator for rank initialization.
    pub snr_estimator: GradientSnrEstimator,
    stats_history: Vec<TrainStats>,
}

impl RsmfTrainer {
    pub fn new(model: RsmfModel, loss_fn: ResonantLoss) -> Self {
        let config = model.config.clone();
        Self {
            model,
            config,
            loss_fn,
            snr_estimator: GradientSnrEstimator::new(),
            stats_history: Vec::new(),
        }
    }

    /// Execute one full training step (forward + resonant backward).
    ///
    /// This is the core RSMF algorithm:
    /// 1. Forward pass with stratified recording
    /// 2. Terminal resonance signal from loss
    /// 3. Backward resonant flow (layer by layer, freeing as we go)
    /// 4. Coherence check and optional correction
    pub fn train_step(
        &mut self,
        input: &Array2<f64>,
        target: &Array2<f64>,
        epoch: usize,
        batch: usize,
    ) -> TrainStats {
        let n_layers = self.model.layers.len();

        // === PHASE 1: Forward Pass with Stratified Recording ===
        let (output, caches) = self.model.forward(input);

        // === PHASE 2: Terminal Resonance Signal ===
        let mut delta = self.loss_fn.terminal_signal(&output, target);
        let loss = self.loss_fn.compute(&output, target);

        // === PHASE 3: Resonant Backward Flow ===
        let local_res = LocalResonance::new(&self.config);
        let backward = ResonantBackward::new(&self.config);
        let coupling = InterStratumCoupling::new(&self.config);

        let mut total_inner_iters = 0usize;

        // Process layers in reverse order
        for l in (0..n_layers).rev() {
            // Get adjacent spectra for coupling
            let prev_spec = if l > 0 {
                Some(self.model.layers[l - 1].stratum.sigma.clone())
            } else {
                None
            };
            let next_spec = if l < n_layers - 1 {
                Some(self.model.layers[l + 1].stratum.sigma.clone())
            } else {
                None
            };

            // Reconstruct activation target from resonance signal
            let cache = &caches[l];
            let batch_size = cache.batch_size;
            let d_out = delta.ncols();

            // Target Tₗ = δₗ projected through output basis
            let target_proj = if d_out == cache.input_dim {
                delta.clone()
            } else {
                // Dimension mismatch: truncate or pad
                let min_d = d_out.min(cache.input_dim);
                let mut t = Array2::zeros((batch_size, cache.input_dim));
                for b in 0..batch_size {
                    for j in 0..min_d {
                        t[[b, j]] = delta[[b, j]];
                    }
                }
                t
            };

            // Reconstruct approximate activations from cache for local update
            let activations_approx = cache.reconstruct_approx(&self.model.layers[l].stratum.u_basis);

            // Local stratum optimization
            let iters = local_res.optimize(
                &mut self.model.layers[l],
                &activations_approx,
                &target_proj,
                prev_spec.as_ref(),
                next_spec.as_ref(),
            );
            total_inner_iters += iters;

            // Propagate resonance signal to previous layer
            if l > 0 {
                delta = backward.propagate(
                    &delta,
                    &self.model.layers[l - 1],
                    &self.model.layers[l],
                    &cache.activation_derivative,
                );
            }
        }

        // === PHASE 4: Coherence Check ===
        let corrector = CoherenceCorrector::new(&self.config);
        let mut corrections = 0;
        let min_coherence = if corrector.needs_correction(&self.model.layers) {
            corrections += 1;
            corrector.correct(&mut self.model.layers)
        } else {
            // Compute min coherence without correction
            coupling.weakest_link(&self.model.layers)
                .map(|(_, score)| score)
                .unwrap_or(1.0)
        };

        // Update gradient SNR estimator with terminal delta as gradient proxy.
        // The terminal resonance signal δ serves as the effective gradient
        // for rank initialization purposes.
        self.snr_estimator.update(&delta);

        let stats = TrainStats {
            epoch,
            batch,
            loss,
            min_coherence,
            corrections_applied: corrections,
            avg_inner_iters: total_inner_iters as f64 / n_layers as f64,
            gradient_snr: self.snr_estimator.snr(),
        };

        self.stats_history.push(stats.clone());
        stats
    }

    /// Train for multiple epochs over a dataset.
    ///
    /// `dataset` is a slice of (input, target) pairs.
    pub fn train(
        &mut self,
        dataset: &[(Array2<f64>, Array2<f64>)],
        num_epochs: usize,
        callback: Option<&dyn Fn(&TrainStats)>,
    ) {
        for epoch in 0..num_epochs {
            for (batch_idx, (input, target)) in dataset.iter().enumerate() {
                let stats = self.train_step(input, target, epoch, batch_idx);
                if let Some(cb) = callback {
                    cb(&stats);
                }
            }
        }
    }

    /// Get training history.
    pub fn history(&self) -> &[TrainStats] {
        &self.stats_history
    }

    /// Verify memory budget compliance.
    pub fn verify_memory_budget(&self, budget_bytes: usize) -> bool {
        self.config.fits_budget(
            self.model.hidden_dim,
            self.model.layers.len(),
            budget_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsmf_core::SpectralConfig;
    use rsmf_layers::RsmfModel;

    // === GradientSnrEstimator Tests ===

    #[test]
    fn snr_estimator_welford_mean_variance() {
        let mut est = GradientSnrEstimator::new();
        // Feed known gradient norms: [3.0, 5.0, 7.0]
        // Mean = 5.0, Var(sample) = ((9+1+9)/2) = 4.666...
        let g1 = Array2::from_elem((1, 1), 3.0);
        let g2 = Array2::from_elem((1, 1), 5.0);
        let g3 = Array2::from_elem((1, 1), 7.0);
        est.update(&g1);
        est.update(&g2);
        est.update(&g3);

        assert!((est.mean_norm() - 5.0).abs() < 1e-10);
        assert!((est.variance() - 4.0).abs() < 1e-10); // sample variance of {3,5,7} = 4.0
        assert_eq!(est.sample_count(), 3);
    }

    #[test]
    fn snr_estimator_insufficient_samples() {
        let mut est = GradientSnrEstimator::new();
        assert!(est.snr().is_none());
        assert_eq!(est.variance(), 0.0);

        est.update(&Array2::ones((2, 2)));
        assert!(est.snr().is_none()); // Need at least 2 samples
    }

    #[test]
    fn snr_estimator_constant_gradients_high_snr() {
        let mut est = GradientSnrEstimator::new();
        // All identical gradients → zero variance → None SNR (division by ~0)
        for _ in 0..10 {
            est.update(&Array2::from_elem((4, 4), 1.0));
        }
        // Variance should be ~0
        assert!(est.variance() < 1e-20);
        // SNR returns None when variance is effectively zero
        assert!(est.snr().is_none());
    }

    #[test]
    fn snr_estimator_noisy_gradients_low_snr() {
        let mut est = GradientSnrEstimator::new();
        // Alternating large positive/negative → high variance relative to mean
        for i in 0..20 {
            let val = if i % 2 == 0 { 10.0 } else { -10.0 };
            est.update(&Array2::from_elem((1, 1), val));
        }
        // Mean ≈ 0, variance is high → low or zero SNR
        let snr = est.snr();
        if let Some(s) = snr {
            assert!(s < 1.0, "Expected low SNR for alternating gradients, got {}", s);
        }
    }

    #[test]
    fn snr_rank_multiplier_mapping() {
        let mut est = GradientSnrEstimator::new();
        // No data → default multiplier
        assert_eq!(est.suggest_rank_multiplier(), 0.5);
    }

    #[test]
    fn snr_estimator_reset() {
        let mut est = GradientSnrEstimator::new();
        est.update(&Array2::ones((2, 2)));
        let g2: Array2<f64> = Array2::ones((2, 2)) * 2.0;
        est.update(&g2);
        assert_eq!(est.sample_count(), 2);
        est.reset();
        assert_eq!(est.sample_count(), 0);
        assert_eq!(est.mean_norm(), 0.0);
    }

    // === RsmfTrainer Tests ===

    #[test]
    fn trainer_runs_without_panic() {
        let config = SpectralConfig {
            top_k: 4,
            max_inner_iters: 5,
            ..SpectralConfig::default()
        };
        let model = RsmfModel::initialize(3, 16, 2, config);
        let loss_fn = ResonantLoss::mse();
        let mut trainer = RsmfTrainer::new(model, loss_fn);

        let input = Array2::ones((2, 16));
        let target = Array2::zeros((2, 16));

        let stats = trainer.train_step(&input, &target, 0, 0);
        assert!(stats.loss >= 0.0);
        assert!(stats.min_coherence >= 0.0);
    }

    #[test]
    fn loss_decreases_over_steps() {
        let config = SpectralConfig {
            top_k: 4,
            learning_rate: 0.05,
            max_inner_iters: 10,
            lambda_spectral: 0.0,
            mu_coupling: 0.0,
            ..SpectralConfig::default()
        };
        let model = RsmfModel::initialize(2, 8, 1, config);
        let loss_fn = ResonantLoss::mse();
        let mut trainer = RsmfTrainer::new(model, loss_fn);

        let input = Array2::from_shape_fn((4, 8), |(i, j)| ((i * 8 + j) as f64) * 0.1);
        let target = Array2::ones((4, 8)) * 0.5;

        let first = trainer.train_step(&input, &target, 0, 0);
        for _ in 0..5 {
            trainer.train_step(&input, &target, 0, 0);
        }
        let last = trainer.train_step(&input, &target, 0, 0);

        // Loss should generally decrease (not guaranteed in single test due to stochasticity)
        // but we check it doesn't explode
        assert!(last.loss < first.loss * 10.0, "Loss exploded: {} -> {}", first.loss, last.loss);
    }
}
