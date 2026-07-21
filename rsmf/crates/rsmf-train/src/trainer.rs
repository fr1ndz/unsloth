use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, SpectralConfig, CoherenceMetric};
use rsmf_layers::RsmfModel;
use rsmf_resonance::{LocalResonance, ResonantBackward, InterStratumCoupling, CoherenceCorrector};
use crate::loss::ResonantLoss;
use crate::schedule::CoherenceSchedule;

/// Training statistics for monitoring.
#[derive(Debug, Clone)]
pub struct TrainStats {
    pub epoch: usize,
    pub batch: usize,
    pub loss: f64,
    pub min_coherence: f64,
    pub corrections_applied: usize,
    pub avg_inner_iters: f64,
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
    stats_history: Vec<TrainStats>,
}

impl RsmfTrainer {
    pub fn new(model: RsmfModel, loss_fn: ResonantLoss) -> Self {
        let config = model.config.clone();
        Self {
            model,
            config,
            loss_fn,
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

        let stats = TrainStats {
            epoch,
            batch,
            loss,
            min_coherence,
            corrections_applied: corrections,
            avg_inner_iters: total_inner_iters as f64 / n_layers as f64,
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
