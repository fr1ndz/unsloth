use rsmf_core::SpectralConfig;

/// Adaptive coherence threshold schedule.
///
/// Starts with lenient coherence requirements early in training
/// (allowing exploration) and tightens them as training progresses
/// (enforcing stability).
#[derive(Debug, Clone)]
pub struct CoherenceSchedule {
    /// Initial coherence threshold (lenient).
    pub initial_threshold: f64,
    /// Final coherence threshold (strict).
    pub final_threshold: f64,
    /// Number of warmup epochs before tightening begins.
    pub warmup_epochs: usize,
    /// Total training epochs for schedule interpolation.
    pub total_epochs: usize,
}

impl CoherenceSchedule {
    pub fn new(total_epochs: usize) -> Self {
        Self {
            initial_threshold: 0.1,
            final_threshold: 0.5,
            warmup_epochs: (total_epochs / 10).max(1),
            total_epochs,
        }
    }

    /// Get coherence threshold for given epoch.
    pub fn threshold_at(&self, epoch: usize) -> f64 {
        if epoch < self.warmup_epochs {
            return self.initial_threshold;
        }
        let progress = (epoch - self.warmup_epochs) as f64
            / (self.total_epochs - self.warmup_epochs).max(1) as f64;
        let progress = progress.min(1.0);
        self.initial_threshold + (self.final_threshold - self.initial_threshold) * progress
    }

    /// Apply current threshold to config.
    pub fn apply_to_config(&self, config: &mut SpectralConfig, epoch: usize) {
        config.coherence_threshold = self.threshold_at(epoch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_interpolates_correctly() {
        let sched = CoherenceSchedule {
            initial_threshold: 0.1,
            final_threshold: 0.5,
            warmup_epochs: 2,
            total_epochs: 12,
        };

        assert!((sched.threshold_at(0) - 0.1).abs() < 1e-10);
        assert!((sched.threshold_at(1) - 0.1).abs() < 1e-10);
        assert!((sched.threshold_at(7) - 0.3).abs() < 0.05);
        assert!((sched.threshold_at(12) - 0.5).abs() < 1e-10);
    }
}
