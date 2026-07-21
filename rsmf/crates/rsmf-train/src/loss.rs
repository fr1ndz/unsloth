use ndarray::Array2;

/// Resonant loss functions for RSMF training.
///
/// Unlike standard losses that produce scalar values, these produce
/// terminal resonance signals δ_L that initiate the backward flow.
#[derive(Debug, Clone)]
pub enum ResonantLoss {
    /// Mean Squared Error: δ = 2·(output - target) / batch_size
    Mse,
    /// Cross-entropy compatible: δ = softmax(output) - target
    CrossEntropy,
    /// Custom weighted combination
    Weighted { mse_weight: f64, ce_weight: f64 },
}

impl ResonantLoss {
    pub fn mse() -> Self {
        Self::Mse
    }

    pub fn cross_entropy() -> Self {
        Self::CrossEntropy
    }

    /// Compute scalar loss value for monitoring.
    pub fn compute(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        match self {
            Self::Mse => {
                let diff = output - target;
                diff.mapv(|x| x * x).sum() / output.nrows() as f64
            }
            Self::CrossEntropy => {
                // Simplified: sum of squared differences as proxy
                let diff = output - target;
                diff.mapv(|x| x.abs()).sum() / output.nrows() as f64
            }
            Self::Weighted { mse_weight, ce_weight } => {
                let mse_loss = ResonantLoss::Mse.compute(output, target);
                let ce_loss = ResonantLoss::CrossEntropy.compute(output, target);
                mse_weight * mse_loss + ce_weight * ce_loss
            }
        }
    }

    /// Compute terminal resonance signal δ_L.
    ///
    /// This replaces the gradient ∂L/∂h_L in backpropagation.
    /// Shape matches output: (batch, d_out).
    pub fn terminal_signal(&self, output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
        let batch = output.nrows() as f64;
        match self {
            Self::Mse => {
                // δ = 2·(output - target) / batch
                let mut delta = output - target;
                delta.mapv_inplace(|x| 2.0 * x / batch);
                delta
            }
            Self::CrossEntropy => {
                // Simplified: δ = softmax(output) - target
                let softmax = softmax_rows(output);
                (&softmax - target) / batch
            }
            Self::Weighted { mse_weight, ce_weight } => {
                let mse_delta = ResonantLoss::Mse.terminal_signal(output, target);
                let ce_delta = ResonantLoss::CrossEntropy.terminal_signal(output, target);
                &mse_delta * *mse_weight + &ce_delta * *ce_weight
            }
        }
    }
}

/// Row-wise softmax for cross-entropy computation.
fn softmax_rows(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for i in 0..x.nrows() {
        // Find max for numerical stability
        let max_val = x.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in 0..x.ncols() {
            result[[i, j]] = (x[[i, j]] - max_val).exp();
            sum += result[[i, j]];
        }
        if sum > 1e-15 {
            for j in 0..x.ncols() {
                result[[i, j]] /= sum;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mse_loss_zero_for_identical() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let loss = ResonantLoss::mse().compute(&x, &x);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn terminal_signal_shape_matches() {
        let output = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let target = array![[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]];
        let delta = ResonantLoss::mse().terminal_signal(&output, &target);
        assert_eq!(delta.shape(), output.shape());
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = array![[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]];
        let s = softmax_rows(&x);
        for i in 0..s.nrows() {
            let row_sum: f64 = s.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sums to {}", i, row_sum);
        }
    }
}
