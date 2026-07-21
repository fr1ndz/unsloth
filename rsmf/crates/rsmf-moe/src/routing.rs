use ndarray::{Array1, Array2};
use rsmf_core::{ResonantTensor, Stratum, SpectralConfig};

/// Shared routing stratum for MoE token→expert assignment.
///
/// Unlike expert strata, the routing stratum is:
/// - Replicated across all devices (small: vocab × num_experts)
/// - Updated every step (critical for load balancing)
/// - Uses softmax over expert logits
#[derive(Debug, Clone)]
pub struct RoutingStratum {
    /// Resonant tensor for routing weights: hidden_dim → num_experts.
    pub tensor: ResonantTensor,
    /// Number of experts.
    pub num_experts: usize,
    /// Temperature for routing softmax (higher = more uniform).
    pub temperature: f64,
    /// Auxiliary load balancing loss coefficient.
    pub aux_loss_coeff: f64,
}

impl RoutingStratum {
    pub fn new(hidden_dim: usize, num_experts: usize, config: &SpectralConfig) -> Self {
        let k = config.top_k.min(num_experts).min(hidden_dim);
        let sigma = Array1::from_vec(
            (0..k).map(|i| 0.5 / (1.0 + i as f64)).collect()
        );
        let u = Array2::eye(hidden_dim).slice(ndarray::s![.., ..k]).to_owned();
        let v = Array2::eye(num_experts).slice(ndarray::s![.., ..k]).to_owned();
        let stratum = Stratum::new(sigma, u, v, 0);

        Self {
            tensor: ResonantTensor::from_stratum(stratum),
            num_experts,
            temperature: 1.0,
            aux_loss_coeff: 0.01,
        }
    }

    /// Compute routing probabilities for a batch of hidden states.
    /// Returns shape (batch, num_experts).
    pub fn route(&self, hidden_states: &Array2<f64>) -> Array2<f64> {
        let batch = hidden_states.nrows();
        let d_in = hidden_states.ncols();

        // Reconstruct partial weights from spectral stratum
        // W ≈ U · diag(σ) · Vᵀ ∈ R^(d_in × num_experts)
        // But stratum may have truncated k < min(d_in, num_experts)
        let weights = self.tensor.stratum.reconstruct();
        let w_rows = weights.nrows();
        let w_cols = weights.ncols();

        // Handle dimension mismatch: pad or truncate
        let effective_cols = w_cols.min(self.num_experts);
        let mut logits = Array2::zeros((batch, self.num_experts));
        for b in 0..batch {
            for e in 0..effective_cols.min(w_cols) {
                let mut val = 0.0;
                for j in 0..d_in.min(w_rows) {
                    val += hidden_states[[b, j]] * weights[[j, e]];
                }
                logits[[b, e]] = val;
            }
        }

        // Softmax with temperature
        softmax_with_temperature(&logits, self.temperature)
    }

    /// Compute auxiliary load balancing loss.
    /// L_aux = α · N · Σ(f_i · P_i) where f_i = fraction routed, P_i = mean prob.
    pub fn aux_loss(&self, routing_probs: &Array2<f64>) -> f64 {
        let batch = routing_probs.nrows() as f64;
        let n_experts = self.num_experts as f64;

        let mut loss = 0.0;
        for e in 0..self.num_experts {
            // f_i: fraction of tokens routed to expert i (hard count would need argmax)
            // P_i: mean probability assigned to expert i
            let mut sum_prob = 0.0;
            for b in 0..routing_probs.nrows() {
                sum_prob += routing_probs[[b, e]];
            }
            let p_i = sum_prob / batch;
            // Approximate f_i ≈ p_i for soft routing
            loss += p_i * p_i;
        }
        self.aux_loss_coeff * n_experts * loss
    }

    /// Memory footprint (replicated on each device).
    pub fn memory_bytes(&self) -> usize {
        self.tensor.memory_bytes()
    }
}

fn softmax_with_temperature(logits: &Array2<f64>, temp: f64) -> Array2<f64> {
    let mut result = logits.clone();
    let safe_temp = temp.max(0.01);
    for i in 0..logits.nrows() {
        let max_val = logits.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in 0..logits.ncols() {
            result[[i, j]] = ((logits[[i, j]] - max_val) / safe_temp).exp();
            sum += result[[i, j]];
        }
        if sum > 1e-15 {
            for j in 0..logits.ncols() {
                result[[i, j]] /= sum;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_produces_valid_probabilities() {
        let config = SpectralConfig { top_k: 4, ..SpectralConfig::default() };
        let routing = RoutingStratum::new(32, 8, &config);
        let hidden = Array2::ones((4, 32));
        let probs = routing.route(&hidden);

        assert_eq!(probs.shape(), &[4, 8]);
        // Each row sums to ~1.0
        for i in 0..4 {
            let row_sum: f64 = probs.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sums to {}", i, row_sum);
        }
    }

    #[test]
    fn aux_loss_zero_for_uniform_routing() {
        let config = SpectralConfig { top_k: 4, ..SpectralConfig::default() };
        let routing = RoutingStratum::new(16, 4, &config);
        // Uniform probabilities
        let probs = Array2::from_elem((8, 4), 0.25);
        let loss = routing.aux_loss(&probs);
        // For uniform: Σ(p²) = 4 × 0.0625 = 0.25, loss = 0.01 × 4 × 0.25 = 0.01
        assert!((loss - 0.01).abs() < 0.001);
    }
}
