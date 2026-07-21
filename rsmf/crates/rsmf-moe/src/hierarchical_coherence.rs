use rsmf_core::{CoherenceMetric, ResonantTensor, SpectralConfig};
use crate::expert::ExpertGroup;

/// Three-level hierarchical coherence checking for MoE models.
///
/// Level 1: Intra-expert — spectral stability within each expert's stratum
/// Level 2: Inter-expert — diversity between experts in same layer
/// Level 3: Inter-layer — coherence between consecutive MoE layers
///
/// This prevents:
/// - Expert collapse (all experts converging to same function)
/// - Layer drift (consecutive layers losing alignment)
/// - Spectral degeneration (singular values collapsing)
#[derive(Debug, Clone)]
pub struct HierarchicalCoherence {
    config: SpectralConfig,
    /// Minimum diversity score between experts (prevents collapse).
    pub min_expert_diversity: f64,
}

/// Result of hierarchical coherence check.
#[derive(Debug, Clone)]
pub struct CoherenceReport {
    pub intra_expert_ok: Vec<bool>,
    pub inter_expert_diversity: f64,
    pub inter_layer_coherence: Option<f64>,
    pub overall_healthy: bool,
    pub warnings: Vec<String>,
}

impl HierarchicalCoherence {
    pub fn new(config: SpectralConfig) -> Self {
        Self {
            config,
            min_expert_diversity: 0.1,
        }
    }

    /// Full hierarchical coherence check for one MoE layer.
    pub fn check_layer(
        &self,
        expert_group: &ExpertGroup,
        prev_group: Option<&ExpertGroup>,
    ) -> CoherenceReport {
        let mut warnings = Vec::new();

        // Level 1: Intra-expert coherence
        let intra_ok: Vec<bool> = expert_group.experts.iter().map(|e| {
            e.tensor.is_well_conditioned(100.0)
        }).collect();

        for (i, ok) in intra_ok.iter().enumerate() {
            if !ok {
                warnings.push(format!("Expert {} ill-conditioned (cond={:.1})",
                    i, expert_group.experts[i].tensor.condition_number));
            }
        }

        // Level 2: Inter-expert diversity
        let diversity = self.compute_expert_diversity(expert_group);
        if diversity < self.min_expert_diversity {
            warnings.push(format!("Low expert diversity: {:.4} < {:.4}",
                diversity, self.min_expert_diversity));
        }

        // Level 3: Inter-layer coherence
        let inter_layer = prev_group.map(|prev| {
            self.compute_inter_layer_coherence(prev, expert_group)
        });

        if let Some(score) = inter_layer {
            if score < self.config.coherence_threshold {
                warnings.push(format!("Low inter-layer coherence: {:.4}", score));
            }
        }

        let overall_healthy = intra_ok.iter().all(|&ok| ok)
            && diversity >= self.min_expert_diversity
            && inter_layer.map_or(true, |s| s >= self.config.coherence_threshold);

        CoherenceReport {
            intra_expert_ok: intra_ok,
            inter_expert_diversity: diversity,
            inter_layer_coherence: inter_layer,
            overall_healthy,
            warnings,
        }
    }

    /// Compute diversity between experts using spectral distance.
    /// High diversity = experts learned different functions.
    fn compute_expert_diversity(&self, group: &ExpertGroup) -> f64 {
        let n = group.experts.len();
        if n < 2 { return 1.0; }

        let mut total_distance = 0.0;
        let mut pairs = 0;

        for i in 0..n {
            for j in (i+1)..n {
                let metric = CoherenceMetric::between(
                    &group.experts[i].tensor,
                    &group.experts[j].tensor,
                );
                // Distance = 1 - similarity
                total_distance += 1.0 - metric.score;
                pairs += 1;
            }
        }

        if pairs == 0 { return 1.0; }
        total_distance / pairs as f64
    }

    /// Compute coherence between corresponding experts of adjacent layers.
    fn compute_inter_layer_coherence(&self, prev: &ExpertGroup, curr: &ExpertGroup) -> f64 {
        let n = prev.experts.len().min(curr.experts.len());
        if n == 0 { return 1.0; }

        let mut total = 0.0;
        for i in 0..n {
            let metric = CoherenceMetric::between(
                &prev.experts[i].tensor,
                &curr.experts[i].tensor,
            );
            total += metric.score;
        }
        total / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_experts_have_zero_diversity() {
        let config = SpectralConfig { top_k: 4, ..SpectralConfig::default() };
        let hc = HierarchicalCoherence::new(config);
        let group = ExpertGroup::new(0, 4, 2, 16, &hc.config);
        // All experts initialized identically → low diversity
        let div = hc.compute_expert_diversity(&group);
        assert!(div < 0.01, "Identical experts should have ~0 diversity: {}", div);
    }

    #[test]
    fn healthy_report_for_fresh_model() {
        let config = SpectralConfig {
            top_k: 4,
            coherence_threshold: 0.0, // Lenient for init
            ..SpectralConfig::default()
        };
        let mut hc = HierarchicalCoherence::new(config);
        // Fresh experts have identical init → zero diversity is expected.
        // Lower threshold to accept initialized state.
        hc.min_expert_diversity = 0.0;
        let group = ExpertGroup::new(0, 4, 2, 16, &hc.config);
        let report = hc.check_layer(&group, None);
        assert!(report.overall_healthy, "Fresh model should be healthy with lenient thresholds");
    }
}
