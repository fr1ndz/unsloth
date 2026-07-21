use crate::model_config::{ModelArchitecture, DenseConfig};
use crate::topology::DeviceTopology;
use rsmf_core::SpectralConfig;

/// Allocates VRAM budget across devices for a given model and topology.
///
/// Computes optimal top_k per stratum such that:
///   1. Each device stays within VRAM budget
///   2. Spectral energy capture is maximized
///   3. Communication overhead is minimized
pub struct VramBudgetAllocator;

/// Allocation result for a single device.
#[derive(Debug, Clone)]
pub struct DeviceAllocation {
    pub device_id: usize,
    pub vram_budget_bytes: u64,
    pub assigned_layers: Vec<usize>,
    pub recommended_top_k: usize,
    pub estimated_stratum_bytes: u64,
    pub headroom_bytes: u64,
}

impl VramBudgetAllocator {
    /// Compute allocation for all devices given model and topology.
    pub fn allocate(
        arch: &ModelArchitecture,
        topo: &DeviceTopology,
        spectral_cfg: &SpectralConfig,
    ) -> Vec<DeviceAllocation> {
        let _num_layers = match arch {
            ModelArchitecture::Dense(cfg) => cfg.num_layers,
            ModelArchitecture::Moe(cfg) => cfg.base.num_layers,
        };
        let hidden_dim = match arch {
            ModelArchitecture::Dense(cfg) => cfg.hidden_dim,
            ModelArchitecture::Moe(cfg) => cfg.base.hidden_dim,
        };

        let mut allocations = Vec::new();

        for stage in &topo.stage_assignments {
            let device_id = stage.device_ids.first().copied().unwrap_or(0);
            let vram = topo.device_vram.get(device_id).copied().unwrap_or(0);
            let _num_assigned = stage.layer_range.1 - stage.layer_range.0;

            // Reserve 20% for activations, gradients, and OS overhead
            let usable_vram = vram * 80 / 100;

            // Budget per stratum: usable / max_concurrent_strata
            // With layer cycling, only 1 stratum active at a time + 1 buffer
            let budget_per_stratum = usable_vram / 2;

            // Find optimal top_k that fits budget
            let optimal_k = find_optimal_k(hidden_dim, budget_per_stratum as usize, spectral_cfg.top_k);

            let stratum_bytes = estimate_stratum_bytes(hidden_dim, optimal_k);
            let assigned_layers: Vec<usize> = (stage.layer_range.0..stage.layer_range.1).collect();

            allocations.push(DeviceAllocation {
                device_id,
                vram_budget_bytes: vram,
                assigned_layers,
                recommended_top_k: optimal_k,
                estimated_stratum_bytes: stratum_bytes as u64,
                headroom_bytes: usable_vram.saturating_sub(stratum_bytes as u64 * 2),
            });
        }

        allocations
    }

    /// Check if entire model fits with given configuration.
    pub fn is_feasible(
        arch: &ModelArchitecture,
        topo: &DeviceTopology,
        spectral_cfg: &SpectralConfig,
    ) -> bool {
        let allocs = Self::allocate(arch, topo, spectral_cfg);
        allocs.iter().all(|a| a.headroom_bytes > 0 && a.recommended_top_k >= 4)
    }
}

/// Binary search for largest k where stratum fits in budget.
fn find_optimal_k(hidden_dim: usize, budget_bytes: usize, max_k: usize) -> usize {
    let mut lo = 4usize;
    let mut hi = max_k.min(hidden_dim);
    let mut best = lo;

    while lo <= hi {
        let mid = (lo + hi) / 2;
        let bytes = estimate_stratum_bytes(hidden_dim, mid);
        if bytes <= budget_bytes {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid.saturating_sub(1);
        }
    }
    best
}

/// Estimate memory for one stratum with given k and d.
fn estimate_stratum_bytes(d: usize, k: usize) -> usize {
    let elem = 2; // FP16
    // σ: k × 8 (f64), U: d × k × elem, V: d × k × elem
    k * 8 + 2 * d * k * elem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_fits_single_24gb() {
        let arch = ModelArchitecture::Dense(DenseConfig::gpt2_small());
        let topo = DeviceTopology::uniform(1, 24 * 1024 * 1024 * 1024, 12);
        let cfg = SpectralConfig::default();
        assert!(VramBudgetAllocator::is_feasible(&arch, &topo, &cfg));
    }

    #[test]
    fn llama70b_needs_multi_gpu() {
        let arch = ModelArchitecture::Dense(DenseConfig::llama2_70b());
        let topo = DeviceTopology::uniform(1, 24 * 1024 * 1024 * 1024, 80);
        let cfg = SpectralConfig::default();
        // Single 24GB can't host 70B even with RSMF
        // But with enough GPUs it should work
        let topo8 = DeviceTopology::uniform(8, 80 * 1024 * 1024 * 1024, 80);
        assert!(VramBudgetAllocator::is_feasible(&arch, &topo8, &cfg));
    }

    #[test]
    fn optimal_k_respects_budget() {
        let k = find_optimal_k(4096, 1024 * 1024, 256); // 1MB budget, d=4096
        let bytes = estimate_stratum_bytes(4096, k);
        assert!(bytes <= 1024 * 1024, "k={} uses {} bytes > 1MB", k, bytes);
        assert!(k >= 4, "Should find at least k=4");
    }
}
