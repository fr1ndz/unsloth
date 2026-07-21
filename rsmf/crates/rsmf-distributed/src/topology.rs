use serde::{Deserialize, Serialize};
use crate::model_config::{ModelArchitecture, ShardStrategy};

/// Describes the physical device topology for distributed RSMF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTopology {
    /// Per-device VRAM capacity in bytes.
    pub device_vram: Vec<u64>,
    /// Inter-device bandwidth in GB/s (symmetric assumption).
    pub bandwidth_gbps: f64,
    /// Number of pipeline stages (groups of consecutive layers).
    pub num_pipeline_stages: usize,
    /// Assignment of layers to pipeline stages.
    pub stage_assignments: Vec<PipelineStage>,
}

/// A pipeline stage: contiguous range of layers assigned to device group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage index.
    pub stage_id: usize,
    /// Device IDs participating in this stage.
    pub device_ids: Vec<usize>,
    /// Layer range [start, end) assigned to this stage.
    pub layer_range: (usize, usize),
    /// Number of micro-batches in flight for pipeline overlap.
    pub micro_batches: usize,
}

impl DeviceTopology {
    /// Create uniform topology: N identical GPUs, simple pipeline.
    pub fn uniform(num_devices: usize, vram_per_device: u64, num_layers: usize) -> Self {
        let layers_per_stage = (num_layers + num_devices - 1) / num_devices;
        let stage_assignments = (0..num_devices)
            .map(|d| {
                let start = d * layers_per_stage;
                let end = (start + layers_per_stage).min(num_layers);
                PipelineStage {
                    stage_id: d,
                    device_ids: vec![d],
                    layer_range: (start, end),
                    micro_batches: 4,
                }
            })
            .collect();

        Self {
            device_vram: vec![vram_per_device; num_devices],
            bandwidth_gbps: 100.0, // Default NVLink-like
            num_pipeline_stages: num_devices,
            stage_assignments,
        }
    }

    /// Total VRAM across all devices.
    pub fn total_vram(&self) -> u64 {
        self.device_vram.iter().sum()
    }

    /// Check if architecture fits this topology with given shard strategy.
    pub fn can_host(&self, arch: &ModelArchitecture, strategy: ShardStrategy) -> bool {
        let needed = arch.vram_full_fp16();
        let available = self.total_vram();
        match strategy {
            ShardStrategy::SpectralK => {
                // With RSMF spectral compression (~10-50×), single layer must fit
                let per_layer = needed / arch_num_layers(arch) as u64;
                let min_device = self.device_vram.iter().min().copied().unwrap_or(0);
                per_layer < min_device
            }
            ShardStrategy::PipelineLayers | ShardStrategy::HybridPipelineSpectral => {
                needed < available * 10 // RSMF compression factor estimate
            }
            ShardStrategy::ExpertParallel => {
                // Experts distributed, routing replicated
                needed < available * 8
            }
        }
    }

    /// Estimate communication cost per training step (bytes transferred).
    pub fn comm_cost_per_step(&self, hidden_dim: usize, batch_size: usize) -> u64 {
        // Pipeline boundaries: activations cross stage boundaries
        let boundaries = self.num_pipeline_stages.saturating_sub(1) as u64;
        let activation_bytes = batch_size as u64 * hidden_dim as u64 * 2; // FP16
        boundaries * activation_bytes
    }
}

fn arch_num_layers(arch: &ModelArchitecture) -> usize {
    match arch {
        ModelArchitecture::Dense(cfg) => cfg.num_layers,
        ModelArchitecture::Moe(cfg) => cfg.base.num_layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::DenseConfig;

    #[test]
    fn uniform_topology_creation() {
        let topo = DeviceTopology::uniform(4, 24 * 1024 * 1024 * 1024, 80);
        assert_eq!(topo.num_pipeline_stages, 4);
        assert_eq!(topo.stage_assignments.len(), 4);
        // All layers covered
        let total_layers: usize = topo.stage_assignments.iter()
            .map(|s| s.layer_range.1 - s.layer_range.0)
            .sum();
        assert_eq!(total_layers, 80);
    }

    #[test]
    fn gpt2_fits_single_gpu() {
        let topo = DeviceTopology::uniform(1, 24 * 1024 * 1024 * 1024, 12);
        let arch = ModelArchitecture::Dense(DenseConfig::gpt2_small());
        assert!(topo.can_host(&arch, ShardStrategy::SpectralK));
    }
}
