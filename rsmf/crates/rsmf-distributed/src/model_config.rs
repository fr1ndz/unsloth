use serde::{Deserialize, Serialize};

/// High-level model architecture descriptor.
///
/// Determines how RSMF distributes computation across devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Standard dense transformer: all tokens flow through all parameters.
    Dense(DenseConfig),
    /// Mixture-of-Experts: sparse activation with shared routing.
    Moe(MoeConfig),
}

/// Configuration for dense transformer models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseConfig {
    /// Total number of parameters (e.g., 70_000_000_000 for 70B).
    pub total_params: u64,
    /// Hidden dimension per layer.
    pub hidden_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Intermediate (FFN) dimension. Typically 4× hidden_dim.
    pub intermediate_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
}

/// Configuration for Mixture-of-Experts models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Base dense config (shared attention + non-expert layers).
    pub base: DenseConfig,
    /// Number of experts per MoE layer.
    pub num_experts: usize,
    /// Number of experts activated per token (top-k routing).
    pub active_experts: usize,
    /// Which layers are MoE layers (indices). Others are dense.
    pub moe_layer_indices: Vec<usize>,
    /// Expert intermediate dimension (may differ from base).
    pub expert_intermediate_dim: usize,
}

/// Strategy for distributing strata across devices.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Split top-k singular components across GPUs along k-dimension.
    /// Best when k is large and inter-device bandwidth is limited.
    SpectralK,
    /// Assign contiguous layer ranges to each GPU (pipeline parallelism).
    /// Best when model depth >> width.
    PipelineLayers,
    /// Hybrid: pipeline over layer groups + spectral-K within each group.
    /// Optimal for 100B+ on multi-node clusters.
    HybridPipelineSpectral,
    /// Each expert assigned to dedicated GPU(s). Routing stratum replicated.
    /// Optimal for MoE with many experts.
    ExpertParallel,
}

impl ModelArchitecture {
    /// Estimate total VRAM needed for full model in FP16.
    pub fn vram_full_fp16(&self) -> u64 {
        match self {
            Self::Dense(cfg) => cfg.total_params * 2, // FP16 = 2 bytes
            Self::Moe(cfg) => {
                // Shared params + expert params
                let shared = cfg.base.num_layers.saturating_sub(cfg.moe_layer_indices.len()) as u64;
                let shared_bytes = shared * (cfg.base.hidden_dim as u64).pow(2) * 4 * 2;
                let expert_layers = cfg.moe_layer_indices.len() as u64;
                let expert_bytes = expert_layers
                    * cfg.num_experts as u64
                    * (cfg.expert_intermediate_dim as u64)
                    * (cfg.base.hidden_dim as u64)
                    * 3 // gate_up + down
                    * 2; // FP16
                shared_bytes + expert_bytes
            }
        }
    }

    /// Recommended shard strategy based on architecture and available VRAM.
    pub fn recommend_shard_strategy(&self, total_vram_bytes: u64) -> ShardStrategy {
        let full_vram = self.vram_full_fp16();
        match self {
            Self::Moe(_) if full_vram > total_vram_bytes => ShardStrategy::ExpertParallel,
            Self::Dense(cfg) if cfg.num_layers > 64 && full_vram > total_vram_bytes * 4 => {
                ShardStrategy::HybridPipelineSpectral
            }
            _ if full_vram > total_vram_bytes => ShardStrategy::PipelineLayers,
            _ => ShardStrategy::SpectralK,
        }
    }
}

impl DenseConfig {
    /// GPT-2 Small (124M)
    pub fn gpt2_small() -> Self {
        Self {
            total_params: 124_000_000,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_dim: 3072,
            vocab_size: 50257,
        }
    }

    /// Llama-2 70B
    pub fn llama2_70b() -> Self {
        Self {
            total_params: 70_000_000_000,
            hidden_dim: 8192,
            num_layers: 80,
            num_heads: 64,
            intermediate_dim: 28672,
            vocab_size: 32000,
        }
    }

    /// Hypothetical 700B dense
    pub fn dense_700b() -> Self {
        Self {
            total_params: 700_000_000_000,
            hidden_dim: 16384,
            num_layers: 128,
            num_heads: 128,
            intermediate_dim: 65536,
            vocab_size: 128000,
        }
    }
}

impl MoeConfig {
    /// Mixtral 8x7B style
    pub fn mixtral_8x7b() -> Self {
        Self {
            base: DenseConfig {
                total_params: 46_700_000_000,
                hidden_dim: 4096,
                num_layers: 32,
                num_heads: 32,
                intermediate_dim: 14336,
                vocab_size: 32000,
            },
            num_experts: 8,
            active_experts: 2,
            moe_layer_indices: (0..32).collect(),
            expert_intermediate_dim: 14336,
        }
    }

    /// Hypothetical 700B MoE (e.g., 128 experts, top-8)
    pub fn moe_700b() -> Self {
        Self {
            base: DenseConfig {
                total_params: 700_000_000_000,
                hidden_dim: 8192,
                num_layers: 96,
                num_heads: 64,
                intermediate_dim: 28672,
                vocab_size: 128000,
            },
            num_experts: 128,
            active_experts: 8,
            moe_layer_indices: (0..96).collect(),
            expert_intermediate_dim: 28672,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama2_70b_vram_estimate() {
        let cfg = ModelArchitecture::Dense(DenseConfig::llama2_70b());
        let vram = cfg.vram_full_fp16();
        // 70B × 2 bytes ≈ 140 GB
        assert!(vram > 130 * 1024 * 1024 * 1024);
        assert!(vram < 150 * 1024 * 1024 * 1024);
    }

    #[test]
    fn moe_recommends_expert_parallel() {
        let arch = ModelArchitecture::Moe(MoeConfig::mixtral_8x7b());
        // Single 24GB GPU
        let strategy = arch.recommend_shard_strategy(24 * 1024 * 1024 * 1024);
        assert_eq!(strategy, ShardStrategy::ExpertParallel);
    }

    #[test]
    fn small_model_recommends_spectral_k() {
        let arch = ModelArchitecture::Dense(DenseConfig::gpt2_small());
        // 24GB GPU is plenty for 124M
        let strategy = arch.recommend_shard_strategy(24 * 1024 * 1024 * 1024);
        assert_eq!(strategy, ShardStrategy::SpectralK);
    }
}
