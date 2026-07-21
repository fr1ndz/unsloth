//! # rsmf-distributed
//!
//! Distributed stratum sharding and pipeline parallelism for
//! training 100B-700B parameter models with RSMF.
//!
//! ## Architecture
//!
//! For models that exceed single-GPU VRAM even with layer cycling,
//! we shard the *stratum representation itself* across devices:
//!
//! - **Spectral Sharding**: U, σ, Vᵀ split across N GPUs along k-dimension
//! - **Pipeline Parallelism**: layers assigned to device groups with micro-batch overlap
//! - **VRAM Budget Allocator**: optimal shard topology given heterogeneous hardware
//!
//! ## Memory Model
//!
//! Each GPU holds only its shard of the current active layer's stratum.
//! Inter-device communication is O(batch × k/N) per boundary — negligible
//! vs O(batch × d) activation transfers in standard pipeline parallelism.

pub mod shard;
pub mod topology;
pub mod budget;
pub mod model_config;

pub use shard::{StratumShard, ShardSpec};
pub use topology::{DeviceTopology, PipelineStage};
pub use budget::VramBudgetAllocator;
pub use model_config::{ModelArchitecture, DenseConfig, MoeConfig, ShardStrategy};
