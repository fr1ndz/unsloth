//! # rsmf-moe
//!
//! Mixture-of-Experts support for RSMF training.
//!
//! ## Key Design
//!
//! - Each expert has its own [`ExpertStratum`] with independent spectral decomposition
//! - A shared [`RoutingStratum`] handles token→expert assignment
//! - Hierarchical coherence: intra-expert → inter-expert → inter-layer
//! - Expert load balancing via spectral energy equalization
//! - Only active experts' strata loaded into VRAM per token batch

pub mod expert;
pub mod routing;
pub mod hierarchical_coherence;
pub mod moe_model;

pub use expert::ExpertStratum;
pub use routing::RoutingStratum;
pub use hierarchical_coherence::HierarchicalCoherence;
pub use moe_model::MoeRsmfModel;
