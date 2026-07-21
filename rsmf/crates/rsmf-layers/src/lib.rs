//! # rsmf-layers
//!
//! Layer abstractions for RSMF forward pass with stratified recording.
//!
//! Key design principle: only ONE layer's working set resides in memory
//! at any given time. Activations are recorded in compressed spectral form,
//! not as full tensors.

pub mod forward;
pub mod activation_cache;
pub mod model;

pub use forward::StratifiedForward;
pub use activation_cache::ActivationCache;
pub use model::RsmfModel;
