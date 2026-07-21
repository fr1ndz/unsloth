//! # rsmf-core
//!
//! Core mathematical types for Resonant Stratified Manifold Flow (RSMF).
//!
//! This crate defines the foundational abstractions:
//! - [`Stratum`] — spectral + tangential decomposition of a weight matrix
//! - [`ResonantTensor`] — Ψ = σ ⊗ U ⊗ Vᵀ with resonance metadata
//! - [`SpectralConfig`] — hyperparameters governing stratum dynamics
//! - [`CoherenceMetric`] — inter-stratum geometric alignment measure

pub mod stratum;
pub mod tensor;
pub mod config;
pub mod coherence;
pub mod error;

pub use stratum::Stratum;
pub use tensor::ResonantTensor;
pub use config::SpectralConfig;
pub use coherence::CoherenceMetric;
pub use error::RsmfError;
