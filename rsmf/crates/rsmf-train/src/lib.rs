//! # rsmf-train
//!
//! RSMF training loop implementing memory-safe layer cycling,
//! coherence checks, and full-parameter updates on 4GB VRAM.
//!
//! This is the main entry point for training models with RSMF.

pub mod trainer;
pub mod loss;
pub mod schedule;

pub use trainer::RsmfTrainer;
pub use loss::ResonantLoss;
pub use schedule::CoherenceSchedule;
