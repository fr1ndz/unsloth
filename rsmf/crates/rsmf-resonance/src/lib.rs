//! # rsmf-resonance
//!
//! Resonance operators implementing the core RSMF mathematics:
//! - [`LocalResonance`] — stratified update with Ω regularizer
//! - [`ResonantBackward`] — backward channel Φ (replaces backprop)
//! - [`InterStratumCoupling`] — Γ coupling between adjacent layers
//! - [`CoherenceCorrector`] — global correction when coherence collapses

pub mod local_update;
pub mod backward_channel;
pub mod coupling;
pub mod corrector;
pub mod symplectic;

pub use local_update::LocalResonance;
pub use backward_channel::ResonantBackward;
pub use coupling::InterStratumCoupling;
pub use corrector::CoherenceCorrector;
pub use symplectic::{SymplecticModulator, SymplecticState};
