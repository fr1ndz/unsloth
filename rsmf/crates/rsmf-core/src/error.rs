use std::fmt;

/// Unified error type for all RSMF operations.
#[derive(Debug, Clone)]
pub enum RsmfError {
    /// Dimension mismatch between tensors or strata.
    DimensionMismatch { expected: Vec<usize>, actual: Vec<usize> },
    /// SVD decomposition failed (singular matrix, numerical instability).
    DecompositionFailure(String),
    /// Stratum reconstruction produced invalid weights.
    ReconstructionError(String),
    /// Memory budget exceeded during training step.
    MemoryBudgetExceeded { requested_bytes: usize, budget_bytes: usize },
    /// Coherence below critical threshold — model may be diverging.
    CoherenceCollapse { layer: usize, value: f64, threshold: f64 },
}

impl fmt::Display for RsmfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {:?}, got {:?}", expected, actual)
            }
            Self::DecompositionFailure(msg) => write!(f, "decomposition failure: {}", msg),
            Self::ReconstructionError(msg) => write!(f, "reconstruction error: {}", msg),
            Self::MemoryBudgetExceeded { requested_bytes, budget_bytes } => {
                write!(f, "memory budget exceeded: {} > {} bytes", requested_bytes, budget_bytes)
            }
            Self::CoherenceCollapse { layer, value, threshold } => {
                write!(f, "coherence collapse at layer {}: {:.6} < {:.6}", layer, value, threshold)
            }
        }
    }
}

impl std::error::Error for RsmfError {}
