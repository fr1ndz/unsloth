use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use rsmf_core::Stratum;

/// Specification for how a stratum is split across devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardSpec {
    /// Device index this shard resides on.
    pub device_id: usize,
    /// Range of singular component indices [start, end) held by this shard.
    pub k_range: (usize, usize),
    /// Layer index this shard belongs to.
    pub layer_id: usize,
    /// Total number of shards for this stratum.
    pub total_shards: usize,
}

/// A partial stratum residing on a single device.
///
/// Contains only the singular components assigned to this shard.
/// Reconstruction requires gathering all shards (or streaming).
#[derive(Debug, Clone)]
pub struct StratumShard {
    pub spec: ShardSpec,
    /// Subset of singular values for this shard's k-range.
    pub sigma: Array1<f64>,
    /// Subset of U columns for this shard's k-range.
    pub u_basis: Array2<f64>,
    /// Subset of V columns for this shard's k-range.
    pub v_basis: Array2<f64>,
}

impl StratumShard {
    /// Extract a shard from a full stratum according to spec.
    pub fn from_stratum(stratum: &Stratum, spec: ShardSpec) -> Self {
        let start = spec.k_range.0;
        let end = spec.k_range.1.min(stratum.rank());
        let _k_local = end - start;

        let sigma = Array1::from_vec(
            stratum.sigma.slice(ndarray::s![start..end]).to_vec()
        );
        let u_basis = stratum.u_basis.slice(ndarray::s![.., start..end]).to_owned();
        let v_basis = stratum.v_basis.slice(ndarray::s![.., start..end]).to_owned();

        Self { spec, sigma, u_basis, v_basis }
    }

    /// Reconstruct partial weight contribution: U_shard · diag(σ_shard) · V_shardᵀ
    /// This is additive — sum across all shards to get full W.
    pub fn partial_reconstruct(&self) -> Array2<f64> {
        let mut u_scaled = self.u_basis.clone();
        for (j, &s) in self.sigma.iter().enumerate() {
            for i in 0..u_scaled.nrows() {
                u_scaled[[i, j]] *= s;
            }
        }
        u_scaled.dot(&self.v_basis.t())
    }

    /// Memory footprint of this shard in bytes.
    pub fn memory_bytes(&self) -> usize {
        let elem = std::mem::size_of::<f64>();
        self.sigma.len() * elem + self.u_basis.len() * elem + self.v_basis.len() * elem
    }

    /// Number of singular components in this shard.
    pub fn local_rank(&self) -> usize {
        self.sigma.len()
    }
}

/// Split a stratum into N equal shards along k-dimension.
pub fn shard_stratum(stratum: &Stratum, num_shards: usize) -> Vec<StratumShard> {
    let k = stratum.rank();
    let chunk_size = (k + num_shards - 1) / num_shards;
    let mut shards = Vec::with_capacity(num_shards);

    for device_id in 0..num_shards {
        let start = device_id * chunk_size;
        let end = (start + chunk_size).min(k);
        if start >= k { break; }

        let spec = ShardSpec {
            device_id,
            k_range: (start, end),
            layer_id: stratum.layer_id,
            total_shards: num_shards,
        };
        shards.push(StratumShard::from_stratum(stratum, spec));
    }
    shards
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn shard_and_reconstruct_exact() {
        let sigma = array![4.0, 3.0, 2.0, 1.0];
        let u = Array2::eye(4);
        let v = Array2::eye(4);
        let stratum = Stratum::new(sigma, u, v, 0);
        let full_w = stratum.reconstruct();

        let shards = shard_stratum(&stratum, 2);
        assert_eq!(shards.len(), 2);

        // Sum partial reconstructions
        let mut reconstructed = Array2::zeros((4, 4));
        for shard in &shards {
            reconstructed += &shard.partial_reconstruct();
        }

        let diff = (&full_w - &reconstructed).mapv(|x| x.abs()).sum();
        assert!(diff < 1e-10, "Reconstruction error: {}", diff);
    }

    #[test]
    fn shard_memory_is_fraction_of_full() {
        let sigma = Array1::from_vec(vec![1.0; 64]);
        let u = Array2::eye(64);
        let v = Array2::eye(64);
        let stratum = Stratum::new(sigma, u, v, 0);

        let shards = shard_stratum(&stratum, 4);
        let shard_mem: usize = shards.iter().map(|s| s.memory_bytes()).sum();
        let full_mem = stratum.memory_bytes();

        // Total shard memory ≈ full memory (slight overhead from splitting)
        assert!(shard_mem <= full_mem + 256);
        // Each shard ≈ 1/4 of full
        assert!(shards[0].memory_bytes() < full_mem / 3);
    }
}
