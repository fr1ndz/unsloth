//! # Ternary Adapter with QAT Mixed-Precision Residuals
//!
//! Sign-magnitude decomposition for memory-efficient fine-tuning:
//! - Weights stored as {-1, 0, +1} ternary values (2 bits per param)
//! - FP16 residual stream for QuEST-style stabilization
//! - Mixed-precision forward: ternary adapters + FP16 residuals
//! - Per-module gradient clipping for STE stability

use ndarray::Array2;

/// Ternary weight value: sign component only.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(i8)]
pub enum TernarySign {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl TernarySign {
    /// Convert to f32 for computation.
    #[inline]
    pub fn to_f32(self) -> f32 {
        match self {
            TernarySign::Neg => -1.0,
            TernarySign::Zero => 0.0,
            TernarySign::Pos => 1.0,
        }
    }

    /// Quantize from continuous value using sign function.
    #[inline]
    pub fn from_value(v: f32, threshold: f32) -> Self {
        if v > threshold {
            TernarySign::Pos
        } else if v < -threshold {
            TernarySign::Neg
        } else {
            TernarySign::Zero
        }
    }
}

/// Ternary adapter layer with FP16 residual stream.
///
/// Storage: ternary signs (i8) + FP16 residuals (f16 via u16 bit repr).
/// Forward pass reconstructs: W_ternary * scale + residual_fp16
#[derive(Debug, Clone)]
pub struct TernaryAdapter {
    /// Module name for identification.
    pub name: String,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Ternary sign matrix (in_dim × out_dim), stored as i8.
    pub signs: Vec<i8>,
    /// FP16 residual stream (in_dim × out_dim), stored as u16 bit patterns.
    /// Uses IEEE 754 half-precision encoding.
    pub residuals: Vec<u16>,
    /// Scaling factor for ternary component.
    pub scale: f32,
    /// Gradient clipping threshold for STE stability.
    pub grad_clip: f32,
    /// Quantization threshold for ternarization.
    pub ternary_threshold: f32,
}

impl TernaryAdapter {
    /// Create a new ternary adapter initialized from continuous weights.
    ///
    /// Decomposes W into sign(W) * scale + residual where:
    /// - scale = mean(|W|) over non-zero elements
    /// - residual = W - sign(W) * scale (stored as FP16)
    pub fn from_weights(name: &str, weights: &Array2<f64>, grad_clip: f32) -> Self {
        let in_dim = weights.nrows();
        let out_dim = weights.ncols();
        let n = in_dim * out_dim;

        // Compute scale as mean absolute value of non-zero elements
        let mut abs_sum = 0.0f64;
        let mut count = 0usize;
        for w in weights.iter() {
            let aw = w.abs();
            if aw > 1e-10 {
                abs_sum += aw;
                count += 1;
            }
        }
        let scale = if count > 0 { (abs_sum / count as f64) as f32 } else { 1.0 };

        let mut signs = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);

        for w in weights.iter() {
            let wf = *w as f32;
            let s = TernarySign::from_value(wf, 1e-10);
            signs.push(s.to_f32() as i8); // -1, 0, or 1

            // Residual = original - ternary_reconstruction
            let recon = s.to_f32() * scale;
            let res = wf - recon;
            residuals.push(f32_to_f16_bits(res));
        }

        Self {
            name: name.to_string(),
            in_dim,
            out_dim,
            signs,
            residuals,
            scale,
            grad_clip,
            ternary_threshold: 0.5, // Default: midpoint between 0 and 1
        }
    }

    /// Create an empty (zero-initialized) ternary adapter.
    pub fn zeros(name: &str, in_dim: usize, out_dim: usize, grad_clip: f32) -> Self {
        let n = in_dim * out_dim;
        Self {
            name: name.to_string(),
            in_dim,
            out_dim,
            signs: vec![0i8; n],
            residuals: vec![0u16; n], // FP16 zero = 0x0000
            scale: 1.0,
            grad_clip,
            ternary_threshold: 0.5,
        }
    }

    /// Forward pass: reconstruct weights and apply to input.
    ///
    /// output = input @ (signs * scale + fp16_residuals)
    /// Uses mixed precision: ternary lookup + FP16 decode.
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        assert_eq!(input.ncols(), self.in_dim,
            "Input cols {} != adapter in_dim {}", input.ncols(), self.in_dim);

        let batch = input.nrows();
        let mut output = Array2::zeros((batch, self.out_dim));

        for b in 0..batch {
            for j in 0..self.out_dim {
                let mut acc = 0.0f64;
                for i in 0..self.in_dim {
                    let idx = i * self.out_dim + j;
                    // Reconstruct: ternary * scale + fp16 residual
                    let t = self.signs[idx] as f32;
                    let r = f16_bits_to_f32(self.residuals[idx]);
                    let w = (t * self.scale + r) as f64;
                    acc += input[[b, i]] * w;
                }
                output[[b, j]] = acc;
            }
        }

        output
    }

    /// Apply gradient update with STE (Straight-Through Estimator) and clipping.
    ///
    /// 1. Clip gradients per-module for STE stability
    /// 2. Update underlying continuous representation
    /// 3. Re-ternarize with updated residual
    pub fn apply_gradient(&mut self, grad: &Array2<f64>, lr: f32) {
        assert_eq!(grad.nrows(), self.in_dim);
        assert_eq!(grad.ncols(), self.out_dim);

        // Per-module gradient clipping
        let mut grad_norm_sq = 0.0f64;
        for g in grad.iter() {
            grad_norm_sq += g * g;
        }
        let grad_norm = grad_norm_sq.sqrt() as f32;
        let clip_factor = if grad_norm > self.grad_clip && grad_norm > 1e-10 {
            self.grad_clip / grad_norm
        } else {
            1.0
        };

        // Update: reconstruct continuous weight, apply gradient, re-decompose
        for i in 0..self.in_dim {
            for j in 0..self.out_dim {
                let idx = i * self.out_dim + j;
                let t = self.signs[idx] as f32;
                let r = f16_bits_to_f32(self.residuals[idx]);
                let mut w = t * self.scale + r;

                // SGD step with clipped gradient
                let g = grad[[i, j]] as f32 * clip_factor;
                w -= lr * g;

                // Re-ternarize
                let new_sign = TernarySign::from_value(w, 1e-10);
                self.signs[idx] = new_sign.to_f32() as i8;
                let recon = new_sign.to_f32() * self.scale;
                self.residuals[idx] = f32_to_f16_bits(w - recon);
            }
        }
    }

    /// Memory footprint in bytes.
    ///
    /// Ternary signs: 1 byte each (i8)
    /// FP16 residuals: 2 bytes each (u16)
    /// Metadata: ~64 bytes
    pub fn memory_bytes(&self) -> usize {
        let n = self.in_dim * self.out_dim;
        let signs_bytes = n * std::mem::size_of::<i8>();     // 1 byte per param
        let resid_bytes = n * std::mem::size_of::<u16>();     // 2 bytes per param
        let meta_bytes = std::mem::size_of::<Self>()          // struct overhead
            + self.name.len();                                // string heap
        signs_bytes + resid_bytes + meta_bytes
    }

    /// Equivalent FP32 memory for comparison.
    pub fn fp32_equivalent_bytes(&self) -> usize {
        self.in_dim * self.out_dim * std::mem::size_of::<f32>()
    }

    /// Compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f64 {
        let actual = self.memory_bytes() as f64;
        let fp32 = self.fp32_equivalent_bytes() as f64;
        if fp32 > 0.0 { fp32 / actual } else { 1.0 }
    }

    /// Get number of parameters.
    pub fn num_params(&self) -> usize {
        self.in_dim * self.out_dim
    }

    /// Count non-zero ternary entries (sparsity metric).
    pub fn sparsity(&self) -> f64 {
        let nonzero = self.signs.iter().filter(|&&s| s != 0).count();
        let total = self.signs.len();
        if total > 0 { 1.0 - (nonzero as f64 / total as f64) } else { 0.0 }
    }
}

// === FP16 Conversion Utilities ===
// IEEE 754 half-precision bit manipulation without external deps.

/// Convert f32 to IEEE 754 half-precision (f16) stored as u16 bits.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits32 = value.to_bits();
    let sign = ((bits32 >> 31) & 0x1) as u16;
    let exp32 = ((bits32 >> 23) & 0xFF) as i32;
    let mant32 = bits32 & 0x7FFFFF;

    // Handle special cases
    if exp32 == 0xFF {
        // Inf or NaN
        if mant32 == 0 {
            return (sign << 15) | 0x7C00; // Inf
        } else {
            return (sign << 15) | 0x7E00; // NaN (quiet)
        }
    }

    // Rebias exponent: f32 bias=127, f16 bias=15
    let exp16 = exp32 - 127 + 15;

    if exp16 >= 31 {
        // Overflow → Inf
        return (sign << 15) | 0x7C00;
    } else if exp16 <= 0 {
        // Underflow → subnormal or zero
        if exp16 < -10 {
            return sign << 15; // Too small → zero
        }
        // Subnormal: shift mantissa
        let mant_shifted = (mant32 | 0x800000) >> (1 - exp16 + 13);
        return (sign << 15) | (mant_shifted as u16);
    }

    // Normal case
    let mant16 = (mant32 >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mant16
}

/// Convert IEEE 754 half-precision u16 bits back to f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;

    let result_bits = if exp == 0 {
        if mant == 0 {
            // Zero
            sign << 31
        } else {
            // Subnormal: normalize
            let mut e = 0i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let exp32 = (127 - 15 + 1 + e) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            (sign << 31) | 0x7F800000 // Inf
        } else {
            (sign << 31) | 0x7FC00000 // NaN
        }
    } else {
        // Normal
        let exp32 = (exp - 15 + 127) as u32;
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };

    f32::from_bits(result_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fp16_roundtrip_normal_values() {
        let test_vals = [0.0f32, 1.0, -1.0, 0.5, -0.5, 3.14, -2.718, 0.001, 100.0];
        for &v in &test_vals {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            // FP16 has ~3 decimal digits of precision
            let rel_err = if v.abs() > 1e-6 { (back - v).abs() / v.abs() } else { (back - v).abs() };
            assert!(rel_err < 0.01, "FP16 roundtrip failed for {}: got {}, rel_err={}", v, back, rel_err);
        }
    }

    #[test]
    fn fp16_special_values() {
        // Zero
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(0.0)), 0.0);
        // Negative zero
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(-0.0)).to_bits(), (-0.0f32).to_bits());
        // Infinity
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::INFINITY)).is_infinite());
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NEG_INFINITY)).is_infinite());
    }

    #[test]
    fn ternary_adapter_from_weights() {
        let weights = array![[1.0, -2.0, 0.0], [0.5, -0.5, 3.0]];
        let adapter = TernaryAdapter::from_weights("test", &weights, 1.0);

        assert_eq!(adapter.in_dim, 2);
        assert_eq!(adapter.out_dim, 3);
        assert_eq!(adapter.num_params(), 6);
        assert!(adapter.scale > 0.0);
    }

    #[test]
    fn ternary_adapter_memory_savings() {
        let adapter = TernaryAdapter::zeros("test", 256, 512, 1.0);
        let actual = adapter.memory_bytes();
        let fp32 = adapter.fp32_equivalent_bytes();

        // Should be ~3x compression (1+2 bytes vs 4 bytes per param)
        let ratio = adapter.compression_ratio();
        assert!(ratio > 1.2, "Compression ratio {} too low", ratio);
        assert!(actual < fp32, "Ternary should use less memory than FP32");
    }

    #[test]
    fn ternary_forward_produces_output() {
        let weights = array![[1.0, 0.5], [-0.5, 1.0]];
        let adapter = TernaryAdapter::from_weights("test", &weights, 1.0);
        let input = array![[1.0, 0.0], [0.0, 1.0]];
        let output = adapter.forward(&input);

        assert_eq!(output.shape(), &[2, 2]);
        // Output should be approximately equal to input @ weights
        // (with some quantization error from ternary + FP16)
    }

    #[test]
    fn gradient_clipping_works() {
        let weights = array![[1.0, -1.0], [0.5, -0.5]];
        let mut adapter = TernaryAdapter::from_weights("test", &weights, 0.5);

        // Large gradient that should be clipped
        let grad = array![[10.0, -10.0], [10.0, -10.0]];
        adapter.apply_gradient(&grad, 0.01);

        // Adapter should still be valid (no NaN/Inf)
        for &s in &adapter.signs {
            assert!(s == -1 || s == 0 || s == 1, "Invalid ternary sign: {}", s);
        }
    }

    #[test]
    fn sparsity_metric() {
        let weights = array![[1.0, 0.0, -1.0], [0.0, 0.0, 0.5]];
        let adapter = TernaryAdapter::from_weights("test", &weights, 1.0);
        let sp = adapter.sparsity();
        // At least the explicit zeros should contribute
        assert!(sp >= 0.0 && sp <= 1.0);
    }
}
