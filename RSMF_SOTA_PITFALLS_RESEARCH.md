# RSMF SOTA Pitfalls Research (2025-2026)

## Executive Summary
Research into current State-of-the-Art (SOTA) fine-tuning pitfalls reveals significant risks for the Resonant Stratified Model Framework (RSMF). Key findings indicate that **rank collapse** in adaptive LoRA is a pervasive issue driven by scaling laws, **ternary training** requires strict Quantization-Aware Training (QAT) from scratch rather than post-hoc quantization, and **orthogonal merging** suffers from numerical instability without Riemannian geometry corrections. While **Gradient SNR** is validated as a layer-selection metric (Spectrum), its use as a self-tuning hyperparameter mechanism remains experimental.

---

## 1. Adaptive-Rank LoRA Failure Modes
### Known Pitfalls
- **Rank Collapse:** Even with high nominal ranks, updates concentrate energy into a single dominant direction. RsLoRA (2025) identifies this as a "collapse of stable rank" where effective dimensionality << nominal rank.
- **Scaling Law Mismatch:** Standard `alpha/rank` scaling fails for adaptive ranks. When rank changes dynamically, fixed alpha causes gradient magnitude explosions or vanishing updates.
- **Federated Rank Collapse:** In distributed/adaptive settings, rank collapse accelerates due to aggregation of low-rank updates (raFLoRA, arXiv:2602.13486).

### RSMF Risk Mapping
- **Spectral-Adaptive Rank:** RSMF's dynamic rank adjustment is highly susceptible to stable rank collapse. 
- **Mitigation Required:** Implement **RsLoRA scaling** (`alpha / sqrt(rank)`) instead of linear scaling. Add explicit **stable rank regularization** (nuclear norm / spectral norm ratio) to prevent singular value concentration.

---

## 2. Ternary Weight Training Instability
### Known Pitfalls
- **Post-Hoc Quantization Failure:** Crushing FP16 models to ternary destroys performance. BitNet b1.58 and QuEST (arXiv:2502.05003) confirm ternary weights require **Quantization-Aware Training (QAT) from scratch**.
- **Gradient Explosion:** Straight-Through Estimators (STE) for ternary activations cause massive gradient variance. QuEST stabilizes this via **mixed-precision residual streams** and specialized normalization.
- **Convergence Sensitivity:** Ternary training requires 2-3x longer warmup and stricter LR schedules. Standard AdamW betas often fail; beta1=0.9/beta2=0.95 is insufficient.

### RSMF Risk Mapping
- **Native Ternary Adapters:** If RSMF adapters are ternary but base is FP16/BF16, expect severe gradient mismatch at adapter-base boundaries.
- **Mitigation Required:** Use **mixed-precision forward passes** (ternary adapters + FP16 residuals). Implement **gradient clipping per adapter module**. Validate convergence with QuEST-style stabilization before scaling.

---

## 3. Orthogonal Adapter Merge Numerical Problems
### Known Pitfalls
- **Euclidean Averaging Failure:** Simple weighted averaging of orthogonal adapters destroys orthogonality, causing catastrophic interference.
- **Numerical Drift:** SVD-based merging accumulates floating-point errors when merging >3 adapters. EigenLoRAx (Feb 2025) shows principal subspace recycling helps but doesn't eliminate drift.
- **Riemannian Solution Required:** OrthoFuse (arXiv:2604.05183) demonstrates that **Riemannian fusion on Stiefel manifold** is necessary for lossless merging. Euclidean approximations fail beyond 2-3 adapters.

### RSMF Risk Mapping
- **Orthogonal Subspaces for Lossless Merging:** RSMF's core merging claim is numerically fragile.
- **Mitigation Required:** Replace naive orthogonal projection with **OrthoFuse-style Riemannian optimization**. Add **orthogonality verification checkpoints** during merge. Test merging >5 adapters explicitly.

---

## 4. Resonance/Spectral Metrics Validity
### Known Pitfalls
- **Weak Downstream Correlation:** Spectral metrics (eigenvalue distribution, condition number) correlate poorly with downstream task performance unless calibrated per-task. "From Benchmarks to Skills" (arXiv:2507.20208) shows aggregate spectral scores miss skill-specific failures.
- **Resonance Undefined:** No established "resonance" metric exists in LLM eval literature. This appears to be an RSMF novel contribution without validation baseline.
- **Marchenko-Pastur Limits:** Spectrum paper (arXiv:2406.06623) uses MP distribution for SNR, but MP assumes i.i.d. Gaussian weights. Fine-tuned adapters violate this assumption.

### RSMF Risk Mapping
- **Resonance-Based Eval Metrics:** High risk of being a vanity metric.
- **Mitigation Required:** **Validate resonance against MMLU/HumanEval/Domain benchmarks immediately.** If correlation < 0.7, replace with proven spectral proxies (effective rank, stable rank). Document MP assumption violations.

---

## 5. Gradient SNR as Hparam Tuner Reliability
### Known Pitfalls
- **Layer Selection ≠ Hparam Tuning:** Spectrum validates SNR for *which layers to train*, not *what LR/rank to use*. Using SNR to tune rank/LR is unvalidated extrapolation.
- **Noise Floor Instability:** Gradient SNR estimates are noisy early in training. Self-tuning based on early SNR causes oscillatory hparam schedules.
- **Compute Overhead:** Per-layer SNR estimation adds 15-20% overhead, negating efficiency gains from adaptive rank.

### RSMF Risk Mapping
- **Self-Tuning Hparams via Gradient SNR:** Novel but risky.
- **Mitigation Required:** Use SNR only for **rank initialization**, not continuous tuning. Switch to **loss-curvature methods** (e.g., Hessian-free) for runtime hparam adaptation. Benchmark overhead vs. static tuned baselines.

---

## 6. Latest Fixes from Unsloth/Axolotl/LitGPT (2025-2026)
### Unsloth (v0.1.471-beta, Jun 2026)
- **No explicit rank collapse fix.** Focus remains on speed (12x MoE, Triton kernels).
- **QLoRA Stability:** Improved 4-bit quantization kernels reduce NaN rates but don't address rank collapse.
- **Recommendation:** RSMF must implement own rank stabilization; Unsloth won't provide it.

### Axolotl (v0.29.0, Feb 2026)
- **Stability improvements** for new model architectures but no adaptive-rank specific patches.
- **GRPO Support:** Relevant if RSMF extends to RLHF, but not core SFT.

### LitGPT / LLaMA-Factory (v0.9.4, Dec 2025)
- **OFT Support:** Orthogonal Fine-Tuning integrated, which may help RSMF's orthogonal subspace implementation.
- **Megatron-LM Backend:** Better distributed training stability for large-scale RSMF validation.

---

## Critical Action Items for RSMF Implementation
1. **Implement RsLoRA scaling** (`alpha/sqrt(rank)`) immediately for spectral-adaptive rank.
2. **Adopt OrthoFuse Riemannian merging** instead of naive orthogonal projection.
3. **Validate resonance metric** against 3+ standard benchmarks before committing.
4. **Use QuEST stabilization** for ternary adapter training (mixed-precision residuals).
5. **Restrict Gradient SNR** to initialization only; use curvature methods for runtime tuning.
6. **Add stable rank monitoring** to training logs to detect collapse early.

---

## Citations
- RsLoRA & Rank Collapse: emergentmind.com/topics/rank-stabilized-low-rank-adaptation-rslora (Sep 2025)
- raFLoRA Federated Rank Collapse: arxiv.org/pdf/2602.13486 (2026)
- QuEST 1-Bit Stability: arxiv.org/abs/2502.05003 (Feb 2025)
- OrthoFuse Riemannian Merging: arxiv.org/html/2604.05183 (2026)
- Spectrum SNR Training: api.emergentmind.com/papers/2406.06623 (2024)
- BitNet b1.58 Ternary: youngju.dev/blog/ai-papers/2026-03-06-ai-papers-bitnet-1bit-llm (Mar 2026)
- Unsloth v0.1.471: github.com/unslothai/unsloth/releases (Jun 2026)
- Axolotl v0.29.0: dev.to/ultraduneai/eval-003-fine-tuning-in-2026 (Mar 2026)
- LLaMA-Factory OFT: spheron.network/blog/axolotl-vs-unsloth-vs-torchtune (Mar 2026)
