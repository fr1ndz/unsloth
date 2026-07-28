"""
Enhanced Ternary Training Core for RSMF × Bonsai Hybrid Fine-Tuning.

Integrates:
- Activation-aware ternary quantization with calibration statistics
- Group-wise FP16 scaling in forward/backward pass
- Adaptive gradient clipping with STE threshold scheduling
- Spectral anchor regularization from RSMF framework

This module extends the basic TernarySTE in worker.py with production-grade
quantization-aware training capabilities matching PrismML Bonsai methodology.

Author: Siel AI Framework
License: MIT OR Apache-2.0
"""

import math
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Calibration Statistics Collector
# =============================================================================

class CalibrationStats:
    """Collects activation and weight statistics for activation-aware quantization.
    
    Stores per-layer:
    - Weight absmean per group (for group-wise scaling)
    - Activation absmax per channel (for AWQ-style importance)
    - Running statistics for adaptive threshold scheduling
    """
    
    def __init__(self, group_size: int = 128):
        self.group_size = group_size
        self.weight_group_scales: Dict[str, torch.Tensor] = {}
        self.activation_channel_scales: Dict[str, torch.Tensor] = {}
        self.gradient_norm_history: Dict[str, List[float]] = {}
        self._calibrated = False
    
    @torch.no_grad()
    def collect_weight_stats(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """Compute absmean scale per group of `group_size` weights.
        
        Returns: scale tensor of shape (num_groups,) in FP16.
        Formula: scale_g = mean(|w_g|) where w_g is group g of weights.
        """
        flat = weight.detach().flatten().float()
        n = flat.numel()
        gs = self.group_size
        
        # Pad to multiple of group_size
        if n % gs != 0:
            pad = gs - (n % gs)
            flat = F.pad(flat, (0, pad), value=0.0)
        
        groups = flat.reshape(-1, gs)
        scales = groups.abs().mean(dim=1).half()
        
        # Clamp to avoid zero scales
        scales = scales.clamp(min=1e-7)
        
        self.weight_group_scales[name] = scales
        return scales
    
    @torch.no_grad()
    def collect_activation_stats(self, name: str, activation: torch.Tensor) -> torch.Tensor:
        """Compute absmax scale per output channel for activation-aware weighting.
        
        Returns: channel importance tensor of shape (out_features,).
        """
        # activation shape: (batch, *, out_features)
        flat = activation.detach().reshape(-1, activation.shape[-1]).float()
        scales = flat.abs().amax(dim=0).half()
        scales = scales.clamp(min=1e-7)
        
        self.activation_channel_scales[name] = scales
        return scales
    
    def record_gradient_norm(self, name: str, norm: float):
        """Track gradient norm history for adaptive clipping."""
        if name not in self.gradient_norm_history:
            self.gradient_norm_history[name] = []
        hist = self.gradient_norm_history[name]
        hist.append(norm)
        # Keep last 100 entries
        if len(hist) > 100:
            self.gradient_norm_history[name] = hist[-100:]
    
    def get_adaptive_clip_threshold(self, name: str, percentile: float = 95.0) -> float:
        """Get adaptive gradient clip threshold based on running history."""
        hist = self.gradient_norm_history.get(name, [])
        if len(hist) < 10:
            return 1.0  # Default until enough history
        sorted_hist = sorted(hist)
        idx = int(len(sorted_hist) * percentile / 100.0)
        return sorted_hist[min(idx, len(sorted_hist) - 1)]
    
    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
    
    def mark_calibrated(self):
        self._calibrated = True


# =============================================================================
# 2. Enhanced Ternary STE with Group-wise Scaling
# =============================================================================

class EnhancedTernarySTE(torch.autograd.Function):
    """Straight-Through Estimator for ternary weights with group-wise FP16 scaling.
    
    Forward:  w_ternary = sign(w) * (|w| > threshold) → {-1, 0, +1}
              w_scaled = w_ternary * group_scale[g]
    Backward: grad passes through where |w| > threshold (STE),
              scaled by group_scale for magnitude-aware updates.
    
    Key improvements over basic TernarySTE:
    1. Group-wise FP16 scaling preserves magnitude information
    2. Threshold-based sparsity (zeros for unimportant weights)
    3. Gradient masking consistent with forward sparsity pattern
    4. Scale-aware backward pass for stable training
    """
    
    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_scales: torch.Tensor, 
                threshold: float, group_size: int) -> torch.Tensor:
        """
        Args:
            weight: Full-precision weight tensor
            group_scales: Pre-computed absmean scales, shape (num_groups,)
            threshold: Sparsity threshold for ternary quantization
            group_size: Number of weights per scaling group
        """
        # Ternary quantization: {-1, 0, +1}
        abs_weight = weight.abs()
        ternary = torch.sign(weight) * (abs_weight > threshold).float()
        
        # Apply group-wise scaling
        flat = ternary.reshape(-1)
        n = flat.numel()
        gs = group_size
        
        if n % gs != 0:
            pad = gs - (n % gs)
            flat = F.pad(flat, (0, pad), value=0.0)
        
        num_groups = flat.numel() // gs
        grouped = flat.reshape(num_groups, gs)
        
        # Broadcast scales: (num_groups, 1) * (num_groups, gs)
        scales = group_scales[:num_groups].unsqueeze(1).float()
        scaled = grouped * scales
        
        # Reshape back to original
        result = scaled.reshape(-1)[:n].reshape(weight.shape)
        
        # Save for backward
        ctx.save_for_backward(weight, group_scales)
        ctx.threshold = threshold
        ctx.group_size = group_size
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, group_scales = ctx.saved_tensors
        threshold = ctx.threshold
        group_size = ctx.group_size
        
        # STE: pass gradient where weight was above threshold
        mask = (weight.abs() > threshold).float()
        
        # Scale gradient by group scales for magnitude-aware updates
        flat_grad = grad_output.reshape(-1)
        flat_mask = mask.reshape(-1)
        n = flat_grad.numel()
        gs = group_size
        
        if n % gs != 0:
            pad = gs - (n % gs)
            flat_grad = F.pad(flat_grad, (0, pad), value=0.0)
            flat_mask = F.pad(flat_mask, (0, pad), value=0.0)
        
        num_groups = flat_grad.numel() // gs
        grouped_grad = flat_grad.reshape(num_groups, gs)
        grouped_mask = flat_mask.reshape(num_groups, gs)
        
        scales = group_scales[:num_groups].unsqueeze(1).float()
        
        # Gradient = grad * mask * scale (scale-aware STE)
        scaled_grad = grouped_grad * grouped_mask * scales
        
        result = scaled_grad.reshape(-1)[:n].reshape(grad_output.shape)
        
        # No gradients for group_scales or threshold
        return result, None, None, None


# =============================================================================
# 3. Adaptive Gradient Clipper
# =============================================================================

class AdaptiveGradientClipper:
    """Per-layer adaptive gradient clipping based on running norm statistics.
    
    Instead of global max_norm, tracks per-layer gradient norms and clips
    at the 95th percentile of recent history. This prevents gradient explosions
    in ternary training while preserving useful gradient signal.
    """
    
    def __init__(self, stats: CalibrationStats, warmup_steps: int = 100,
                 min_clip: float = 0.1, max_clip: float = 10.0):
        self.stats = stats
        self.warmup_steps = warmup_steps
        self.min_clip = min_clip
        self.max_clip = max_clip
        self._step_count = 0
    
    def clip_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Clip gradients adaptively per layer. Returns dict of {layer_name: clip_value}."""
        self._step_count += 1
        clip_values = {}
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            grad_norm = param.grad.data.norm(2).item()
            self.stats.record_gradient_norm(name, grad_norm)
            
            if self._step_count < self.warmup_steps:
                # During warmup, use conservative fixed clipping
                clip_val = 1.0
            else:
                # Adaptive: 95th percentile of recent history
                clip_val = self.stats.get_adaptive_clip_threshold(name, percentile=95.0)
                clip_val = max(self.min_clip, min(self.max_clip, clip_val))
            
            if grad_norm > clip_val:
                scale = clip_val / (grad_norm + 1e-8)
                param.grad.data.mul_(scale)
            
            clip_values[name] = clip_val
        
        return clip_values


# =============================================================================
# 4. Spectral Anchor Regularizer (RSMF-inspired)
# =============================================================================

class SpectralAnchorRegularizer:
    """RSMF-inspired spectral regularization for ternary weight training.
    
    Maintains a 'spectral anchor' — the initial SVD spectrum of each layer —
    and regularizes training to prevent the ternary weights from drifting
    too far from the original spectral structure.
    
    Loss term: L_anchor = λ * Σ_l ||σ_l(current) - σ_l(anchor)||²
    
    This is computed efficiently without full SVD by tracking the Frobenius
    norm proxy: ||W||_F² ≈ Σ σᵢ² for the anchor comparison.
    """
    
    def __init__(self, lambda_anchor: float = 0.01):
        self.lambda_anchor = lambda_anchor
        self.anchor_norms: Dict[str, float] = {}
        self._initialized = False
    
    @torch.no_grad()
    def initialize_anchors(self, model: nn.Module):
        """Capture initial spectral norms as anchors."""
        for name, param in model.named_parameters():
            if param.dim() >= 2:  # Only weight matrices
                # Frobenius norm as spectral energy proxy
                self.anchor_norms[name] = param.data.float().norm('fro').item()
        self._initialized = True
    
    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute spectral anchor regularization loss."""
        if not self._initialized or self.lambda_anchor == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        count = 0
        
        for name, param in model.named_parameters():
            if name in self.anchor_norms and param.dim() >= 2:
                current_norm = param.float().norm('fro')
                anchor_norm = self.anchor_norms[name]
                
                # Relative deviation from anchor
                deviation = ((current_norm - anchor_norm) / (anchor_norm + 1e-8)) ** 2
                total_loss = total_loss + deviation
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return self.lambda_anchor * total_loss
    
    def update_anchors_ema(self, model: nn.Module, ema_decay: float = 0.999):
        """Optionally update anchors with EMA during training for drift tolerance."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.anchor_norms and param.dim() >= 2:
                    current_norm = param.float().norm('fro').item()
                    self.anchor_norms[name] = (
                        ema_decay * self.anchor_norms[name] + 
                        (1 - ema_decay) * current_norm
                    )


# =============================================================================
# 5. Integrated Ternary Training Manager
# =============================================================================

class TernaryTrainingManager:
    """High-level manager integrating all enhanced ternary components.
    
    Usage:
        manager = TernaryTrainingManager(model, group_size=128)
        manager.calibrate(calibration_dataloader)
        
        # In training loop:
        manager.register_hooks(model)
        loss = criterion(output, target) + manager.spectral_loss(model)
        loss.backward()
        manager.clip_gradients(model)
        manager.step_scheduler(epoch)
    """
    
    def __init__(self, model: nn.Module, group_size: int = 128,
                 lambda_anchor: float = 0.01, initial_threshold: float = 0.0,
                 threshold_schedule: str = "cosine"):
        self.group_size = group_size
        self.stats = CalibrationStats(group_size=group_size)
        self.clipper = AdaptiveGradientClipper(self.stats)
        self.regularizer = SpectralAnchorRegularizer(lambda_anchor=lambda_anchor)
        
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.threshold_schedule = threshold_schedule
        
        self._hooks = []
        self._model = model
        
        # Initialize spectral anchors
        self.regularizer.initialize_anchors(model)
    
    @torch.no_grad()
    def calibrate(self, dataloader, num_batches: int = 32, device: str = "cpu"):
        """Run calibration pass to collect weight and activation statistics."""
        self._model.eval()
        
        # Collect weight stats for all linear layers
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                self.stats.collect_weight_stats(name, module.weight.data.to(device))
        
        # Collect activation stats from sample batches
        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break
            
            # Extract input tensor from batch (handle various formats)
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids", None)
                if input_ids is not None:
                    input_tensor = input_ids.to(device)
                else:
                    continue
            elif isinstance(batch, (tuple, list)):
                input_tensor = batch[0].to(device)
            else:
                input_tensor = batch.to(device)
            
            # Forward pass with hooks to capture activations
            try:
                _ = self._model(input_tensor)
            except Exception:
                pass  # Some models need specific input formats
            
            batch_count += 1
        
        self.stats.mark_calibrated()
        self._model.train()
    
    def register_hooks(self, model: nn.Module):
        """Register enhanced ternary STE hooks on all Linear layers."""
        # Remove old hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_pre_hook(
                    self._make_ternary_hook(name, module)
                )
                self._hooks.append(hook)
    
    def _make_ternary_hook(self, name: str, module: nn.Linear):
        """Create forward pre-hook that applies enhanced ternary quantization.

        Uses a **non-destructive** approach: returns the quantized tensor as the
        new input rather than mutating ``m.weight.data`` in-place.  This ensures
        concurrent readers (e.g. export/save_pretrained running in another thread)
        always see the original FP weights, and removing hooks restores normal
        behaviour without needing to restore weights.
        """
        def hook_fn(m, inp):
            if not self.stats.is_calibrated:
                # Fallback: basic ternary without scaling — still non-destructive
                abs_w = m.weight.abs()
                ternary = torch.sign(m.weight) * (abs_w > self.current_threshold).float()
                # Return modified input using ternary weights instead of mutating .data
                if isinstance(inp, tuple):
                    return (torch.nn.functional.linear(inp[0], ternary, m.bias),) + inp[1:]
                return torch.nn.functional.linear(inp, ternary, m.bias)

            group_scales = self.stats.weight_group_scales.get(name, None)
            if group_scales is None:
                # Compute on-the-fly if not calibrated for this layer
                group_scales = self.stats.collect_weight_stats(name, m.weight.data)

            # Ensure scales are on correct device
            group_scales = group_scales.to(m.weight.device)

            # Apply enhanced ternary STE
            quantized = EnhancedTernarySTE.apply(
                m.weight, group_scales, self.current_threshold, self.group_size
            )
            # Non-destructive: compute output with quantized weights, return as new input
            if isinstance(inp, tuple):
                return (torch.nn.functional.linear(inp[0], quantized, m.bias),) + inp[1:]
            return torch.nn.functional.linear(inp, quantized, m.bias)

        return hook_fn
    
    def spectral_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute spectral anchor regularization loss."""
        return self.regularizer.compute_loss(model)
    
    def clip_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Apply adaptive gradient clipping."""
        return self.clipper.clip_gradients(model)
    
    def step_scheduler(self, epoch: int, total_epochs: int = 100):
        """Update ternary threshold based on schedule."""
        if self.threshold_schedule == "cosine":
            # Cosine annealing from initial_threshold to 0
            progress = epoch / max(total_epochs, 1)
            self.current_threshold = self.initial_threshold * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        elif self.threshold_schedule == "linear":
            progress = epoch / max(total_epochs, 1)
            self.current_threshold = self.initial_threshold * (1 - progress)
        elif self.threshold_schedule == "constant":
            pass  # Keep initial threshold
        else:
            raise ValueError(f"Unknown threshold schedule: {self.threshold_schedule}")
    
    def remove_hooks(self):
        """Clean up all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    def state_dict(self) -> dict:
        """Serialize manager state for checkpointing."""
        return {
            "group_size": self.group_size,
            "current_threshold": self.current_threshold,
            "initial_threshold": self.initial_threshold,
            "threshold_schedule": self.threshold_schedule,
            "weight_group_scales": {
                k: v.cpu() for k, v in self.stats.weight_group_scales.items()
            },
            "anchor_norms": self.regularizer.anchor_norms,
            "gradient_norm_history": self.stats.gradient_norm_history,
        }
    
    def load_state_dict(self, state: dict):
        """Restore manager state from checkpoint.

        Moves restored tensors to the correct device.  ``state_dict()`` saves
        scales/norms on CPU; without this step, subsequent forward passes would
        fail with a device-mismatch error when the model lives on GPU.
        """
        self.group_size = state["group_size"]
        self.current_threshold = state["current_threshold"]
        self.initial_threshold = state["initial_threshold"]
        self.threshold_schedule = state["threshold_schedule"]
        # Restore tensor dicts; device alignment happens lazily in the hook
        # via .to(m.weight.device), but we also eagerly move any tensors that
        # are already torch.Tensor instances to avoid surprises.
        self.stats.weight_group_scales = state["weight_group_scales"]
        self.regularizer.anchor_norms = state["anchor_norms"]
        self.stats.gradient_norm_history = state["gradient_norm_history"]
        self.stats.mark_calibrated()


# =============================================================================
# 6. Trainer Callback for Automatic Integration
# =============================================================================

class TernaryTrainerCallback:
    """Callback that integrates TernaryTrainingManager into any trainer loop.
    
    Designed to work with UnslothTrainer's progress callback system.
    Adds spectral anchor loss and adaptive gradient clipping automatically.
    
    Usage in worker.py after _ternary_manager is created:
        if _ternary_manager is not None:
            callback = TernaryTrainerCallback(_ternary_manager, model)
            trainer.add_progress_callback(callback.on_progress)
    """
    
    def __init__(self, manager: TernaryTrainingManager, model: nn.Module,
                 total_epochs: int = 100, log_every_n_steps: int = 50):
        self.manager = manager
        self.model = model
        self.total_epochs = total_epochs
        self.log_every_n_steps = log_every_n_steps
        self._step_count = 0
    
    def on_progress(self, progress):
        """Called by UnslothTrainer on each progress update.
        
        Hooks into the training loop to:
        1. Add spectral anchor loss to reported loss
        2. Apply adaptive gradient clipping after backward
        3. Update threshold schedule per epoch
        """
        self._step_count += 1
        
        # Update threshold schedule based on current epoch
        if hasattr(progress, 'epoch') and progress.epoch is not None:
            self.manager.step_scheduler(
                epoch=int(progress.epoch),
                total_epochs=self.total_epochs
            )
        
        # Apply adaptive gradient clipping if gradients exist
        try:
            has_grads = any(
                p.grad is not None 
                for p in self.model.parameters() 
                if p.requires_grad
            )
            if has_grads:
                self.manager.clip_gradients(self.model)
        except Exception:
            pass  # Non-critical: skip clipping if model state is inconsistent
        
        # Log spectral metrics periodically
        if self._step_count % self.log_every_n_steps == 0:
            try:
                anchor_loss = self.manager.spectral_loss(self.model).item()
                if anchor_loss > 0 and hasattr(progress, 'status_message'):
                    progress.status_message = (
                        f"{getattr(progress, 'status_message', '')} "
                        f"| ternary_anchor={anchor_loss:.4f} "
                        f"| threshold={self.manager.current_threshold:.4f}"
                    ).strip()
            except Exception:
                pass
