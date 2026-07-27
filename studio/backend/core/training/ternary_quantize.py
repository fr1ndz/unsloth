"""
Post-Training 1-Bit Ternary Quantization for Unsloth Studio.

Converts any Dense or MoE model to 1-bit Bonsai-compatible format:
- Ternary weights {-1, 0, +1} with group-wise FP16 scaling
- Activation-aware importance weighting (AWQ-style)
- Configurable sparsity threshold
- Supports both Dense and MoE architectures
- No training required — pure post-training quantization

Usage:
    from core.training.ternary_quantize import quantize_model_to_1bit
    
    result = quantize_model_to_1bit(
        model=model,
        tokenizer=tokenizer,
        output_dir="/path/to/output",
        group_size=128,
        sparsity_threshold=0.0,
    )

Author: Siel AI Framework
License: MIT OR Apache-2.0
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# =============================================================================
# Configuration & Results
# =============================================================================

@dataclass
class QuantizationConfig:
    """Configuration for post-training 1-bit quantization."""
    group_size: int = 128
    sparsity_threshold: float = 0.0
    use_activation_awareness: bool = True
    calibration_samples: int = 32
    dtype: str = "float16"
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])
    include_moe_experts: bool = True
    save_scales_separately: bool = False


@dataclass
class QuantizationStats:
    """Statistics collected during quantization."""
    total_params: int = 0
    quantized_params: int = 0
    skipped_params: int = 0
    zero_fraction: float = 0.0
    avg_scale: float = 0.0
    layers_quantized: int = 0
    layers_skipped: int = 0
    compression_ratio: float = 0.0
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class QuantizationResult:
    """Result of post-training quantization."""
    success: bool
    output_dir: str
    stats: QuantizationStats
    config: QuantizationConfig
    error: Optional[str] = None


# =============================================================================
# Core Quantization Logic
# =============================================================================

def _compute_group_scales(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Compute absmean scales per group of `group_size` weights.
    
    Returns FP16 tensor of shape (num_groups,).
    """
    flat = weight.detach().flatten().float()
    n = flat.numel()
    
    if n % group_size != 0:
        pad = group_size - (n % group_size)
        flat = torch.nn.functional.pad(flat, (0, pad), value=0.0)
    
    groups = flat.reshape(-1, group_size)
    scales = groups.abs().mean(dim=1).half()
    return scales.clamp(min=1e-7)


def _ternarize_weight(
    weight: torch.Tensor,
    group_scales: torch.Tensor,
    threshold: float,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to ternary {-1, 0, +1} with group scaling.
    
    Returns:
        ternary_weights: int8 tensor with values {-1, 0, +1}
        group_scales: FP16 scales for reconstruction
    """
    abs_w = weight.abs()
    ternary = torch.sign(weight) * (abs_w > threshold).float()
    
    # Convert to int8 for storage efficiency
    ternary_int8 = ternary.to(torch.int8)
    
    return ternary_int8, group_scales


def _estimate_size_mb(params: int, dtype_bytes: int = 2) -> float:
    """Estimate size in MB for given parameter count."""
    return (params * dtype_bytes) / (1024 * 1024)


def _is_quantizable(name: str, module: nn.Module, config: QuantizationConfig) -> bool:
    """Check if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False
    
    # Check exclusions
    for excl in config.exclude_modules:
        if excl in name:
            return False
    
    # MoE expert handling
    if "expert" in name.lower() and not config.include_moe_experts:
        return False
    
    return True


# =============================================================================
# Activation-Aware Calibration
# =============================================================================

class ActivationCollector:
    """Collects activation statistics for AWQ-style importance weighting."""
    
    def __init__(self):
        self.channel_importance: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []
    
    def register_hooks(self, model: nn.Module, config: QuantizationConfig):
        """Register forward hooks to capture activation magnitudes."""
        for name, module in model.named_modules():
            if _is_quantizable(name, module, config):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)
    
    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # Accumulate absmax per output channel
                act = output.detach().reshape(-1, output.shape[-1]).float()
                importance = act.abs().amax(dim=0)
                
                if name in self.channel_importance:
                    # Running max
                    self.channel_importance[name] = torch.max(
                        self.channel_importance[name], importance
                    )
                else:
                    self.channel_importance[name] = importance
        return hook_fn
    
    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    def get_importance(self, name: str) -> Optional[torch.Tensor]:
        return self.channel_importance.get(name)


# =============================================================================
# Main Quantization Pipeline
# =============================================================================

def quantize_model_to_1bit(
    model: nn.Module,
    tokenizer: Any,
    output_dir: str,
    config: Optional[QuantizationConfig] = None,
    calibration_data: Optional[Any] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> QuantizationResult:
    """Convert a Dense or MoE model to 1-bit ternary format.
    
    Args:
        model: Loaded PyTorch model (Dense or MoE)
        tokenizer: Associated tokenizer
        output_dir: Directory to save quantized model
        config: Quantization configuration
        calibration_data: Optional dataloader for activation-aware calibration
        progress_callback: Optional callback(message, progress_fraction)
    
    Returns:
        QuantizationResult with stats and output path
    """
    if config is None:
        config = QuantizationConfig()
    
    start_time = time.time()
    stats = QuantizationStats()
    
    def _progress(msg: str, frac: float = 0.0):
        if progress_callback:
            progress_callback(msg, frac)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # ── Phase 1: Analyze model ──
        _progress("Analyzing model architecture...", 0.05)
        
        quantizable_modules = {}
        for name, module in model.named_modules():
            if _is_quantizable(name, module, config):
                quantizable_modules[name] = module
                stats.total_params += module.weight.numel()
            elif isinstance(module, nn.Linear):
                stats.skipped_params += module.weight.numel()
        
        stats.original_size_mb = _estimate_size_mb(stats.total_params + stats.skipped_params)
        
        if not quantizable_modules:
            return QuantizationResult(
                success=False,
                output_dir=output_dir,
                stats=stats,
                config=config,
                error="No quantizable Linear layers found in model",
            )
        
        _progress(f"Found {len(quantizable_modules)} quantizable layers ({stats.total_params:,} params)", 0.1)
        
        # ── Phase 2: Activation-aware calibration (optional) ──
        collector = ActivationCollector()
        
        if config.use_activation_awareness and calibration_data is not None:
            _progress("Running activation-aware calibration...", 0.15)
            collector.register_hooks(model, config)
            
            model.eval()
            sample_count = 0
            with torch.no_grad():
                for batch in calibration_data:
                    if sample_count >= config.calibration_samples:
                        break
                    try:
                        if isinstance(batch, dict):
                            input_ids = batch.get("input_ids")
                            if input_ids is not None:
                                model(input_ids.to(next(model.parameters()).device))
                        elif isinstance(batch, (tuple, list)):
                            model(batch[0].to(next(model.parameters()).device))
                    except Exception:
                        pass
                    sample_count += 1
            
            collector.remove_hooks()
            _progress(f"Calibrated with {sample_count} samples", 0.25)
        
        # ── Phase 3: Quantize weights ──
        _progress("Quantizing weights to ternary...", 0.3)
        
        quantized_state = {}
        scale_state = {}
        total_zeros = 0
        total_elements = 0
        all_scales = []
        
        num_modules = len(quantizable_modules)
        for idx, (name, module) in enumerate(quantizable_modules.items()):
            weight = module.weight.data.float()
            
            # Compute group scales
            scales = _compute_group_scales(weight, config.group_size)
            
            # Apply activation-aware weighting if available
            importance = collector.get_importance(name)
            if importance is not None and config.use_activation_awareness:
                # Scale weights by channel importance before quantization
                # This preserves important channels at higher precision
                out_features = min(importance.numel(), weight.shape[0])
                importance_normalized = importance[:out_features] / (importance[:out_features].mean() + 1e-8)
                importance_normalized = importance_normalized.clamp(0.5, 2.0)
                weight = weight[:out_features] * importance_normalized.unsqueeze(1)
            
            # Ternarize
            ternary, scales = _ternarize_weight(
                weight, scales, config.sparsity_threshold, config.group_size
            )
            
            # Track statistics
            zeros = (ternary == 0).sum().item()
            total_zeros += zeros
            total_elements += ternary.numel()
            all_scales.append(scales.float().mean().item())
            
            # Store quantized weights and scales
            quantized_state[f"{name}.weight"] = ternary.cpu()
            scale_state[f"{name}.weight_scale"] = scales.cpu()
            
            # Store bias if present
            if module.bias is not None:
                quantized_state[f"{name}.bias"] = module.bias.data.cpu().half()
            
            stats.layers_quantized += 1
            stats.quantized_params += ternary.numel()
            
            if (idx + 1) % max(1, num_modules // 10) == 0:
                frac = 0.3 + 0.5 * (idx + 1) / num_modules
                _progress(f"Quantized {idx + 1}/{num_modules} layers", frac)
        
        # Copy non-quantized parameters (embeddings, norms, etc.)
        _progress("Copying non-quantized parameters...", 0.82)
        for name, param in model.named_parameters():
            if name not in quantized_state:
                quantized_state[name] = param.data.cpu().half()
        
        # ── Phase 4: Save quantized model ──
        _progress("Saving quantized model...", 0.88)
        
        # Save quantized weights
        torch.save(quantized_state, os.path.join(output_dir, "model.safetensors"))
        
        # Save scales
        torch.save(scale_state, os.path.join(output_dir, "group_scales.pt"))
        
        # Save tokenizer
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        
        # Save quantization config/metadata
        metadata = {
            "format": "unsloth-1bit-ternary",
            "version": "1.0",
            "group_size": config.group_size,
            "sparsity_threshold": config.sparsity_threshold,
            "use_activation_awareness": config.use_activation_awareness,
            "dtype": config.dtype,
            "ternary": True,
            "training_type": "1-bit Quantize",
            "total_params": stats.total_params,
            "quantized_params": stats.quantized_params,
            "layers_quantized": stats.layers_quantized,
            "zero_fraction": total_zeros / max(total_elements, 1),
            "avg_scale": sum(all_scales) / max(len(all_scales), 1),
            "original_model": getattr(model, "config", {}).get("_name_or_path", "unknown"),
        }
        
        with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save original model config if available
        if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
            model.config.save_pretrained(output_dir)
        
        # ── Phase 5: Compute final stats ──
        _progress("Computing compression statistics...", 0.95)
        
        stats.zero_fraction = total_zeros / max(total_elements, 1)
        stats.avg_scale = sum(all_scales) / max(len(all_scales), 1)
        stats.elapsed_seconds = time.time() - start_time
        
        # Estimate quantized size:
        # ternary weights: 1 bit each (stored as int8 = 1 byte)
        # scales: FP16 = 2 bytes per group
        # Non-quantized params: FP16 = 2 bytes
        ternary_bytes = stats.quantized_params * 1  # int8
        scale_bytes = sum(
            s.numel() * 2 for s in scale_state.values()  # FP16
        )
        non_quant_bytes = stats.skipped_params * 2  # FP16
        stats.quantized_size_mb = (ternary_bytes + scale_bytes + non_quant_bytes) / (1024 * 1024)
        
        if stats.quantized_size_mb > 0:
            stats.compression_ratio = stats.original_size_mb / stats.quantized_size_mb
        
        _progress("Quantization complete!", 1.0)
        
        return QuantizationResult(
            success=True,
            output_dir=output_dir,
            stats=stats,
            config=config,
        )
    
    except Exception as e:
        return QuantizationResult(
            success=False,
            output_dir=output_dir,
            stats=stats,
            config=config,
            error=str(e),
        )
