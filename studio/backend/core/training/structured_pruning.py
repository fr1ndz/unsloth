"""
Structured Pruning for LLMs — Training-Free Layer & Neuron Removal.

Implements importance-based structured pruning:
- Per-layer importance scoring via weight magnitude × input activation
- Configurable pruning ratio (0-100%)
- Preview mode: analyze without modifying model
- Supports Dense and MoE architectures

Based on LLM-Pruner (Ma et al., 2023) and Wanda (Sun et al., 2023).

Usage:
    from core.training.structured_pruning import StructuredPruner
    
    pruner = StructuredPruner(model, tokenizer)
    analysis = pruner.analyze(ratio=0.3)
    pruned_model = pruner.prune(ratio=0.3)

Author: Siel AI Framework
License: MIT OR Apache-2.0
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LayerImportance:
    """Importance score for a single layer/module."""
    name: str
    module_type: str  # "linear", "attention", "mlp", "expert"
    num_params: int
    importance_score: float
    weight_magnitude: float  # mean |W|
    activation_magnitude: float  # mean |X| if available
    prune_candidate: bool = False
    cumulative_ratio: float = 0.0  # % of total params removed if this layer is pruned


@dataclass
class PruningAnalysis:
    """Result of pruning analysis (preview mode)."""
    total_params: int
    target_pruned_params: int
    target_ratio: float
    actual_pruned_params: int
    actual_ratio: float
    layers: List[LayerImportance]
    layers_to_prune: List[str]
    estimated_speedup: float
    elapsed_seconds: float = 0.0


@dataclass
class PruningConfig:
    """Configuration for structured pruning."""
    ratio: float = 0.3  # 0.0 to 1.0
    method: str = "magnitude"  # "magnitude", "wanda", "random"
    granularity: str = "layer"  # "layer", "neuron"
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])
    include_moe_experts: bool = True
    calibration_samples: int = 16


# =============================================================================
# Importance Scoring
# =============================================================================

def _compute_weight_importance(weight: torch.Tensor) -> float:
    """Compute L1-norm based importance for a weight matrix."""
    return weight.abs().mean().item()


def _compute_wanda_importance(
    weight: torch.Tensor,
    input_activation: Optional[torch.Tensor] = None,
) -> float:
    """Wanda-style importance: |W| * |X| per column, then mean.
    
    If no activation available, falls back to pure magnitude.
    """
    mag = weight.abs()
    if input_activation is not None:
        # input_activation shape: (batch, seq, hidden) or (batch, hidden)
        act = input_activation.detach().reshape(-1, input_activation.shape[-1]).float()
        act_mag = act.abs().mean(dim=0)  # (hidden,)
        # Broadcast: weight is (out, in), act_mag is (in,)
        if act_mag.numel() == weight.shape[1]:
            importance = (mag * act_mag.unsqueeze(0)).mean()
        else:
            importance = mag.mean()
    else:
        importance = mag.mean()
    return importance.item()


def _count_params(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# =============================================================================
# Activation Collector
# =============================================================================

class ActivationCollector:
    """Collects input activations for Wanda-style importance scoring."""
    
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []
    
    def register_hooks(self, model: nn.Module, config: PruningConfig):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not any(e in name for e in config.exclude_modules):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
    
    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                act = input[0].detach()
                if name in self.activations:
                    # Running mean to save memory
                    old = self.activations[name]
                    self.activations[name] = (old + act.float().mean(dim=0)) / 2.0
                else:
                    self.activations[name] = act.float().mean(dim=0)
        return hook_fn
    
    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        return self.activations.get(name)


# =============================================================================
# Main Pruner
# =============================================================================

class StructuredPruner:
    """Training-free structured pruning for LLMs."""
    
    def __init__(self, model: nn.Module, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self._collector = ActivationCollector()
    
    def analyze(
        self,
        config: Optional[PruningConfig] = None,
        calibration_data: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PruningAnalysis:
        """Analyze model for pruning without modifying it.
        
        Returns PruningAnalysis with layer importance scores and pruning plan.
        """
        import time
        start = time.time()
        
        if config is None:
            config = PruningConfig()
        
        def _progress(msg, frac=0.0):
            if progress_callback:
                progress_callback(msg, frac)
        
        _progress("Collecting activation statistics...", 0.1)
        
        # Collect activations if calibration data provided
        if calibration_data is not None and config.method == "wanda":
            self._collector.register_hooks(self.model, config)
            self.model.eval()
            sample_count = 0
            with torch.no_grad():
                for batch in calibration_data:
                    if sample_count >= config.calibration_samples:
                        break
                    try:
                        if isinstance(batch, dict):
                            ids = batch.get("input_ids")
                            if ids is not None:
                                self.model(ids.to(next(self.model.parameters()).device))
                        elif isinstance(batch, (tuple, list)):
                            self.model(batch[0].to(next(self.model.parameters()).device))
                    except Exception:
                        pass
                    sample_count += 1
            self._collector.remove_hooks()
        
        _progress("Scoring layer importance...", 0.4)
        
        # Score each linear layer
        layers: List[LayerImportance] = []
        total_params = 0
        
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(e in name for e in config.exclude_modules):
                continue
            if "expert" in name.lower() and not config.include_moe_experts:
                continue
            
            n_params = _count_params(module)
            total_params += n_params
            
            # Determine module type from name
            module_type = "linear"
            name_lower = name.lower()
            if "attn" in name_lower or "attention" in name_lower:
                module_type = "attention"
            elif "mlp" in name_lower or "ffn" in name_lower or "feed" in name_lower:
                module_type = "mlp"
            elif "expert" in name_lower:
                module_type = "expert"
            
            # Compute importance
            weight_mag = _compute_weight_importance(module.weight.data)
            act = self._collector.get_activation(name)
            
            if config.method == "wanda" and act is not None:
                importance = _compute_wanda_importance(module.weight.data, act)
                act_mag = act.abs().mean().item()
            elif config.method == "random":
                import random
                importance = random.random()
                act_mag = 0.0
            else:  # magnitude
                importance = weight_mag
                act_mag = 0.0
            
            layers.append(LayerImportance(
                name=name,
                module_type=module_type,
                num_params=n_params,
                importance_score=importance,
                weight_magnitude=weight_mag,
                activation_magnitude=act_mag,
            ))
        
        _progress("Computing pruning plan...", 0.7)
        
        # Sort by importance (lowest first = prune first)
        layers.sort(key=lambda l: l.importance_score)
        
        # Determine which layers to prune to reach target ratio
        target_pruned = int(total_params * config.ratio)
        cumulative = 0
        layers_to_prune = []
        
        for layer in layers:
            layer.cumulative_ratio = (cumulative + layer.num_params) / max(total_params, 1)
            if cumulative < target_pruned:
                layer.prune_candidate = True
                layers_to_prune.append(layer.name)
                cumulative += layer.num_params
        
        actual_ratio = cumulative / max(total_params, 1)
        
        # Estimate speedup (rough: proportional to params removed)
        estimated_speedup = 1.0 / max(1.0 - actual_ratio, 0.1)
        
        _progress("Analysis complete", 1.0)
        
        return PruningAnalysis(
            total_params=total_params,
            target_pruned_params=target_pruned,
            target_ratio=config.ratio,
            actual_pruned_params=cumulative,
            actual_ratio=actual_ratio,
            layers=layers,
            layers_to_prune=layers_to_prune,
            estimated_speedup=estimated_speedup,
            elapsed_seconds=time.time() - start,
        )
    
    def prune(
        self,
        config: Optional[PruningConfig] = None,
        calibration_data: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[nn.Module, PruningAnalysis]:
        """Prune model by removing least important layers.
        
        Returns pruned model and analysis.
        Note: For production use, this creates a new model with pruned architecture.
        Current implementation zeros out pruned weights (soft pruning) for safety.
        """
        analysis = self.analyze(config, calibration_data, progress_callback)
        
        if progress_callback:
            progress_callback("Applying pruning...", 0.9)
        
        # Soft pruning: zero out weights of pruned layers
        # (Hard structural pruning requires architecture modification + re-export)
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in analysis.layers_to_prune and isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
        
        return self.model, analysis
