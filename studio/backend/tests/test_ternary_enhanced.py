"""Unit tests for enhanced ternary training core."""
import pytest
import torch
import torch.nn as nn

from core.training.ternary_enhanced import (
    CalibrationStats,
    EnhancedTernarySTE,
    AdaptiveGradientClipper,
    SpectralAnchorRegularizer,
    TernaryTrainingManager,
    TernaryTrainerCallback,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_linear():
    return nn.Linear(64, 32)


@pytest.fixture
def stats():
    return CalibrationStats(group_size=16)


# ── CalibrationStats ────────────────────────────────────────────────────────

class TestCalibrationStats:
    def test_weight_group_scales_shape(self, stats):
        w = torch.randn(32, 64)
        scales = stats.collect_weight_stats("test", w)
        expected_groups = (32 * 64) // 16
        assert scales.shape == (expected_groups,)
        assert scales.dtype == torch.float16

    def test_weight_group_scales_positive(self, stats):
        w = torch.randn(32, 64)
        scales = stats.collect_weight_stats("test", w)
        assert (scales > 0).all()

    def test_activation_channel_scales(self, stats):
        act = torch.randn(4, 32)
        scales = stats.collect_activation_stats("test", act)
        assert scales.shape == (32,)
        assert (scales > 0).all()

    def test_adaptive_clip_threshold_default(self, stats):
        # Not enough history → returns default 1.0
        assert stats.get_adaptive_clip_threshold("missing") == 1.0

    def test_adaptive_clip_threshold_with_history(self, stats):
        for i in range(50):
            stats.record_gradient_norm("layer", float(i))
        threshold = stats.get_adaptive_clip_threshold("layer", percentile=95.0)
        assert 40.0 <= threshold <= 50.0

    def test_mark_calibrated(self, stats):
        assert not stats.is_calibrated
        stats.mark_calibrated()
        assert stats.is_calibrated


# ── EnhancedTernarySTE ──────────────────────────────────────────────────────

class TestEnhancedTernarySTE:
    def test_output_values_are_scaled_ternary(self):
        w = torch.tensor([2.0, -1.5, 0.01, -0.02, 3.0, 0.0, -2.5, 1.0])
        scales = torch.tensor([1.5, 2.0], dtype=torch.float16)
        out = EnhancedTernarySTE.apply(w, scales, 0.1, 4)
        # Non-zero outputs should be ±scale values
        nonzero = out[out.abs() > 1e-6]
        for v in nonzero:
            assert abs(abs(v.item()) - 1.5) < 1e-4 or abs(abs(v.item()) - 2.0) < 1e-4

    def test_zero_threshold_keeps_all(self):
        w = torch.randn(16)
        scales = torch.ones(1, dtype=torch.float16)
        out = EnhancedTernarySTE.apply(w, scales, 0.0, 16)
        # All non-zero weights should survive with threshold=0
        assert (out != 0).sum() == (w != 0).sum()

    def test_high_threshold_zeros_most(self):
        w = torch.randn(64) * 0.1  # Small values
        scales = torch.ones(4, dtype=torch.float16)
        out = EnhancedTernarySTE.apply(w, scales, 1.0, 16)
        # Most should be zeroed
        assert (out == 0).sum() > 48

    def test_backward_passes_gradient(self):
        w = torch.randn(16, requires_grad=True)
        scales = torch.ones(1, dtype=torch.float16)
        out = EnhancedTernarySTE.apply(w, scales, 0.0, 16)
        loss = out.sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape

    def test_backward_masks_small_weights(self):
        w = torch.tensor([2.0, 0.01, -3.0, 0.02], requires_grad=True)
        scales = torch.ones(1, dtype=torch.float16)
        out = EnhancedTernarySTE.apply(w, scales, 0.5, 4)
        loss = out.sum()
        loss.backward()
        # Weights below threshold should have zero gradient
        assert w.grad[1].item() == 0.0
        assert w.grad[3].item() == 0.0
        # Weights above threshold should have non-zero gradient
        assert w.grad[0].item() != 0.0
        assert w.grad[2].item() != 0.0


# ── SpectralAnchorRegularizer ───────────────────────────────────────────────

class TestSpectralAnchorRegularizer:
    def test_zero_loss_at_init(self, small_linear):
        reg = SpectralAnchorRegularizer(lambda_anchor=0.01)
        reg.initialize_anchors(small_linear)
        loss = reg.compute_loss(small_linear)
        assert loss.item() < 1e-6

    def test_nonzero_loss_after_weight_change(self, small_linear):
        reg = SpectralAnchorRegularizer(lambda_anchor=0.01)
        reg.initialize_anchors(small_linear)
        with torch.no_grad():
            small_linear.weight.data *= 2.0
        loss = reg.compute_loss(small_linear)
        assert loss.item() > 0.0

    def test_zero_lambda_gives_zero_loss(self, small_linear):
        reg = SpectralAnchorRegularizer(lambda_anchor=0.0)
        reg.initialize_anchors(small_linear)
        with torch.no_grad():
            small_linear.weight.data *= 5.0
        loss = reg.compute_loss(small_linear)
        assert loss.item() == 0.0


# ── AdaptiveGradientClipper ─────────────────────────────────────────────────

class TestAdaptiveGradientClipper:
    def test_clips_large_gradients(self, small_linear):
        stats = CalibrationStats(group_size=16)
        clipper = AdaptiveGradientClipper(stats, warmup_steps=0)
        # Fill history with small norms
        for _ in range(20):
            stats.record_gradient_norm("weight", 0.5)
        # Set large gradient
        small_linear.weight.grad = torch.ones_like(small_linear.weight) * 100.0
        clipper.clip_gradients(small_linear)
        # Gradient should be clipped
        assert small_linear.weight.grad.norm().item() < 100.0


# ── TernaryTrainingManager ──────────────────────────────────────────────────

class TestTernaryTrainingManager:
    def test_manager_creation(self, small_linear):
        mgr = TernaryTrainingManager(small_linear, group_size=16)
        assert mgr.group_size == 16
        assert mgr.current_threshold == 0.0

    def test_state_dict_roundtrip(self, small_linear):
        mgr = TernaryTrainingManager(small_linear, group_size=16)
        state = mgr.state_dict()
        mgr2 = TernaryTrainingManager(small_linear, group_size=32)
        mgr2.load_state_dict(state)
        assert mgr2.group_size == 16
        assert mgr2.stats.is_calibrated

    def test_threshold_cosine_schedule(self, small_linear):
        mgr = TernaryTrainingManager(
            small_linear, initial_threshold=1.0, threshold_schedule="cosine"
        )
        mgr.step_scheduler(epoch=0, total_epochs=100)
        assert abs(mgr.current_threshold - 1.0) < 1e-6
        mgr.step_scheduler(epoch=50, total_epochs=100)
        assert abs(mgr.current_threshold - 0.5) < 0.05
        mgr.step_scheduler(epoch=100, total_epochs=100)
        assert mgr.current_threshold < 0.01

    def test_register_and_remove_hooks(self, small_linear):
        mgr = TernaryTrainingManager(small_linear, group_size=16)
        mgr.stats.collect_weight_stats("", small_linear.weight.data)
        mgr.stats.mark_calibrated()
        mgr.register_hooks(small_linear)
        assert len(mgr._hooks) > 0
        mgr.remove_hooks()
        assert len(mgr._hooks) == 0
