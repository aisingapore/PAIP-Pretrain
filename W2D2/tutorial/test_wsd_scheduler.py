"""Smoke tests and plotting utility for the WSD scheduler exercise.

Usage:
    # Run correctness tests
    pytest test_wsd_scheduler.py -v

    # Generate LR schedule plot (saves wsd_schedule.png)
    python test_wsd_scheduler.py
"""
import math
import os
import sys
from unittest.mock import MagicMock

import pytest

# Ensure the exercise directory is on the path so we can import locally.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimizer_param_scheduler import OptimizerParamScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Realistic CPT parameters (mirrors MegatronBridge config.yaml defaults):
#   max_duration_in_token = 5e9, GBS = 1024, seq_length = 4096
#   train_iters = 5e9 / (1024 * 4096) = 1220
#   warmup_ratio = 0.1  -> warmup_iters = 122
#   decay_ratio  = 0.1  -> decay_iters  = 122
#
# For simplicity the scheduler works in "steps" (= iters in our exercise).
TOTAL_STEPS = 1220
WARMUP_STEPS = 122
DECAY_STEPS = 122  # WSD decay phase length
MAX_LR = 1e-4
MIN_LR = 1e-5


def _make_scheduler(decay_style="minus_sqrt"):
    """Create a WSD scheduler with realistic CPT parameters."""
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.0, "weight_decay": 0.0}]
    return OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=0.0,
        max_lr=MAX_LR,
        min_lr=MIN_LR,
        lr_warmup_steps=WARMUP_STEPS,
        lr_decay_steps=TOTAL_STEPS,
        lr_decay_style="WSD",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=TOTAL_STEPS,
        wd_incr_style="constant",
        wsd_decay_steps=DECAY_STEPS,
        lr_wsd_decay_style=decay_style,
    )


def _collect_lr_curve(scheduler, total_steps):
    """Step through the scheduler and record LR at each step."""
    param_group = {"max_lr": MAX_LR, "min_lr": MIN_LR}
    lrs = []
    for step in range(total_steps + 1):
        scheduler.num_steps = step
        lrs.append(scheduler.get_lr(param_group))
    return lrs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWSDMinusSqrt:
    """Verify the WSD schedule at key phase boundaries."""

    def test_warmup_start(self):
        """At step 0, LR should be init_lr (0.0)."""
        s = _make_scheduler()
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}
        s.num_steps = 0
        assert s.get_lr(pg) == pytest.approx(0.0)

    def test_warmup_end(self):
        """At the end of warmup, LR should reach max_lr."""
        s = _make_scheduler()
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}
        s.num_steps = WARMUP_STEPS
        assert s.get_lr(pg) == pytest.approx(MAX_LR)

    def test_stable_phase(self):
        """During the stable phase, LR should stay at max_lr."""
        s = _make_scheduler()
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}
        stable_mid = WARMUP_STEPS + (TOTAL_STEPS - WARMUP_STEPS - DECAY_STEPS) // 2
        s.num_steps = stable_mid
        assert s.get_lr(pg) == pytest.approx(MAX_LR)

    def test_decay_start(self):
        """At the start of decay, LR should still be at max_lr (coeff=1)."""
        s = _make_scheduler()
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}
        decay_start = TOTAL_STEPS - DECAY_STEPS
        s.num_steps = decay_start
        assert s.get_lr(pg) == pytest.approx(MAX_LR)

    def test_decay_end(self):
        """At the end of decay, LR should be at min_lr (coeff=0)."""
        s = _make_scheduler()
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}
        s.num_steps = TOTAL_STEPS
        assert s.get_lr(pg) == pytest.approx(MIN_LR)

    def test_decay_midpoint(self):
        """At the midpoint of decay, minus_sqrt gives coeff = 1 - sqrt(0.5)."""
        s = _make_scheduler()
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}
        s.num_steps = TOTAL_STEPS - DECAY_STEPS // 2
        expected_coeff = 1.0 - math.sqrt(0.5)
        expected_lr = MIN_LR + expected_coeff * (MAX_LR - MIN_LR)
        assert s.get_lr(pg) == pytest.approx(expected_lr, rel=1e-4)

    def test_beyond_total_steps(self):
        """After total steps, LR should be min_lr."""
        s = _make_scheduler()
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}
        s.num_steps = TOTAL_STEPS + 100
        assert s.get_lr(pg) == pytest.approx(MIN_LR)


class TestWSDMonotonic:
    """The decay phase LR must be monotonically decreasing."""

    def test_minus_sqrt_monotonic(self):
        s = _make_scheduler("minus_sqrt")
        lrs = _collect_lr_curve(s, TOTAL_STEPS)
        decay_start = TOTAL_STEPS - DECAY_STEPS
        decay_lrs = lrs[decay_start:]
        for i in range(1, len(decay_lrs)):
            assert decay_lrs[i] <= decay_lrs[i - 1] + 1e-12, (
                f"LR increased at decay step {i}: {decay_lrs[i-1]} -> {decay_lrs[i]}"
            )


class TestWSDAllStyles:
    """All four WSD decay styles should start at max_lr and end at min_lr."""

    @pytest.mark.parametrize("style", ["minus_sqrt", "linear", "cosine", "exponential"])
    def test_start_and_end(self, style):
        s = _make_scheduler(style)
        pg = {"max_lr": MAX_LR, "min_lr": MIN_LR}

        # At start of decay phase → max_lr
        s.num_steps = TOTAL_STEPS - DECAY_STEPS
        assert s.get_lr(pg) == pytest.approx(MAX_LR)

        # At end of decay phase → min_lr
        s.num_steps = TOTAL_STEPS
        assert s.get_lr(pg) == pytest.approx(MIN_LR)

    @pytest.mark.parametrize("style", ["minus_sqrt", "linear", "cosine", "exponential"])
    def test_monotonic_decay(self, style):
        s = _make_scheduler(style)
        lrs = _collect_lr_curve(s, TOTAL_STEPS)
        decay_start = TOTAL_STEPS - DECAY_STEPS
        decay_lrs = lrs[decay_start:]
        for i in range(1, len(decay_lrs)):
            assert decay_lrs[i] <= decay_lrs[i - 1] + 1e-12, (
                f"[{style}] LR increased at decay step {i}: "
                f"{decay_lrs[i-1]} -> {decay_lrs[i]}"
            )


# ---------------------------------------------------------------------------
# Plotting (run as: python test_wsd_scheduler.py)
# ---------------------------------------------------------------------------

def plot_lr_schedule():
    """Generate and save a plot of the WSD LR schedule."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left panel: minus_sqrt schedule with phase annotations ---
    ax = axes[0]
    s = _make_scheduler("minus_sqrt")
    lrs = _collect_lr_curve(s, TOTAL_STEPS)
    steps = list(range(TOTAL_STEPS + 1))

    ax.plot(steps, lrs, linewidth=2, color="tab:blue")
    ax.axvline(x=WARMUP_STEPS, color="gray", linestyle="--", alpha=0.7, label="End warmup")
    ax.axvline(
        x=TOTAL_STEPS - DECAY_STEPS,
        color="gray", linestyle=":", alpha=0.7, label="Start decay",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("WSD Schedule (minus_sqrt decay)")
    ax.legend()

    # Phase labels
    warmup_mid = WARMUP_STEPS // 2
    stable_mid = WARMUP_STEPS + (TOTAL_STEPS - WARMUP_STEPS - DECAY_STEPS) // 2
    decay_mid = TOTAL_STEPS - DECAY_STEPS // 2
    label_y = MAX_LR * 0.5
    for x, label in [(warmup_mid, "Warmup"), (stable_mid, "Stable"), (decay_mid, "Decay")]:
        ax.annotate(
            label, xy=(x, label_y), fontsize=10, ha="center", fontstyle="italic",
            color="gray",
        )

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # --- Right panel: all four decay styles overlaid ---
    ax2 = axes[1]
    styles = ["minus_sqrt", "linear", "cosine", "exponential"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for style, color in zip(styles, colors):
        s = _make_scheduler(style)
        lrs = _collect_lr_curve(s, TOTAL_STEPS)
        ax2.plot(steps, lrs, linewidth=2, color=color, label=style)

    ax2.axvline(x=WARMUP_STEPS, color="gray", linestyle="--", alpha=0.7)
    ax2.axvline(x=TOTAL_STEPS - DECAY_STEPS, color="gray", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("WSD Decay Style Comparison")
    ax2.legend()
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wsd_schedule.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved LR schedule plot to: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_lr_schedule()
