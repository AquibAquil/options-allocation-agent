"""Metrics tests. Closed-form cases wherever one exists."""

from __future__ import annotations

import numpy as np
import pytest

from alloc_agent.evidence import metrics
from conftest import synthetic_returns


# --- returns ---------------------------------------------------------------


def test_multi_horizon_returns_are_exact():
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 110.0])
    out = metrics.multi_horizon_returns(closes, horizons=(1, 5))
    assert out["1d"] == pytest.approx(110.0 / 104.0 - 1.0)
    assert out["5d"] == pytest.approx(110.0 / 100.0 - 1.0)


def test_horizon_longer_than_series_is_none_not_truncated():
    """A truncated window would misreport the horizon it claims to cover."""
    out = metrics.multi_horizon_returns(np.array([100.0, 101.0]), horizons=(1, 63))
    assert out["1d"] is not None
    assert out["63d"] is None


# --- volatility ------------------------------------------------------------


def test_realized_volatility_annualises():
    daily = 0.01
    r = np.array([daily, -daily] * 40)
    out = metrics.realized_volatility(r, windows=(20,))
    expected = np.std(r[-20:], ddof=1) * np.sqrt(252)
    assert out["20d"] == pytest.approx(expected)


def test_realized_volatility_window_too_short_is_none():
    out = metrics.realized_volatility(np.array([0.01, -0.01]), windows=(5,))
    assert out["5d"] is None


def test_parkinson_matches_closed_form():
    """Constant log range r gives sigma = sqrt(r^2 / (4 ln2) * 252)."""
    ratio = 1.02
    highs = np.full(30, 100.0 * ratio)
    lows = np.full(30, 100.0)
    got = metrics.parkinson_volatility(highs, lows, window=21)
    expected = np.sqrt(np.log(ratio) ** 2 / (4 * np.log(2)) * 252)
    assert got == pytest.approx(expected)


def test_parkinson_needs_a_full_window():
    assert metrics.parkinson_volatility(np.full(5, 101.0), np.full(5, 100.0), 21) is None


def test_parkinson_rejects_inverted_bars():
    with pytest.raises(ValueError, match="invalid high/low"):
        metrics.parkinson_volatility(np.full(30, 99.0), np.full(30, 100.0), 21)


def test_parkinson_sees_range_that_close_to_close_misses():
    """A wide-range market closing flat reads as zero vol close-to-close."""
    closes_flat = np.zeros(30)
    assert metrics.realized_volatility(closes_flat, windows=(21,))["21d"] == 0.0
    parkinson = metrics.parkinson_volatility(np.full(30, 103.0), np.full(30, 97.0), 21)
    assert parkinson > 0.2


# --- percentile ------------------------------------------------------------


def test_percentile_rank_endpoints():
    hist = np.arange(100, dtype=float)
    assert metrics.percentile_rank(hist, -1.0) == 0.0
    assert metrics.percentile_rank(hist, 1000.0) == 100.0


def test_percentile_rank_uses_midrank_for_ties():
    hist = np.array([5.0, 5.0, 5.0, 5.0])
    assert metrics.percentile_rank(hist, 5.0) == 50.0


def test_percentile_rank_is_share_below():
    hist = np.arange(1, 101, dtype=float)
    assert metrics.percentile_rank(hist, 25.5) == pytest.approx(25.0)


def test_percentile_rank_ignores_nan():
    hist = np.array([1.0, 2.0, np.nan, 3.0])
    assert metrics.percentile_rank(hist, 2.5) == pytest.approx(200.0 / 3.0)


def test_percentile_rank_rejects_empty():
    with pytest.raises(ValueError, match="empty history"):
        metrics.percentile_rank(np.array([np.nan]), 1.0)


def test_trailing_window_takes_the_tail():
    assert list(metrics.trailing_window(np.arange(10.0), 3)) == [7.0, 8.0, 9.0]
    assert metrics.trailing_window(np.arange(2.0), 10).size == 2


# --- strike distance -------------------------------------------------------


def test_sigma_distance_signs():
    assert metrics.sigma_distance(100.0, 110.0, 0.2, 30) > 0
    assert metrics.sigma_distance(100.0, 90.0, 0.2, 30) < 0
    assert metrics.sigma_distance(100.0, 100.0, 0.2, 30) == pytest.approx(0.0)


def test_sigma_distance_normalises_across_regimes():
    """The same percentage buffer is nearer in sigma when volatility is higher."""
    calm = abs(metrics.sigma_distance(100.0, 98.0, 0.08, 10))
    wild = abs(metrics.sigma_distance(100.0, 98.0, 0.30, 10))
    assert calm > wild


def test_sigma_distance_shrinks_as_expiry_approaches():
    far = abs(metrics.sigma_distance(100.0, 98.0, 0.2, 30))
    near = abs(metrics.sigma_distance(100.0, 98.0, 0.2, 3))
    assert near > far


def test_sigma_distance_at_expiry_is_infinite():
    assert metrics.sigma_distance(100.0, 110.0, 0.2, 0) == float("inf")
    assert metrics.sigma_distance(100.0, 90.0, 0.2, 0) == float("-inf")


def test_sigma_distance_rejects_bad_inputs():
    with pytest.raises(ValueError):
        metrics.sigma_distance(0.0, 100.0, 0.2, 10)
    with pytest.raises(ValueError):
        metrics.sigma_distance(100.0, 100.0, 0.0, 10)


# --- drawdown --------------------------------------------------------------


def test_max_drawdown_known_curve():
    assert metrics.max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(0.25)


def test_max_drawdown_of_a_rising_curve_is_zero():
    assert metrics.max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0


def test_max_drawdown_rejects_non_positive_curve():
    with pytest.raises(ValueError, match="must be positive"):
        metrics.max_drawdown(np.array([100.0, -1.0]))


# --- IV/RV -----------------------------------------------------------------


def test_vol_risk_premium_reports_spread_and_ratio():
    out = metrics.vol_risk_premium(0.20, 0.16)
    assert out["spread"] == pytest.approx(0.04)
    assert out["ratio"] == pytest.approx(1.25)


def test_vol_risk_premium_can_be_negative():
    """Realised above implied is a real state, not an error."""
    assert metrics.vol_risk_premium(0.14, 0.20)["spread"] < 0


def test_vol_risk_premium_rejects_zero_realized():
    with pytest.raises(ValueError, match="realized must be positive"):
        metrics.vol_risk_premium(0.2, 0.0)


# --- the layer emits evidence, not verdicts (PRD 2.1) ----------------------


def test_no_metric_returns_a_boolean():
    """A boolean here would make the allocator decorative."""
    r = synthetic_returns()
    closes = 100 * np.exp(np.cumsum(r))
    results = [
        metrics.multi_horizon_returns(closes),
        metrics.realized_volatility(r),
        metrics.percentile_rank(r, float(r[-1])),
        metrics.sigma_distance(100.0, 98.0, 0.2, 10),
        metrics.vol_risk_premium(0.2, 0.15),
    ]
    for result in results:
        values = result.values() if isinstance(result, dict) else [result]
        assert not any(isinstance(v, bool) for v in values)
