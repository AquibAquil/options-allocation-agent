"""Correlation precompute tests.

These run offline against a synthetic price series, so the construction can be
checked before any credential exists. The synthetic returns are fat-tailed on
purpose: a Gaussian sample produces too few 25-delta breaches for the piecewise
construction to calibrate against.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloc_agent.evidence import bs
from alloc_agent.evidence import correlation as corr
from alloc_agent.strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, LONG_STRANGLE
from conftest import synthetic_closes, synthetic_returns


# --- Black-Scholes ---------------------------------------------------------


def test_put_call_parity():
    """C - P = S - K at zero rates. Catches sign errors in the pricer."""
    s, k, sigma, t = 580.0, 570.0, 0.18, 30 / 252
    call = bs.price(s, k, sigma, t, "call")
    put = bs.price(s, k, sigma, t, "put")
    assert call - put == pytest.approx(s - k, abs=1e-9)


def test_strike_for_delta_round_trips():
    s, sigma, t = 580.0, 0.16, 10 / 252
    for right, target in (("put", 0.25), ("call", 0.25), ("call", 0.175)):
        k = bs.strike_for_delta(s, sigma, t, right, target)
        assert abs(bs.delta(s, k, sigma, t, right)) == pytest.approx(target, abs=1e-9)


def test_25_delta_put_strike_sits_below_spot():
    k = bs.strike_for_delta(580.0, 0.16, 10 / 252, "put", 0.25)
    assert 560.0 < k < 580.0


def test_ewma_volatility_is_a_forecast():
    """sigma[i] must not use return i, or the proxy prices with hindsight."""
    r = synthetic_returns()
    base = bs.ewma_volatility(r)
    bumped = r.copy()
    bumped[100] *= 10.0
    after = bs.ewma_volatility(bumped)
    assert after[100] == pytest.approx(base[100], abs=1e-15)
    assert after[101] > base[101]


def test_ewma_burn_in_is_masked():
    sigma = bs.ewma_volatility(synthetic_returns(), burn_in=20)
    assert np.all(np.isnan(sigma[:20]))
    assert np.all(np.isfinite(sigma[20:]))


# --- Revaluation construction (the default input) --------------------------


def test_revaluation_shape_check_passes(closes):
    """PRD 2.4: spreads strongly negative, strangle low or negative on both."""
    series, _ = corr.revaluation_series(closes)
    keys, m = corr.correlation_matrix(series)
    check = corr.check_expected_shape(keys, m)
    assert check["passed"], check["failures"]


def test_spreads_disagree_on_direction(closes):
    """The two spreads must respond oppositely to the same day."""
    series, _ = corr.revaluation_series(closes)
    bull = series[BULL_PUT_SPREAD.key]
    bear = series[BEAR_CALL_SPREAD.key]
    assert np.corrcoef(bull, bear)[0, 1] < -0.3


def test_strangle_gains_on_the_biggest_move(closes):
    """A long strangle must pay on the day the underlying moves most."""
    series, meta = corr.revaluation_series(closes)
    returns = np.diff(np.log(closes))
    aligned = returns[meta["burn_in"] : meta["burn_in"] + meta["n_days"]]
    biggest = int(np.argmax(np.abs(aligned)))
    assert series[LONG_STRANGLE.key][biggest] > 0


def test_strangle_bleeds_on_quiet_days(closes):
    """Its ordinary state is a small loss. This is thesis, not malfunction."""
    series, _ = corr.revaluation_series(closes)
    strangle = series[LONG_STRANGLE.key]
    assert np.median(strangle) < 0
    assert strangle.max() > 0


def test_spread_loss_is_bounded_by_defined_risk(closes):
    """Normalised by max loss, a spread day can never be worse than -1."""
    series, _ = corr.revaluation_series(closes)
    for key in (BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key):
        assert series[key].min() >= -1.0 - 1e-12


def test_strangle_loss_is_bounded_by_the_debit(closes):
    series, _ = corr.revaluation_series(closes)
    assert series[LONG_STRANGLE.key].min() >= -1.0 - 1e-12


def test_revaluation_rejects_short_series():
    with pytest.raises(ValueError, match="at least"):
        corr.revaluation_series(synthetic_closes(n=30))


# --- Piecewise construction (PRD 2.4 as written) ---------------------------


def test_piecewise_credits_are_fair_value_calibrated(returns):
    """Each spread proxy has mean zero: no assumed edge in the input."""
    inputs = corr.build_proxy_inputs(returns)
    series = corr.piecewise_series(returns, inputs)
    assert series[BULL_PUT_SPREAD.key].mean() == pytest.approx(0.0, abs=1e-12)
    assert series[BEAR_CALL_SPREAD.key].mean() == pytest.approx(0.0, abs=1e-12)


def test_piecewise_thresholds_scale_with_volatility():
    calm = corr.build_proxy_inputs(synthetic_returns(annual_vol=0.10))
    wild = corr.build_proxy_inputs(synthetic_returns(annual_vol=0.35))
    assert wild.breach_threshold_up > calm.breach_threshold_up
    assert wild.breach_threshold_down < calm.breach_threshold_down


def test_piecewise_fails_its_own_shape_check(returns):
    """Documents why the revaluation construction replaced it (see module docstring).

    If this ever starts passing, the finding that motivated the change no
    longer holds and the choice of construction should be revisited.
    """
    inputs = corr.build_proxy_inputs(returns)
    keys, m = corr.correlation_matrix(corr.piecewise_series(returns, inputs))
    check = corr.check_expected_shape(keys, m)
    assert not check["passed"]
    assert abs(check["bull_put_vs_bear_call"]) < 0.3


def test_piecewise_rejects_sample_without_breaches():
    tiny = np.full(252, 1e-6)
    tiny[0] = 2e-6  # non-zero variance, still no breaches
    with pytest.raises(ValueError, match="no breaches"):
        corr.build_proxy_inputs(tiny)


def test_piecewise_rejects_non_finite_input():
    bad = synthetic_returns()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        corr.build_proxy_inputs(bad)


# --- Artifact --------------------------------------------------------------


def test_matrix_is_valid_correlation(closes):
    artifact = corr.precompute(closes, sample_start="s", sample_end="e")
    m = np.array(artifact.matrix)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m, m.T)
    assert np.all(np.abs(m) <= 1.0 + 1e-12)


def test_artifact_reports_both_constructions(closes):
    artifact = corr.precompute(closes, sample_start="s", sample_end="e")
    assert artifact.construction == "revaluation"
    assert artifact.shape_check["passed"]
    assert not artifact.shape_check_piecewise["passed"]


def test_artifact_round_trips(tmp_path, closes):
    artifact = corr.precompute(closes, sample_start="s", sample_end="e")
    path = corr.save(artifact, str(tmp_path / "correlation.json"))
    reloaded = corr.load(path)
    assert reloaded.matrix == artifact.matrix
    assert reloaded.keys == artifact.keys
