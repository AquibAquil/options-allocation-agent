"""Cross-strategy correlation precompute (PRD 2.4).

Live correlation is impossible with four trading days of data, so this is
precomputed from 252 days of QQQ history or the input is noise.

WHAT THIS IS NOT: a backtest. No chain data, no real strikes, no quotes, no
fills, no expiration handling, no transaction costs. Three synthetic daily P&L
series stand in for the three strategies. It answers exactly one question --
do these three respond to different conditions -- which is all the allocator
needs correlation for. Say so in the write-up.

Two constructions live here
---------------------------
`piecewise`   The construction described literally in PRD 2.4: a small constant
              on quiet days, the return itself when a threshold is breached.

`revaluation` The default. Each structure is set up on the day's close at its
              target deltas and DTE, then revalued one day later on the real
              next close with an updated volatility. Daily P&L is normalised by
              the structure's own max loss, which is the same denominator the
              allocator allocates in (PRD 2.2).

The piecewise construction FAILS the shape check that PRD 2.4 itself specifies,
and it fails for a structural reason rather than a sampling one: outside a
breach both spread proxies are constant, so roughly 240 of 252 days carry no
information and the two series almost never move on the same day. Their
covariance collapses to about -2 * credit_bull * credit_bear, which is
negligible against standard deviations set by breach-day magnitudes. Real
spread P&L varies continuously with the underlying through delta; the piecewise
version only registers a breach. Both matrices are reported so the difference
is visible rather than asserted.

Sign convention: every series is P&L. Positive is a gain for the strategy
holding it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from ..config import CORRELATION_LOOKBACK_DAYS, UNDERLYING
from ..strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, LONG_STRANGLE
from . import bs

TRADING_DAYS = bs.TRADING_DAYS

# Standard normal quantile for a 25-delta strike. norm.ppf(0.75).
_DELTA_25_Z = float(norm.ppf(0.75))

# QQQ lists $1 strikes around the money. Strikes are rounded to this grid so
# the proxy sits on structures that could actually be traded.
STRIKE_SPACING = 1.0

# "2-3 strikes below/above" the short leg (PRD 2.3).
SPREAD_WIDTH_STRIKES = 3


# ---------------------------------------------------------------------------
# Piecewise construction, exactly as PRD 2.4 describes it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxyInputs:
    """Everything the piecewise construction derives from the sample.

    The breach thresholds are also reported on their own, because "the move
    that would put spot at a 25-delta short strike" is useful context for the
    write-up regardless of which construction is used.
    """

    n_days: int
    sigma_annual: float
    sigma_daily: float
    breach_threshold_down: float   # log return, negative
    breach_threshold_up: float     # log return, positive
    spread_dte: int
    strangle_decay: float
    bull_put_credit: float
    bear_call_credit: float
    n_breach_down: int
    n_breach_up: int


def _strike_log_distance(sigma_annual: float, dte: int, right: str) -> float:
    """Log distance from spot to the 25-delta short strike, zero rates.

    A daily return beyond this distance puts spot at or through the short
    strike in a single session.
    """
    t = dte / TRADING_DAYS
    drift = 0.5 * sigma_annual**2 * t
    diffusion = _DELTA_25_Z * sigma_annual * np.sqrt(t)
    if right == "put":
        return float(-diffusion + drift)
    if right == "call":
        return float(diffusion + drift)
    raise ValueError(f"right must be put or call, got {right!r}")


def build_proxy_inputs(log_returns: np.ndarray) -> ProxyInputs:
    r = np.asarray(log_returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("log_returns must be one dimensional")
    if r.size < 60:
        raise ValueError(f"need a meaningful sample, got {r.size} days")
    if not np.all(np.isfinite(r)):
        raise ValueError("log_returns contains non-finite values")

    sigma_daily = float(r.std(ddof=1))
    sigma_annual = sigma_daily * np.sqrt(TRADING_DAYS)

    spread_dte = BULL_PUT_SPREAD.dte_target
    thr_down = _strike_log_distance(sigma_annual, spread_dte, "put")
    thr_up = _strike_log_distance(sigma_annual, spread_dte, "call")

    breach_down = r < thr_down
    breach_up = r > thr_up
    n_down = int(breach_down.sum())
    n_up = int(breach_up.sum())
    if n_down == 0 or n_up == 0:
        raise ValueError(
            f"no breaches in sample (down={n_down}, up={n_up}); thresholds cannot "
            "be calibrated against a sample that never tests them"
        )

    # Fair-value credit: credit collected on quiet days exactly offsets losses
    # on breach days, so the proxy carries no assumed edge.
    bull_put_credit = float(-r[breach_down].sum() / (r.size - n_down))
    bear_call_credit = float(r[breach_up].sum() / (r.size - n_up))

    # Expected absolute daily move. E|X| for X ~ N(0, s).
    strangle_decay = float(sigma_daily * np.sqrt(2.0 / np.pi))

    return ProxyInputs(
        n_days=int(r.size),
        sigma_annual=sigma_annual,
        sigma_daily=sigma_daily,
        breach_threshold_down=float(thr_down),
        breach_threshold_up=float(thr_up),
        spread_dte=spread_dte,
        strangle_decay=strangle_decay,
        bull_put_credit=bull_put_credit,
        bear_call_credit=bear_call_credit,
        n_breach_down=n_down,
        n_breach_up=n_up,
    )


def piecewise_series(
    log_returns: np.ndarray, inputs: ProxyInputs
) -> dict[str, np.ndarray]:
    """PRD 2.4 as written. Retained for comparison, not used as the input.

    PRD 2.4 phrases the bear call proxy as "the return when above"; taken
    literally that makes a rip higher a GAIN for a short call spread. The sign
    is corrected here, which is the more favourable reading for this
    construction, and it still fails the shape check.
    """
    r = np.asarray(log_returns, dtype=float)
    return {
        BULL_PUT_SPREAD.key: np.where(
            r < inputs.breach_threshold_down, r, inputs.bull_put_credit
        ),
        BEAR_CALL_SPREAD.key: np.where(
            r > inputs.breach_threshold_up, -r, inputs.bear_call_credit
        ),
        LONG_STRANGLE.key: np.abs(r) - inputs.strangle_decay,
    }


# ---------------------------------------------------------------------------
# Revaluation construction (default)
# ---------------------------------------------------------------------------


def _round_strike(k: float) -> float:
    return round(k / STRIKE_SPACING) * STRIKE_SPACING


def _spread_day(
    spot: float, spot_next: float, sigma: float, sigma_next: float, dte: int, right: str
) -> float:
    """One day of P&L for a short vertical spread, as a fraction of max loss.

    Sells the 25-delta strike and buys three strikes further out of the money,
    then revalues on the next close with one less day and an updated vol.
    """
    t, t_next = dte / TRADING_DAYS, (dte - 1) / TRADING_DAYS
    short_k = _round_strike(bs.strike_for_delta(spot, sigma, t, right, 0.25))
    offset = -SPREAD_WIDTH_STRIKES if right == "put" else SPREAD_WIDTH_STRIKES
    long_k = short_k + offset * STRIKE_SPACING
    width = SPREAD_WIDTH_STRIKES * STRIKE_SPACING

    credit = bs.price(spot, short_k, sigma, t, right) - bs.price(
        spot, long_k, sigma, t, right
    )
    max_loss = width - credit
    if max_loss <= 0:
        raise ValueError(f"credit {credit:.4f} exceeds width {width} for {right} spread")

    value_next = bs.price(spot_next, short_k, sigma_next, t_next, right) - bs.price(
        spot_next, long_k, sigma_next, t_next, right
    )
    # Short the spread: collected `credit`, now owe `value_next`. Defined risk
    # floors the loss at max_loss even if the model says otherwise.
    pnl = credit - value_next
    return float(max(pnl, -max_loss) / max_loss)


def _strangle_day(
    spot: float, spot_next: float, sigma: float, sigma_next: float, dte: int
) -> float:
    """One day of P&L for a long strangle, as a fraction of max loss (the debit)."""
    t, t_next = dte / TRADING_DAYS, (dte - 1) / TRADING_DAYS
    call_k = _round_strike(bs.strike_for_delta(spot, sigma, t, "call", 0.175))
    put_k = _round_strike(bs.strike_for_delta(spot, sigma, t, "put", 0.175))

    debit = bs.price(spot, call_k, sigma, t, "call") + bs.price(
        spot, put_k, sigma, t, "put"
    )
    if debit <= 0:
        raise ValueError("strangle debit is non-positive")

    value_next = bs.price(spot_next, call_k, sigma_next, t_next, "call") + bs.price(
        spot_next, put_k, sigma_next, t_next, "put"
    )
    return float((value_next - debit) / debit)


def revaluation_series(
    closes: np.ndarray, *, burn_in: int = 20
) -> tuple[dict[str, np.ndarray], dict]:
    """Daily P&L per unit of max loss, one series per strategy.

    Positions are re-struck every day rather than held. That is deliberate: the
    question is whether the three respond differently to the SAME day, and a
    rolling setup isolates that without holding-period path effects confounding
    it.
    """
    closes = np.asarray(closes, dtype=float)
    if closes.size < burn_in + 30:
        raise ValueError(f"need at least {burn_in + 30} closes, got {closes.size}")
    if np.any(closes <= 0):
        raise ValueError("non-positive close in series")

    returns = np.diff(np.log(closes))
    sigma = bs.ewma_volatility(returns, burn_in=burn_in)

    bull, bear, strangle = [], [], []
    # returns[i] carries closes[i] -> closes[i+1], and ewma_volatility builds
    # sigma[i] from returns strictly before i, so sigma[i] is the vol standing
    # at closes[i]: a forecast, never fitted to the move it is used to price.
    for i in range(burn_in, returns.size - 1):
        s, s_next = closes[i], closes[i + 1]
        v, v_next = sigma[i], sigma[i + 1]
        if not (np.isfinite(v) and np.isfinite(v_next)) or v <= 0 or v_next <= 0:
            continue
        bull.append(_spread_day(s, s_next, v, v_next, BULL_PUT_SPREAD.dte_target, "put"))
        bear.append(
            _spread_day(s, s_next, v, v_next, BEAR_CALL_SPREAD.dte_target, "call")
        )
        strangle.append(
            _strangle_day(s, s_next, v, v_next, LONG_STRANGLE.dte_target)
        )

    series = {
        BULL_PUT_SPREAD.key: np.array(bull),
        BEAR_CALL_SPREAD.key: np.array(bear),
        LONG_STRANGLE.key: np.array(strangle),
    }
    finite_sigma = sigma[np.isfinite(sigma)]
    meta = {
        "n_days": len(bull),
        "burn_in": burn_in,
        "strike_spacing": STRIKE_SPACING,
        "spread_width_strikes": SPREAD_WIDTH_STRIKES,
        "ewma_lambda": 0.94,
        "sigma_mean": float(finite_sigma.mean()),
        "sigma_min": float(finite_sigma.min()),
        "sigma_max": float(finite_sigma.max()),
        "mean_daily_pnl_frac": {k: float(v.mean()) for k, v in series.items()},
        "std_daily_pnl_frac": {k: float(v.std(ddof=1)) for k, v in series.items()},
    }
    return series, meta


# ---------------------------------------------------------------------------
# Matrix, validation, artifact
# ---------------------------------------------------------------------------


def correlation_matrix(series: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    keys = list(series.keys())
    stacked = np.vstack([series[k] for k in keys])
    if np.any(stacked.std(axis=1) == 0):
        raise ValueError("a proxy series is constant; correlation undefined")
    return keys, np.corrcoef(stacked)


def check_expected_shape(keys: list[str], m: np.ndarray) -> dict:
    """PRD 2.4 states the shape a valid construction must produce.

    Spreads strongly negatively correlated. Strangle low or negative against
    both. If the strangle correlates strongly with either spread, the
    construction is wrong -- flag it rather than feed the allocator garbage.
    """
    i = {k: n for n, k in enumerate(keys)}
    bull_bear = float(m[i[BULL_PUT_SPREAD.key], i[BEAR_CALL_SPREAD.key]])
    bull_strangle = float(m[i[BULL_PUT_SPREAD.key], i[LONG_STRANGLE.key]])
    bear_strangle = float(m[i[BEAR_CALL_SPREAD.key], i[LONG_STRANGLE.key]])

    failures = []
    if bull_bear >= -0.3:
        failures.append(
            f"spreads not strongly negatively correlated ({bull_bear:+.3f} >= -0.30)"
        )
    if bull_strangle > 0.5:
        failures.append(f"strangle tracks bull put ({bull_strangle:+.3f} > 0.50)")
    if bear_strangle > 0.5:
        failures.append(f"strangle tracks bear call ({bear_strangle:+.3f} > 0.50)")

    return {
        "bull_put_vs_bear_call": bull_bear,
        "bull_put_vs_strangle": bull_strangle,
        "bear_call_vs_strangle": bear_strangle,
        "passed": not failures,
        "failures": failures,
    }


@dataclass(frozen=True)
class CorrelationArtifact:
    underlying: str
    lookback_days: int
    sample_start: str
    sample_end: str
    construction: str
    keys: list[str]
    matrix: list[list[float]]
    matrix_piecewise: list[list[float]]
    revaluation_meta: dict
    piecewise_inputs: dict
    shape_check: dict
    shape_check_piecewise: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def precompute(
    closes: np.ndarray, *, sample_start: str, sample_end: str
) -> CorrelationArtifact:
    closes = np.asarray(closes, dtype=float)
    returns = np.diff(np.log(closes))

    series, meta = revaluation_series(closes)
    keys, m = correlation_matrix(series)

    piecewise_inputs = build_proxy_inputs(returns)
    keys_pw, m_pw = correlation_matrix(piecewise_series(returns, piecewise_inputs))
    if keys_pw != keys:
        raise AssertionError("construction key order diverged")

    return CorrelationArtifact(
        underlying=UNDERLYING,
        lookback_days=CORRELATION_LOOKBACK_DAYS,
        sample_start=sample_start,
        sample_end=sample_end,
        construction="revaluation",
        keys=keys,
        matrix=[[float(x) for x in row] for row in m],
        matrix_piecewise=[[float(x) for x in row] for row in m_pw],
        revaluation_meta=meta,
        piecewise_inputs=asdict(piecewise_inputs),
        shape_check=check_expected_shape(keys, m),
        shape_check_piecewise=check_expected_shape(keys, m_pw),
    )


def save(artifact: CorrelationArtifact, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(artifact.to_json())
    return path


def load(path: str) -> CorrelationArtifact:
    with open(path, encoding="utf-8") as fh:
        return CorrelationArtifact(**json.load(fh))
