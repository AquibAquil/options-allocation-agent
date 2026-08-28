"""Deterministic metrics for the evidence packet (PRD 2.1).

Every function here returns a NUMBER. None of them return a judgement.

That constraint is the whole point of the layer. If this code computed
"thesis conditions hold: true", the allocator would be writing prose around a
boolean and the model would be decorative. So there is no function here called
`is_cheap`, `trend_intact`, or `should_cut`. The allocator is handed distances,
ratios, ranks and percentages, and has to do the reconciling itself.

The one thing that looks like a verdict and is not: `sigma_distance`. How far
spot sits from a strike, measured in standard deviations of the move expected
over the remaining life, is an arithmetic fact. Whether that distance is
comfortable is the allocator's call.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252

# Horizons the allocator sees. 1 and 5 days speak to the current move, 21 and 63
# to whether it is a break from context or continuation of it.
RETURN_HORIZONS = (1, 5, 10, 21, 63)
VOL_WINDOWS = (5, 10, 21, 63)


def multi_horizon_returns(
    closes: np.ndarray, horizons: tuple[int, ...] = RETURN_HORIZONS
) -> dict[str, float | None]:
    """Simple returns over each horizon, ending on the last close.

    None where the series is too short, rather than a silently truncated
    window that would misreport the horizon it claims to cover.
    """
    closes = np.asarray(closes, dtype=float)
    if closes.size < 2:
        raise ValueError("need at least two closes")

    out: dict[str, float | None] = {}
    for h in horizons:
        if closes.size <= h:
            out[f"{h}d"] = None
        else:
            out[f"{h}d"] = float(closes[-1] / closes[-1 - h] - 1.0)
    return out


def realized_volatility(
    log_returns: np.ndarray, windows: tuple[int, ...] = VOL_WINDOWS
) -> dict[str, float | None]:
    """Annualised close-to-close realised volatility over each trailing window."""
    r = np.asarray(log_returns, dtype=float)
    out: dict[str, float | None] = {}
    for w in windows:
        if r.size < w or w < 2:
            out[f"{w}d"] = None
        else:
            out[f"{w}d"] = float(r[-w:].std(ddof=1) * np.sqrt(TRADING_DAYS))
    return out


def parkinson_volatility(
    highs: np.ndarray, lows: np.ndarray, window: int = 21
) -> float | None:
    """Annualised Parkinson range volatility.

    Reported alongside close-to-close because it uses the intraday range rather
    than discarding it, and is the more efficient estimator for the same
    sample. Where the two disagree, the gap is itself evidence: a market
    travelling a wide daily range but closing flat is not the same market as one
    grinding steadily, and close-to-close cannot tell them apart.
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    if highs.size != lows.size:
        raise ValueError("highs and lows must be the same length")
    if highs.size < window or window < 1:
        return None

    h, l = highs[-window:], lows[-window:]
    if np.any(h <= 0) or np.any(l <= 0) or np.any(h < l):
        raise ValueError("invalid high/low bars")

    log_range_sq = np.log(h / l) ** 2
    daily_var = log_range_sq.mean() / (4.0 * np.log(2.0))
    return float(np.sqrt(daily_var * TRADING_DAYS))


def percentile_rank(history: np.ndarray, value: float) -> float:
    """Where `value` sits within `history`, 0 to 100.

    Uses the midrank convention so that ties do not bias the rank up or down.
    This is the standard IV-percentile definition: the share of the lookback
    spent below the current level.
    """
    hist = np.asarray(history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        raise ValueError("empty history")
    below = float(np.sum(hist < value))
    equal = float(np.sum(hist == value))
    return float(100.0 * (below + 0.5 * equal) / hist.size)


def trailing_window(series: np.ndarray, lookback: int) -> np.ndarray:
    """Last `lookback` finite observations, or everything available."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    return s[-lookback:] if s.size > lookback else s


def sigma_distance(
    spot: float, strike: float, annual_vol: float, days_remaining: float
) -> float:
    """Distance from spot to strike in standard deviations of the remaining move.

    Signed: positive when the strike is above spot, negative when below. The
    magnitude answers "how many typical moves away is this strike", which is
    what makes a 2% buffer at 8% volatility comparable to a 2% buffer at 30%.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if annual_vol <= 0:
        raise ValueError("annual_vol must be positive")
    if days_remaining <= 0:
        return float("inf") if strike > spot else float("-inf")
    sigma_move = annual_vol * np.sqrt(days_remaining / TRADING_DAYS)
    return float(np.log(strike / spot) / sigma_move)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Largest peak-to-trough decline, as a positive fraction.

    Takes an equity curve, not a return series. Returns 0.0 for a curve that
    only ever rises.
    """
    curve = np.asarray(equity_curve, dtype=float)
    if curve.size == 0:
        raise ValueError("empty equity curve")
    if np.any(curve <= 0):
        raise ValueError("equity curve must be positive to express drawdown as a fraction")
    peaks = np.maximum.accumulate(curve)
    return float(np.max((peaks - curve) / peaks))


def vol_risk_premium(implied: float, realized: float) -> dict[str, float]:
    """The IV/RV relationship, reported three ways and judged none of them.

    Implied above realised is the volatility risk premium, not a mispricing.
    Selling it is a risk-bearing business, not arbitrage. Both the spread and
    the ratio are given because they answer different questions: five points of
    premium means something different at 10 vol than at 40.
    """
    if realized <= 0:
        raise ValueError("realized must be positive")
    return {
        "implied": float(implied),
        "realized": float(realized),
        "spread": float(implied - realized),
        "ratio": float(implied / realized),
    }
