"""Black-Scholes, zero rates and dividends.

Used only by the correlation precompute, to value the three structures on
historical closes so their daily P&L series can be correlated. Nothing in the
live path prices anything: live Greeks, quotes and premiums come from the chain
through MCP (PRD 2.6). This exists so the correlation input has no hand-set
constants in it.

Zero rates is a real simplification. Over a 7-to-35 day horizon on a
proxy series whose purpose is to compare the SHAPE of three payoffs, carry is
immaterial. It would matter for pricing a position, and this never prices one.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


def _d1_d2(s: float, k: float, sigma: float, t: float):
    if t <= 0 or sigma <= 0:
        raise ValueError(f"need positive t and sigma, got t={t}, sigma={sigma}")
    vol_t = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + 0.5 * sigma**2 * t) / vol_t
    return d1, d1 - vol_t


def price(s: float, k: float, sigma: float, t: float, right: str) -> float:
    """Option price per share. Intrinsic value at or past expiry."""
    if t <= 0:
        return max(s - k, 0.0) if right == "call" else max(k - s, 0.0)
    d1, d2 = _d1_d2(s, k, sigma, t)
    if right == "call":
        return float(s * norm.cdf(d1) - k * norm.cdf(d2))
    if right == "put":
        return float(k * norm.cdf(-d2) - s * norm.cdf(-d1))
    raise ValueError(f"right must be call or put, got {right!r}")


def delta(s: float, k: float, sigma: float, t: float, right: str) -> float:
    if t <= 0:
        if right == "call":
            return 1.0 if s > k else 0.0
        return -1.0 if s < k else 0.0
    d1, _ = _d1_d2(s, k, sigma, t)
    return float(norm.cdf(d1)) if right == "call" else float(norm.cdf(d1) - 1.0)


def strike_for_delta(s: float, sigma: float, t: float, right: str, target: float) -> float:
    """Strike whose absolute delta equals `target`. Closed form, no solver.

    call: N(d1) = target        -> d1 = z(target)
    put:  N(d1) - 1 = -target   -> d1 = z(1 - target)
    K = S * exp(0.5*sigma^2*t - d1*sigma*sqrt(t))
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target delta must be in (0,1), got {target}")
    d1 = float(norm.ppf(target if right == "call" else 1.0 - target))
    return float(s * np.exp(0.5 * sigma**2 * t - d1 * sigma * np.sqrt(t)))


def ewma_volatility(
    log_returns: np.ndarray, *, lam: float = 0.94, burn_in: int = 20
) -> np.ndarray:
    """Annualised EWMA volatility, one value per return.

    lam = 0.94 is the RiskMetrics daily convention, chosen because it is a
    named standard rather than something fitted to this sample.

    Returns an array aligned with `log_returns`; the first `burn_in` entries are
    NaN because they are seeded rather than estimated.
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size <= burn_in:
        raise ValueError(f"need more than {burn_in} returns, got {r.size}")

    var = np.empty(r.size, dtype=float)
    seed = float(np.var(r[:burn_in], ddof=1))
    var[0] = seed
    for i in range(1, r.size):
        var[i] = lam * var[i - 1] + (1.0 - lam) * r[i - 1] ** 2

    sigma = np.sqrt(var * TRADING_DAYS)
    sigma[:burn_in] = np.nan
    return sigma
