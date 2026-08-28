"""Shared synthetic fixtures.

The whole suite runs offline. Nothing here touches Alpaca or CBOE, so the
maths can be checked before any credential exists and the tests stay
deterministic once one does.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPOT = 580.0
ASOF = dt.date(2026, 8, 28)


def synthetic_returns(n: int = 252, annual_vol: float = 0.17, seed: int = 7):
    """Fat-tailed daily log returns, scaled to a target annualised vol."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df=4, size=n)
    raw = raw / raw.std(ddof=1)
    return raw * (annual_vol / np.sqrt(252))


def synthetic_closes(n: int = 252, annual_vol: float = 0.17, seed: int = 7):
    r = synthetic_returns(n, annual_vol, seed)
    return SPOT * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def synthetic_bars(n: int = 252, annual_vol: float = 0.17, seed: int = 7, end=ASOF):
    """Alpaca-shaped daily bars with a plausible intraday range.

    Dates are consecutive weekdays counted back from `end`, which is enough for
    code that only ever reads the last date.
    """
    closes = synthetic_closes(n, annual_vol, seed)
    rng = np.random.default_rng(seed + 1)
    dates, day = [], end
    while len(dates) < closes.size:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day -= dt.timedelta(days=1)
    dates.reverse()

    bars = []
    prev = closes[0]
    for date, close in zip(dates, closes):
        rng_frac = abs(rng.normal(0, 0.004)) + abs(close / prev - 1.0)
        high = max(close, prev) * (1.0 + rng_frac)
        low = min(close, prev) * (1.0 - rng_frac)
        bars.append(
            {
                "t": date,
                "o": float(prev),
                "h": float(high),
                "l": float(low),
                "c": float(close),
                "v": 5.0e7,
            }
        )
        prev = close
    return bars


def synthetic_vxn(n: int = 800, level: float = 20.0, seed: int = 3, end=ASOF):
    """VXN-shaped history: mean-reverting, positive, occasional spikes."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = np.log(level)
    for i in range(1, n):
        shock = rng.normal(0, 0.06) + (0.5 if rng.random() < 0.01 else 0.0)
        x[i] = x[i - 1] + 0.05 * (np.log(level) - x[i - 1]) + shock
    values = np.exp(x)

    dates, day = [], end
    while len(dates) < n:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day -= dt.timedelta(days=1)
    dates.reverse()

    return [
        {"date": d, "high": float(v * 1.03), "low": float(v * 0.97), "close": float(v)}
        for d, v in zip(dates, values)
    ]


@pytest.fixture
def bars():
    return synthetic_bars()


@pytest.fixture
def vxn():
    return synthetic_vxn()


@pytest.fixture
def closes():
    return synthetic_closes()


@pytest.fixture
def returns():
    return synthetic_returns()
