"""VXN history from CBOE, with an on-disk cache.

IV percentile is load-bearing evidence. The long strangle's invalidation is
anchored to whether protection is still cheap relative to its own history, and
the spreads' theses turn on implied sitting above realised by enough to pay for
the risk taken. None of that can be evaluated from a live chain snapshot, and
four trading days of accumulated IV is not a distribution.

Alpaca does not serve index data and historical options data is out of scope
(PRD appendix), so the reference series is CBOE's published VXN history. VXN is
the Nasdaq-100 volatility index, which is the correct reference for QQQ; VIX
would be the S&P and would understate it, currently by around five vol points.

CBOE is an external data provider and is named in the README, as the rules
require.

What this is: a market-wide reference for where implied volatility sits within
its own range. What it is not: the implied volatility of the specific contracts
held. Those come from the chain through MCP and are reported alongside, never
replaced by this.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os

import httpx
import numpy as np

from ..config import CACHE_DIR

VXN_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

INDEX_URLS = {"VXN": VXN_URL, "VIX": VIX_URL}


class VolIndexUnavailable(RuntimeError):
    pass


def cache_path(index: str) -> str:
    return os.path.join(CACHE_DIR, f"{index.lower()}_history.csv")


def _parse(text: str, index: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise VolIndexUnavailable(f"{index}: empty CSV")

    out: list[dict] = []
    for row in rows:
        raw_date = (row.get("DATE") or "").strip()
        raw_close = (row.get("CLOSE") or "").strip()
        if not raw_date or not raw_close:
            continue
        try:
            date = dt.datetime.strptime(raw_date, "%m/%d/%Y").date()
        except ValueError:
            try:
                date = dt.date.fromisoformat(raw_date)
            except ValueError:
                continue
        try:
            close = float(raw_close)
        except ValueError:
            continue
        if close <= 0:
            continue
        out.append(
            {
                "date": date.isoformat(),
                "high": float(row.get("HIGH") or close),
                "low": float(row.get("LOW") or close),
                "close": close,
            }
        )

    if not out:
        raise VolIndexUnavailable(f"{index}: no usable rows")
    out.sort(key=lambda r: r["date"])
    return out


def fetch(index: str = "VXN", *, timeout: float = 30.0) -> list[dict]:
    """Fetch the full published history, oldest first. No credentials needed."""
    url = INDEX_URLS.get(index.upper())
    if url is None:
        raise VolIndexUnavailable(f"unknown index {index!r}")
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise VolIndexUnavailable(f"{index}: {exc}") from exc
    return _parse(resp.text, index)


def write_cache(index: str, rows: list[dict]) -> str:
    path = cache_path(index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "high", "low", "close"])
        for row in rows:
            writer.writerow([row["date"], row["high"], row["low"], row["close"]])
    return path


def read_cache(index: str = "VXN") -> list[dict]:
    path = cache_path(index)
    if not os.path.exists(path):
        raise VolIndexUnavailable(f"no cached {index} at {path}")
    with open(path, encoding="utf-8") as fh:
        rows = [
            {
                "date": r["date"],
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
            for r in csv.DictReader(fh)
        ]
    if not rows:
        raise VolIndexUnavailable(f"cached {index} at {path} is empty")
    return rows


def load(index: str = "VXN", *, refresh: bool = True) -> list[dict]:
    """Cache-backed load. Falls back to cache when the fetch fails.

    A stale cache is better than a missing input mid-window, but the caller is
    told how stale it is via `as_of` so the packet can carry that fact rather
    than hide it.
    """
    if refresh:
        try:
            rows = fetch(index)
            write_cache(index, rows)
            return rows
        except VolIndexUnavailable:
            pass
    return read_cache(index)


def closes(rows: list[dict]) -> np.ndarray:
    return np.array([r["close"] for r in rows], dtype=float)


def as_of(rows: list[dict]) -> str:
    return rows[-1]["date"]
