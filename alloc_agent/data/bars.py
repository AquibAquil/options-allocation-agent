"""QQQ daily bars from Alpaca, with an on-disk cache.

Only the historical equity bars endpoint is used here. Chains, quotes, Greeks
and execution all go through MCP (PRD 2.6); this module exists because the
correlation precompute needs 252 days of history before any of that is wired,
and because realised volatility must be anchored to COMPLETED bars.

PRD 2.7: on the Basic plan the most recent 15 minutes of historical bars and
trades cannot be pulled. Daily bars are therefore requested through yesterday's
session, never through today's, so a partial bar never enters a volatility
calculation.
"""

from __future__ import annotations

import csv
import datetime as dt
import os

import httpx
import numpy as np

from ..config import ALPACA_DATA_BASE, ALPACA_KEY_ID, ALPACA_SECRET_KEY, CACHE_DIR

_FEEDS = ("sip", "iex")


class BarsUnavailable(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    if not (ALPACA_KEY_ID and ALPACA_SECRET_KEY):
        raise BarsUnavailable(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set in the environment"
        )
    return {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }


def cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol.lower()}_daily.csv")


def fetch_daily_bars(
    symbol: str,
    *,
    days: int,
    end: dt.date | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Fetch daily bars ending on the last COMPLETED session.

    Returns oldest-first. Overshoots the calendar window so `days` trading
    sessions survive weekends and holidays.
    """
    end = end or (dt.date.today() - dt.timedelta(days=1))
    start = end - dt.timedelta(days=int(days * 1.6) + 20)

    last_error: Exception | None = None
    for feed in _FEEDS:
        params = {
            "symbols": symbol,
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "all",
            "feed": feed,
            "limit": 10000,
        }
        try:
            with httpx.Client(timeout=timeout) as client:
                bars: list[dict] = []
                page_token = None
                while True:
                    if page_token:
                        params["page_token"] = page_token
                    resp = client.get(
                        f"{ALPACA_DATA_BASE}/v2/stocks/bars",
                        params=params,
                        headers=_headers(),
                    )
                    if resp.status_code in (401, 403):
                        raise BarsUnavailable(
                            f"feed={feed} rejected ({resp.status_code}): {resp.text[:200]}"
                        )
                    resp.raise_for_status()
                    payload = resp.json()
                    bars.extend(payload.get("bars", {}).get(symbol, []))
                    page_token = payload.get("next_page_token")
                    if not page_token:
                        break
            if not bars:
                raise BarsUnavailable(f"feed={feed} returned no bars for {symbol}")
            return bars[-days:]
        except BarsUnavailable as exc:
            last_error = exc
            continue

    raise BarsUnavailable(f"no usable feed for {symbol}: {last_error}")


def write_cache(symbol: str, bars: list[dict]) -> str:
    path = cache_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for bar in bars:
            writer.writerow(
                [bar["t"][:10], bar["o"], bar["h"], bar["l"], bar["c"], bar["v"]]
            )
    return path


def read_cache(symbol: str) -> list[dict]:
    path = cache_path(symbol)
    if not os.path.exists(path):
        raise BarsUnavailable(f"no cached bars at {path}")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise BarsUnavailable(f"cache at {path} is empty")
    return [
        {
            "t": row["date"],
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row["volume"]),
        }
        for row in rows
    ]


def log_returns(bars: list[dict]) -> np.ndarray:
    closes = np.array([float(b["c"]) for b in bars], dtype=float)
    if closes.size < 2:
        raise BarsUnavailable("need at least two closes for a return series")
    if np.any(closes <= 0):
        raise BarsUnavailable("non-positive close in bar series")
    return np.diff(np.log(closes))


def bar_date(bar: dict) -> str:
    return str(bar["t"])[:10]
