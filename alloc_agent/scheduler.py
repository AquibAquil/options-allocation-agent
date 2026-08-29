"""Decision cadence (PRD 2.5).

Two cycles per trading day, at 10:00 and 14:00 ET -- 30 minutes after the open
and two hours before the close, avoiding the widest-spread windows. Roughly
eight to ten decisions across the hackathon window.

Scheduling uses Alpaca's market calendar, never a hardcoded set of dates:
holidays and early closes are handled by asking (the caller passes the trading
dates it got from the broker) rather than assuming. This module is pure -- given
the trading dates and the current time, it computes the next decision moment --
so it is fully testable, and the DST-correct wall-clock conversion is the only
thing it needs zoneinfo for.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from . import config

ET = ZoneInfo(config.MARKET_TZ)


def _parse_hhmm(value: str) -> tuple[int, int]:
    hh, mm = value.split(":")
    return int(hh), int(mm)


def decision_slots(date: dt.date, times: tuple[str, ...] = config.DECISION_TIMES_ET) -> list[dt.datetime]:
    """The decision datetimes (ET, tz-aware) for one date, in order."""
    slots = []
    for t in times:
        hh, mm = _parse_hhmm(t)
        slots.append(dt.datetime(date.year, date.month, date.day, hh, mm, tzinfo=ET))
    return sorted(slots)


def _as_et(now: dt.datetime) -> dt.datetime:
    """Interpret a naive datetime as ET; convert an aware one into ET."""
    if now.tzinfo is None:
        return now.replace(tzinfo=ET)
    return now.astimezone(ET)


def next_decision_time(
    now: dt.datetime,
    trading_dates: set[str] | list[str],
    *,
    times: tuple[str, ...] = config.DECISION_TIMES_ET,
) -> dt.datetime | None:
    """The next decision moment strictly after `now`, on a trading day.

    `trading_dates` is the set of ISO dates the broker's calendar reports as
    trading days. Returns None if none of the supplied dates has a remaining
    slot after `now` -- the caller then refreshes the calendar further out.
    """
    now = _as_et(now)
    dates = sorted({d if isinstance(d, str) else d.isoformat() for d in trading_dates})
    for iso in dates:
        date = dt.date.fromisoformat(iso)
        for slot in decision_slots(date, times):
            if slot > now:
                return slot
    return None


def is_due(
    now: dt.datetime,
    trading_dates: set[str] | list[str],
    *,
    times: tuple[str, ...] = config.DECISION_TIMES_ET,
    window_minutes: float = 15.0,
) -> dt.datetime | None:
    """The slot `now` falls within, if a decision is due (for cron triggers).

    A cron or timer that fires near a slot needs to know "is this a decision
    moment?" without hitting exactly on the second. Returns the slot datetime if
    `now` is within `window_minutes` AFTER a slot on a trading day, else None.
    Firing only after the slot (never before) keeps a decision from running on
    stale pre-open data.
    """
    now = _as_et(now)
    today = now.date().isoformat()
    dates = {d if isinstance(d, str) else d.isoformat() for d in trading_dates}
    if today not in dates:
        return None
    for slot in decision_slots(now.date(), times):
        delta = (now - slot).total_seconds()
        if 0 <= delta <= window_minutes * 60.0:
            return slot
    return None


def seconds_until(target: dt.datetime, now: dt.datetime) -> float:
    """Non-negative seconds from `now` until `target` (both coerced to ET)."""
    return max(0.0, (_as_et(target) - _as_et(now)).total_seconds())
