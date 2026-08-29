"""Scheduler tests (PRD 2.5).

The real trading dates from the hackathon window are used (Aug 31 - Sep 4, with
Sep 7 Labor Day absent), so the cadence is checked against the actual calendar
the broker returned, not an assumed one.
"""

from __future__ import annotations

import datetime as dt

import pytest

from alloc_agent import scheduler as sch
from alloc_agent.scheduler import ET

# The real window from Alpaca's calendar: Sep 1-4 all trade, Sep 7 is Labor Day.
WINDOW = ["2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-08"]


def et(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=ET)


def test_two_slots_per_day_at_10_and_14():
    slots = sch.decision_slots(dt.date(2026, 8, 31))
    assert [s.strftime("%H:%M") for s in slots] == ["10:00", "14:00"]
    assert all(s.tzinfo is ET for s in slots)


def test_before_open_next_is_10am_same_day():
    now = et(2026, 8, 31, 8, 0)
    assert sch.next_decision_time(now, WINDOW) == et(2026, 8, 31, 10, 0)


def test_between_slots_next_is_2pm():
    now = et(2026, 8, 31, 11, 0)
    assert sch.next_decision_time(now, WINDOW) == et(2026, 8, 31, 14, 0)


def test_after_last_slot_rolls_to_next_trading_day():
    now = et(2026, 8, 31, 15, 0)
    assert sch.next_decision_time(now, WINDOW) == et(2026, 9, 1, 10, 0)


def test_friday_afternoon_skips_the_weekend():
    """After Fri Sep 4's last slot, the next is Tue Sep 8 -- the weekend and
    Labor Day (Sep 7) are simply absent from the calendar."""
    now = et(2026, 9, 4, 15, 0)
    assert sch.next_decision_time(now, WINDOW) == et(2026, 9, 8, 10, 0)


def test_non_trading_day_is_never_scheduled():
    """Sep 7 (Labor Day) is not in the calendar, so nothing schedules on it."""
    scheduled_dates = {
        sch.next_decision_time(et(2026, 9, d, 8, 0), WINDOW).date().isoformat()
        for d in range(1, 5)
    }
    assert "2026-09-07" not in scheduled_dates


def test_exactly_on_a_slot_returns_the_next_one():
    """Strictly after: standing exactly on 10:00 points to 14:00, not itself."""
    assert sch.next_decision_time(et(2026, 8, 31, 10, 0), WINDOW) == et(2026, 8, 31, 14, 0)


def test_returns_none_when_calendar_is_exhausted():
    now = et(2026, 9, 8, 15, 0)  # after the last slot of the last known date
    assert sch.next_decision_time(now, WINDOW) is None


def test_naive_datetime_is_treated_as_et():
    naive = dt.datetime(2026, 8, 31, 8, 0)
    assert sch.next_decision_time(naive, WINDOW) == et(2026, 8, 31, 10, 0)


# --- is_due (cron-style trigger) ------------------------------------------


def test_is_due_within_window_after_slot():
    assert sch.is_due(et(2026, 8, 31, 10, 5), WINDOW) == et(2026, 8, 31, 10, 0)


def test_is_due_none_before_slot():
    """Never fire before the slot -- avoids running on stale pre-open data."""
    assert sch.is_due(et(2026, 8, 31, 9, 55), WINDOW) is None


def test_is_due_none_outside_window():
    assert sch.is_due(et(2026, 8, 31, 10, 30), WINDOW) is None


def test_is_due_none_on_non_trading_day():
    assert sch.is_due(et(2026, 9, 7, 10, 5), WINDOW) is None


# --- seconds_until ---------------------------------------------------------


def test_seconds_until_is_positive_and_correct():
    now = et(2026, 8, 31, 9, 30)
    target = et(2026, 8, 31, 10, 0)
    assert sch.seconds_until(target, now) == pytest.approx(1800.0)


def test_seconds_until_never_negative():
    now = et(2026, 8, 31, 11, 0)
    target = et(2026, 8, 31, 10, 0)
    assert sch.seconds_until(target, now) == 0.0
