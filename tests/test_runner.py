"""Autonomous runner tests (PRD 2.5).

The loop is driven by an injected clock and sleep, so scheduling is tested
without real time: the fake sleep advances the fake clock, so the loop marches
through slots deterministically. A fake gateway factory records how many cycles
actually ran and on which dates.

The cases that matter: cycles fire at the scheduled slots, only on trading days;
a broker/calendar failure does not end the run; a cycle that raises does not end
the run; stop conditions are honoured.
"""

from __future__ import annotations

import datetime as dt

import pytest

from alloc_agent import runner as rn
from alloc_agent.scheduler import ET
from alloc_agent.orchestrator import CycleResult, DRY_RUN

WINDOW = ["2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]


def et(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=ET)


class FakeClock:
    def __init__(self, start):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t = self.t + dt.timedelta(seconds=seconds)


class FakeGateway:
    def __init__(self, dates, *, fail_calendar=False):
        self._dates = dates
        self._fail_calendar = fail_calendar

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def trading_dates(self, start, end):
        if self._fail_calendar:
            raise RuntimeError("calendar down")
        return [d for d in self._dates if start.isoformat() <= d <= end.isoformat()]


class RecordingCycle:
    """Stands in for DecisionCycle; records each run_cycle call."""

    def __init__(self, *, raise_on=None):
        self.calls = []
        self._raise_on = raise_on or set()

    def run_cycle(self, gateway, *, cycle_id, asof, dry_run=False):
        self.calls.append({"cycle_id": cycle_id, "asof": asof.isoformat(), "dry_run": dry_run})
        if asof.isoformat() in self._raise_on:
            raise RuntimeError("boom")
        return CycleResult(cycle_id=cycle_id, asof=asof.isoformat(), status=DRY_RUN)

    def state_dict(self):
        return {"calls": len(self.calls)}

    def apply_state(self, state):
        pass


def make_runner(clock, cycle, *, dates=WINDOW, fail_calendar=False, dry_run=True, poll=600.0):
    factory = lambda asof: FakeGateway(dates, fail_calendar=fail_calendar)
    return rn.Runner(
        cycle, factory, dry_run=dry_run, poll_seconds=poll,
        now_fn=clock.now, sleep_fn=clock.sleep,
    )


# --- run_one ---------------------------------------------------------------


def test_run_one_runs_a_single_cycle():
    clock = FakeClock(et(2026, 8, 31, 10, 0))
    cycle = RecordingCycle()
    runner = make_runner(clock, cycle)
    result = runner.run_one(target=et(2026, 8, 31, 10, 0))
    assert len(cycle.calls) == 1
    assert cycle.calls[0]["dry_run"] is True
    assert result.status == DRY_RUN


# --- the loop --------------------------------------------------------------


def test_loop_fires_at_the_two_daily_slots():
    clock = FakeClock(et(2026, 8, 31, 8, 0))   # before the first slot
    cycle = RecordingCycle()
    runner = make_runner(clock, cycle)
    runner.run(max_cycles=2, handle_signals=False)
    ran = [c["cycle_id"][:16] for c in cycle.calls]
    assert ran == ["2026-08-31T10:00", "2026-08-31T14:00"]


def test_loop_rolls_across_trading_days():
    clock = FakeClock(et(2026, 8, 31, 15, 0))  # after Monday's last slot
    cycle = RecordingCycle()
    runner = make_runner(clock, cycle)
    runner.run(max_cycles=2, handle_signals=False)
    dates = [c["asof"] for c in cycle.calls]
    # Next two slots are Tuesday's 10:00 and 14:00.
    assert dates == ["2026-09-01", "2026-09-01"]


def test_loop_skips_non_trading_days():
    """From Friday afternoon, the next cycle is the following trading day, never
    the weekend or Labor Day."""
    clock = FakeClock(et(2026, 9, 4, 15, 0))
    cycle = RecordingCycle()
    # Calendar includes the following week's Tuesday (Sep 8; Sep 7 is Labor Day).
    runner = make_runner(clock, cycle, dates=WINDOW + ["2026-09-08"])
    runner.run(max_cycles=1, handle_signals=False)
    assert cycle.calls[0]["asof"] == "2026-09-08"


def test_stop_at_ends_the_run():
    clock = FakeClock(et(2026, 8, 31, 8, 0))
    cycle = RecordingCycle()
    runner = make_runner(clock, cycle)
    # Stop after the first slot but before the second.
    runner.run(stop_at=et(2026, 8, 31, 12, 0), handle_signals=False)
    assert [c["cycle_id"][:16] for c in cycle.calls] == ["2026-08-31T10:00"]


def test_a_crashing_cycle_does_not_end_the_run():
    """PRD 2.5 outer guard: one cycle raising must not stop the loop."""
    clock = FakeClock(et(2026, 8, 31, 8, 0))
    cycle = RecordingCycle(raise_on={"2026-08-31"})  # both Monday cycles raise
    runner = make_runner(clock, cycle)
    runner.run(max_cycles=3, handle_signals=False)
    # It kept going past the crashing Monday cycles into Tuesday.
    assert len(cycle.calls) == 3
    assert cycle.calls[-1]["asof"] == "2026-09-01"


def test_calendar_failure_does_not_end_the_run():
    """A broker/calendar error retries rather than crashing the loop."""
    clock = FakeClock(et(2026, 8, 31, 8, 0))
    cycle = RecordingCycle()

    class FlakyFactory:
        def __init__(self):
            self.attempts = 0

        def __call__(self, asof):
            self.attempts += 1
            # Fail the first two calendar fetches, then recover.
            return FakeGateway(WINDOW, fail_calendar=self.attempts <= 2)

    runner = rn.Runner(
        cycle, FlakyFactory(), dry_run=True, poll_seconds=600.0,
        now_fn=clock.now, sleep_fn=clock.sleep,
    )
    runner.run(max_cycles=1, handle_signals=False)
    assert len(cycle.calls) == 1   # eventually ran despite early calendar failures


def test_stop_request_breaks_the_loop():
    clock = FakeClock(et(2026, 8, 31, 8, 0))
    cycle = RecordingCycle()
    runner = make_runner(clock, cycle)
    runner.request_stop()
    ran = runner.run(handle_signals=False)
    assert ran == 0
    assert cycle.calls == []


# --- CLI wiring ------------------------------------------------------------


def test_load_correlation_reads_the_artifact(tmp_path):
    import json
    art = {"keys": ["a", "b", "c"], "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "extra": "ignored"}
    p = tmp_path / "correlation.json"
    p.write_text(json.dumps(art))
    loaded = rn.load_correlation(str(p))
    assert loaded == {"keys": ["a", "b", "c"], "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


# --- cross-run state persistence (GitHub Actions) --------------------------


def test_state_round_trips_through_a_file(tmp_path):
    """run_one with a state_path saves cumulative state and reloads it next run."""
    import json
    from alloc_agent.orchestrator import DecisionCycle
    from alloc_agent.benchmark import CycleReturn
    from alloc_agent.strategies import KEYS

    corr = {"keys": list(KEYS), "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    # First "process": accumulate some state and save it.
    cyc1 = DecisionCycle(None, None, correlation=corr, log_dir=str(tmp_path))
    cyc1.rejections.record("MODIFY")
    cyc1.delta.record(CycleReturn("c1", {k: 0.01 for k in KEYS}, {k: 0.3 for k in KEYS}))
    cyc1.shadow.mark(spot=580.0, annual_vol=0.18, asof=dt.date(2026, 8, 31))
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(cyc1.state_dict(), default=str))

    # Second "process": a fresh cycle loads the saved state.
    cyc2 = DecisionCycle(None, None, correlation=corr, log_dir=str(tmp_path))
    cyc2.apply_state(json.loads(state_file.read_text()))

    assert cyc2.rejections.modify == 1
    assert len(cyc2.delta.history) == 1
    assert set(cyc2.shadow.positions) == set(KEYS)   # shadow positions restored


def test_runner_persists_state_across_run_one_calls(tmp_path):
    clock = FakeClock(et(2026, 8, 31, 10, 0))
    cycle = RecordingCycle()
    state = tmp_path / "state.json"
    runner = rn.Runner(
        cycle, lambda asof: FakeGateway(WINDOW), dry_run=True,
        now_fn=clock.now, sleep_fn=clock.sleep, state_path=str(state),
    )
    runner.run_one(target=et(2026, 8, 31, 10, 0))
    assert state.exists()          # state written after the cycle
