"""The autonomous runner (PRD 2.5).

The loop that runs the agent unattended for the trading window: on the broker's
calendar, at 10:00 and 14:00 ET, open a broker session, run one decision cycle,
close, and wait for the next slot. Runs on a small Linux VPS; a laptop sleep or
wifi loss costs at most one decision, so it is designed to survive and resume,
not to hold state in memory it cannot rebuild.

Robustness is the whole point (PRD 2.5):
- Scheduling reads Alpaca's calendar every loop; it never hardcodes dates.
- A broker session is opened fresh per cycle, so a dropped connection costs one
  cycle, not the run.
- A cycle that raises is logged and the loop continues -- one bad cycle must not
  end the run. (Within a cycle, hold-last-valid already governs; this is the
  outer guard.)
- SIGINT/SIGTERM request a graceful stop between cycles.

The loop takes an injectable clock and sleep so the scheduling can be tested
deterministically without real time. The per-cycle broker session comes from a
factory, so tests drive the whole loop with a fake gateway.
"""

from __future__ import annotations

import datetime as dt
import logging
import signal
import threading
import time
from typing import Callable, ContextManager

from . import config
from .gateway import BrokerGateway
from .orchestrator import CycleResult, DecisionCycle
from .scheduler import ET, next_decision_time, seconds_until

log = logging.getLogger("alloc_agent.runner")

GatewayFactory = Callable[[dt.date], ContextManager[BrokerGateway]]


class Runner:
    def __init__(
        self,
        cycle: DecisionCycle,
        gateway_factory: GatewayFactory,
        *,
        dry_run: bool = False,
        times: tuple[str, ...] = config.DECISION_TIMES_ET,
        poll_seconds: float = 30.0,
        calendar_lookahead_days: int = 14,
        now_fn: Callable[[], dt.datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        state_path: str | None = None,
    ):
        self.cycle = cycle
        self.gateway_factory = gateway_factory
        self.dry_run = dry_run
        self.times = times
        self.poll_seconds = poll_seconds
        self.calendar_lookahead_days = calendar_lookahead_days
        self._now = now_fn or (lambda: dt.datetime.now(ET))
        self._sleep = sleep_fn or time.sleep
        self._stop = threading.Event()
        # When set, cumulative state is loaded before and saved after each cycle,
        # so a stateless per-invocation scheduler keeps continuity across runs.
        self.state_path = state_path

    # -- control ------------------------------------------------------------

    def request_stop(self, *args) -> None:
        self._stop.set()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                # Not in the main thread (e.g. under test); skip.
                pass

    # -- one cycle ----------------------------------------------------------

    @staticmethod
    def cycle_id(target: dt.datetime) -> str:
        return target.strftime("%Y-%m-%dT%H:%M%z")

    def run_one(
        self,
        *,
        asof: dt.date | None = None,
        cycle_id: str | None = None,
        target: dt.datetime | None = None,
    ) -> CycleResult:
        """Open a fresh broker session, run one cycle, close it.

        If a state path is configured, the cumulative cross-cycle state is loaded
        before the cycle and saved after -- continuity for a stateless scheduler.
        """
        target = target or self._now()
        asof = asof or target.date()
        cycle_id = cycle_id or self.cycle_id(target)
        self._load_state()
        try:
            with self.gateway_factory(asof) as gateway:
                return self.cycle.run_cycle(
                    gateway, cycle_id=cycle_id, asof=asof, dry_run=self.dry_run
                )
        finally:
            self._save_state()

    def _load_state(self) -> None:
        import json
        import os

        if not self.state_path or not hasattr(self.cycle, "apply_state"):
            return
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                self.cycle.apply_state(json.load(fh))
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("could not load state from %s (%s); starting fresh",
                        self.state_path, exc)

    def _save_state(self) -> None:
        import json
        import os

        if not self.state_path or not hasattr(self.cycle, "state_dict"):
            return
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.cycle.state_dict(), fh, indent=2, default=str)
        os.replace(tmp, self.state_path)

    def _trading_dates(self, now: dt.datetime) -> list[str]:
        start = now.date()
        end = start + dt.timedelta(days=self.calendar_lookahead_days)
        with self.gateway_factory(start) as gateway:
            return gateway.trading_dates(start, end)

    # -- sleeping -----------------------------------------------------------

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in poll-sized chunks so a stop signal is honoured promptly."""
        remaining = seconds
        while remaining > 0 and not self._stop.is_set():
            chunk = min(self.poll_seconds, remaining)
            self._sleep(chunk)
            remaining -= chunk

    # -- the loop -----------------------------------------------------------

    def run(
        self,
        *,
        stop_at: dt.datetime | None = None,
        max_cycles: int | None = None,
        handle_signals: bool = True,
    ) -> int:
        """Run until stopped. Returns the number of cycles executed.

        stop_at ends the run at a wall-clock time; max_cycles bounds it by count
        (mostly for tests). Neither is required -- the default is to run until a
        signal requests a stop.
        """
        if handle_signals:
            self.install_signal_handlers()
        log.info("runner starting (dry_run=%s, times=%s)", self.dry_run, self.times)

        count = 0
        while not self._stop.is_set():
            now = self._now()
            if stop_at and now >= stop_at:
                break

            try:
                dates = self._trading_dates(now)
            except Exception as exc:
                log.warning("calendar fetch failed (%s); retrying after poll", exc)
                self._interruptible_sleep(self.poll_seconds)
                continue

            target = next_decision_time(now, dates, times=self.times)
            if target is None:
                log.info("no decision slot in the next %d days; waiting",
                         self.calendar_lookahead_days)
                self._interruptible_sleep(min(3600.0, self.poll_seconds * 20))
                continue
            if stop_at and target >= stop_at:
                break

            wait = seconds_until(target, now)
            log.info("next cycle %s in %.0fs", self.cycle_id(target), wait)
            self._interruptible_sleep(wait)
            if self._stop.is_set():
                break
            if self._now() < target:
                # Woke early (clock skew or a spurious wake); recompute.
                continue

            try:
                result = self.run_one(
                    asof=target.date(), cycle_id=self.cycle_id(target), target=target
                )
                log.info("cycle %s -> %s%s", result.cycle_id, result.status,
                         f" ({result.reason})" if result.reason else "")
            except Exception as exc:
                # Outer guard: one cycle raising must not end the run.
                log.exception("cycle at %s crashed, continuing: %s",
                              self.cycle_id(target), exc)

            count += 1
            if max_cycles is not None and count >= max_cycles:
                break
            # Advance past the slot just handled so it is not picked again.
            self._interruptible_sleep(self.poll_seconds)

        log.info("runner stopped after %d cycle(s)", count)
        return count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def load_correlation(path: str | None = None) -> dict:
    import json
    import os

    path = path or os.path.join(config.ARTIFACT_DIR, "correlation.json")
    with open(path, encoding="utf-8") as fh:
        art = json.load(fh)
    return {"keys": art["keys"], "matrix": art["matrix"]}


def build_runner(*, dry_run: bool = False, state_path: str | None = None) -> Runner:
    """Wire the production runner: real models, real MCP gateway."""
    from .allocator import Allocator
    from .broker_mcp import McpBrokerGateway
    from .challenger import Challenger
    from .llm import make_client

    cycle = DecisionCycle(
        Allocator(make_client()),
        Challenger(make_client()),
        correlation=load_correlation(),
        log_dir=config.LOG_DIR,
    )

    def factory(asof: dt.date):
        return McpBrokerGateway(asof=asof)

    return Runner(cycle, factory, dry_run=dry_run, state_path=state_path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-strategy allocation agent runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the full pipeline but place no orders")
    parser.add_argument("--once", action="store_true",
                        help="run a single cycle immediately and exit")
    parser.add_argument("--stop-at", type=str, default=None,
                        help="ISO datetime (ET) to end the run, e.g. 2026-09-04T16:00")
    parser.add_argument("--state", type=str, default=None,
                        help="JSON file to persist cumulative state across runs "
                             "(for --once under a scheduler like GitHub Actions)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runner = build_runner(dry_run=args.dry_run, state_path=args.state)

    if args.once:
        result = runner.run_one()
        log.info("single cycle -> %s (%s)", result.status, result.reason)
        return 0

    stop_at = None
    if args.stop_at:
        stop_at = dt.datetime.fromisoformat(args.stop_at)
        if stop_at.tzinfo is None:
            stop_at = stop_at.replace(tzinfo=ET)

    runner.run(stop_at=stop_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
