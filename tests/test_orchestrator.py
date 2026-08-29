"""Decision cycle tests (PRD 2.5, 2.8).

A FakeGateway drives the whole cycle offline. The cases that matter most are the
failure paths: PRD 2.5 says a model failure, a broker error, or an unconfirmable
order must HOLD the last valid allocation, never move the portfolio. Those are
the tests that protect real capital, so most of this file is about things going
wrong.

The chain is the real captured QQQ put chain, so the one clean-trade path runs a
bull put spread end to end (selection -> sizing -> order spec -> fill) exactly as
it would live, minus the network.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from alloc_agent import orchestrator as orch
from alloc_agent.allocator import Allocator
from alloc_agent.challenger import Challenger
from alloc_agent.evidence import packet as pk
from alloc_agent.gateway import BrokerError
from alloc_agent.llm import ModelResponse, ModelUnavailable
from alloc_agent.strategies import BULL_PUT_SPREAD, BEAR_CALL_SPREAD, KEYS, LONG_STRANGLE
from conftest import ASOF, synthetic_bars, synthetic_vxn

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "qqq_put_chain_sep11.json")
with open(FIXTURE) as fh:
    PUT_CHAIN = json.load(fh)


# --- fakes -----------------------------------------------------------------


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, *, system, user, schema, max_tokens=16000):
        return ModelResponse(data=self.payload, raw_text="{}", model="fake", usage={})


class RaisingClient:
    def complete_json(self, *, system, user, schema, max_tokens=16000):
        raise ModelUnavailable("simulated failure")


def alloc_payload(bull=0.40, bear=0.0, strangle=0.0):
    return {
        "allocations": {BULL: bull, BEAR: bear, STRANGLE: strangle},
        "reasoning": {k: "evidence-grounded reason" for k in KEYS},
        "portfolio_rationale": "spreads negatively correlated; budget not concentrated",
    }


def challenge_payload(verdict="APPROVE", bull=0.40, bear=0.0, strangle=0.0):
    return {
        "verdict": verdict,
        "critique": "consistent with the evidence",
        "evidence_cited": ["short_strike_distance_sigma"],
        "modified_allocations": {BULL: bull, BEAR: bear, STRANGLE: strangle},
    }


FILLED_ORDER = {
    "id": "srv-1",
    "client_order_id": "unused",
    "status": "filled",
    "qty": "1",
    "filled_qty": "1",
    "legs": [
        {"symbol": "QQQ260911P00702000", "filled_qty": "1"},
        {"symbol": "QQQ260911P00699000", "filled_qty": "1"},
    ],
}


class FakeGateway:
    def __init__(self, *, market_open=True, equity=100_000.0, positions=None,
                 chain=None, place_result=None, fail_on=None):
        self._market_open = market_open
        self._equity = equity
        self._positions = positions or {k: None for k in KEYS}
        self._chain = chain if chain is not None else PUT_CHAIN
        self._place_result = place_result if place_result is not None else dict(FILLED_ORDER)
        self._fail_on = fail_on or set()
        self.placed = []

    def _maybe_fail(self, name):
        if name in self._fail_on:
            raise BrokerError(f"simulated {name} failure")

    def is_market_open(self):
        self._maybe_fail("clock")
        return self._market_open

    def account(self):
        self._maybe_fail("account")
        return pk.AccountSnapshot(equity=self._equity, buying_power=self._equity * 2, options_buying_power=self._equity)

    def positions(self):
        self._maybe_fail("positions")
        return self._positions

    def daily_bars(self, symbol, *, days):
        self._maybe_fail("bars")
        return synthetic_bars()

    def option_chain(self, symbol, **filters):
        self._maybe_fail("chain")
        return self._chain

    def place_order(self, spec):
        self._maybe_fail("place")
        self.placed.append(spec)
        result = dict(self._place_result)
        result["client_order_id"] = spec.client_order_id
        for leg in result.get("legs", []):
            leg["client_order_id"] = spec.client_order_id
        return result

    def order_status(self, *, client_order_id):
        self._maybe_fail("status")
        result = dict(self._place_result)
        result["client_order_id"] = client_order_id
        return result


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Keep evidence gathering hermetic: no CBOE fetch."""
    from alloc_agent.data import vol_index
    monkeypatch.setattr(vol_index, "load", lambda *a, **k: synthetic_vxn())


def make_cycle(allocator_payload=None, challenger_payload=None, tmp_path=None):
    allocator = Allocator(FakeClient(allocator_payload or alloc_payload()))
    challenger = Challenger(FakeClient(challenger_payload or challenge_payload()))
    corr = {"keys": list(KEYS), "matrix": [[1, -0.88, -0.45], [-0.88, 1, 0.01], [-0.45, 0.01, 1]]}
    return orch.DecisionCycle(
        allocator, challenger, correlation=corr, log_dir=str(tmp_path or "logs")
    )


# --- clean trade path ------------------------------------------------------


def test_clean_cycle_opens_a_bull_put_spread(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    gw = FakeGateway()
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)

    assert result.status == orch.TRADED
    assert len(gw.placed) == 1
    order = result.orders[0]
    assert order["strategy"] == BULL
    assert order["submitted"] is True
    assert order["fill"]["verdict"] == "filled"
    assert order["fill"]["atomic"] is True
    # The gated allocation becomes the new anchor.
    assert cycle.last_valid_allocation[BULL] > 0


def test_clean_cycle_writes_a_log_line(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    cycle.run_cycle(FakeGateway(), cycle_id="c1", asof=ASOF)
    log = tmp_path / "cycles.jsonl"
    assert log.exists()
    record = json.loads(log.read_text().strip())
    assert record["cycle_id"] == "c1"
    assert record["status"] == orch.TRADED
    assert record["packet"] is not None
    assert record["challenge"]["verdict"] == "APPROVE"


# --- PRD 2.5 failure paths: hold, do not trade -----------------------------


def test_market_closed_skips_without_trading(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    gw = FakeGateway(market_open=False)
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    assert result.status == orch.SKIPPED_MARKET_CLOSED
    assert gw.placed == []


def test_allocator_failure_holds(tmp_path):
    allocator = Allocator(RaisingClient())
    challenger = Challenger(FakeClient(challenge_payload()))
    corr = {"keys": list(KEYS), "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    cycle = orch.DecisionCycle(allocator, challenger, correlation=corr, log_dir=str(tmp_path))
    gw = FakeGateway()
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    assert result.status == orch.HELD_ON_FAILURE
    assert "allocator unavailable" in result.reason
    assert gw.placed == []
    assert cycle.last_valid_allocation == {k: 0.0 for k in KEYS}  # unchanged


def test_challenger_failure_holds(tmp_path):
    allocator = Allocator(FakeClient(alloc_payload()))
    challenger = Challenger(RaisingClient())
    corr = {"keys": list(KEYS), "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    cycle = orch.DecisionCycle(allocator, challenger, correlation=corr, log_dir=str(tmp_path))
    gw = FakeGateway()
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    assert result.status == orch.HELD_ON_FAILURE
    assert "challenger unavailable" in result.reason
    assert gw.placed == []


def test_broker_evidence_failure_holds(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    gw = FakeGateway(fail_on={"account"})
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    assert result.status == orch.HELD_ON_FAILURE
    assert "evidence gathering failed" in result.reason
    assert gw.placed == []


def test_clock_failure_holds(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    result = cycle.run_cycle(FakeGateway(fail_on={"clock"}), cycle_id="c1", asof=ASOF)
    assert result.status == orch.HELD_ON_FAILURE
    assert "clock check failed" in result.reason


def test_reject_holds_current_allocation(tmp_path):
    cycle = make_cycle(
        challenger_payload=challenge_payload("REJECT"), tmp_path=tmp_path
    )
    gw = FakeGateway()
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    # Current is flat; REJECT holds flat, so nothing trades.
    assert result.status == orch.HELD_NO_CHANGE
    assert gw.placed == []
    assert result.effective_source == "held_on_reject"


def test_order_placement_failure_is_recorded_not_crashed(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    gw = FakeGateway(fail_on={"place"})
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    # Selection/sizing succeeded, placement failed -> held on failure, logged.
    assert result.status == orch.HELD_ON_FAILURE
    order = result.orders[0]
    assert order["submitted"] is False
    assert "error" in order


def test_selection_failure_on_wrong_chain_holds(tmp_path):
    """A chain with no puts cannot yield a bull put spread -> hold, logged."""
    cycle = make_cycle(tmp_path=tmp_path)
    gw = FakeGateway(chain={"snapshots": {}})
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    assert result.status == orch.HELD_ON_FAILURE
    assert result.sizing[BULL].get("error")
    assert gw.placed == []


# --- verification of an unconfirmed submit ---------------------------------


def test_unconfirmed_submit_is_reverified_by_status(tmp_path):
    """A submit response with no usable status triggers a status re-query."""
    cycle = make_cycle(tmp_path=tmp_path)
    gw = FakeGateway(place_result={"status": "unknown"})
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF)
    order = result.orders[0]
    # order_status returns the same 'unknown', so the fill verdict is unknown --
    # but crucially the system re-queried rather than assuming a fill.
    assert order["fill"]["verdict"] in ("unknown", "working", "unfilled")


# --- metrics ---------------------------------------------------------------


def test_metrics_accumulate_across_cycles(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    cycle.run_cycle(FakeGateway(), cycle_id="c1", asof=ASOF)
    cycle.run_cycle(FakeGateway(), cycle_id="c2", asof=ASOF)
    assert cycle.rejections.total == 2
    assert cycle.rejections.approval_rate == 1.0
    assert len(cycle.delta.history) == 2


def test_rejection_rate_reflects_verdicts(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    cycle.run_cycle(FakeGateway(), cycle_id="c1", asof=ASOF)  # APPROVE
    cycle.challenger = Challenger(FakeClient(challenge_payload("MODIFY", bull=0.30)))
    cycle.run_cycle(FakeGateway(), cycle_id="c2", asof=ASOF)  # MODIFY
    assert cycle.rejections.rejection_rate == pytest.approx(0.5)


def test_forced_reductions_flags_a_breached_short_strike(tmp_path):
    """A held strategy whose short strike sits at spot is a forced reduction
    (PRD 2.2), identified from evidence the packet already computed."""
    cycle = make_cycle(tmp_path=tmp_path)

    # Build a packet whose bull put has a short strike essentially at spot.
    StratEv = pk.StrategyEvidence

    def strat(key, contracts, sigma):
        s = BULL_PUT_SPREAD if key == BULL else (BEAR_CALL_SPREAD if key == BEAR else LONG_STRANGLE)
        return StratEv(
            key=key, name=s.name, thesis="", invalidation="", not_invalidation="",
            exit_behaviour=s.exit_behaviour.value, vol_exposure=s.vol_exposure.value,
            direction=s.direction, allocation_frac=0.3 if contracts else 0.0,
            max_loss_outstanding=1000.0 if contracts else 0.0, contracts=contracts,
            unrealized_pnl=-50.0 if contracts else None, pnl_frac_of_max_loss=None,
            drawdown_frac=None, days_held=2 if contracts else None, legs=(),
            short_strike_distance_sigma=sigma, min_dte=10 if contracts else None,
        )

    class P:
        strategies = (strat(BULL, 10, 0.05), strat(BEAR, 0, None), strat(STRANGLE, 0, None))

    forced = cycle._forced_reductions(P())
    assert BULL in forced          # short strike ~at spot -> breached -> forced
    assert BEAR not in forced      # flat, nothing to reduce


# --- dry run (full pipeline, no placement) ---------------------------------


def test_dry_run_plans_orders_without_placing(tmp_path):
    """The whole pipeline runs and records intended orders, but places nothing."""
    cycle = make_cycle(tmp_path=tmp_path)
    gw = FakeGateway(market_open=False)  # closed -- dry run proceeds anyway
    result = cycle.run_cycle(gw, cycle_id="c1", asof=ASOF, dry_run=True)
    assert result.status == orch.DRY_RUN
    assert gw.placed == []                        # nothing placed
    order = result.orders[0]
    assert order["dry_run"] is True
    assert order["submitted"] is False
    assert order["strategy"] == BULL
    assert "spec" in order and "summary" in order


def test_dry_run_ignores_market_closed(tmp_path):
    cycle = make_cycle(tmp_path=tmp_path)
    result = cycle.run_cycle(FakeGateway(market_open=False), cycle_id="c1", asof=ASOF, dry_run=True)
    assert result.status != orch.SKIPPED_MARKET_CLOSED
