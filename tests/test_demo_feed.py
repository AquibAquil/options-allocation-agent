"""Demo feed tests (supports PRD 4.3).

The feed is what the demo screen reads, so the contract that matters is: it
never invents data, it renders an empty state before the first cycle, it keeps
simulated data flagged and separate from scored P&L, and it's JSON-serialisable.
"""

from __future__ import annotations

import json

import pytest

from alloc_agent import demo_feed as df
from alloc_agent.strategies import BULL_PUT_SPREAD, BEAR_CALL_SPREAD, KEYS, LONG_STRANGLE

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key


def a_cycle_record(cid="c1", *, with_decision=True, actual_equity=1.02, delta=0.01):
    packet = {
        "market": {
            "symbol": "QQQ", "spot": 716.0, "spot_asof": "2026-08-31",
            "realized_vol": {"21d": 0.145}, "implied_vol": 0.175,
            "implied_vol_source": "CBOE VXN", "implied_vol_percentile_252d": 20.0,
            "iv_rv": {"spread": 0.03, "ratio": 1.2},
        },
        "portfolio": {
            "equity": 100000.0, "buying_power": 400000.0,
            "risk_budget_total": 20000.0, "risk_budget_utilisation": 0.9,
            "max_loss_as_frac_of_equity": 0.18,
        },
        "strategies": [
            {"key": BULL, "allocation_frac": 0.25, "contracts": 14, "pnl_frac_of_max_loss": 0.05,
             "unrealized_pnl": 250.0, "short_strike_distance_sigma": -1.9},
            {"key": BEAR, "allocation_frac": 0.25, "contracts": 14, "pnl_frac_of_max_loss": 0.03,
             "unrealized_pnl": 120.0, "short_strike_distance_sigma": 1.9},
            {"key": STRANGLE, "allocation_frac": 0.20, "contracts": 6, "pnl_frac_of_max_loss": -0.06,
             "unrealized_pnl": -240.0, "short_strike_distance_sigma": None},
        ],
    }
    record = {
        "cycle_id": cid, "asof": "2026-08-31", "status": "traded", "reason": "",
        "packet": packet,
        "metrics": {
            "this_cycle": {
                "cycle_id": cid, "actual_equity": actual_equity,
                "equal_weight_equity": 1.01, "cumulative_delta": delta,
            },
            "allocation_delta": {"allocation_delta": delta, "beats_equal_weight": delta > 0},
            "rejection": {"rejection_rate": 0.5, "total_reviews": 2},
        },
    }
    if with_decision:
        record["proposal"] = {
            "allocations": {BULL: 0.45, BEAR: 0.35, STRANGLE: 0.20},
            "reasoning": {BULL: "trend intact", BEAR: "weak upside", STRANGLE: "cheap protection"},
            "portfolio_rationale": "spreads negatively correlated",
            "model": "openai/gpt-oss-120b",
        }
        record["challenge"] = {
            "verdict": "MODIFY", "critique": "45% exceeds preferred exposure",
            "evidence_cited": ["iv_rv.ratio"],
            "modified_allocations": {BULL: 0.35, BEAR: 0.35, STRANGLE: 0.20},
        }
        record["gate_result"] = {
            "final": {BULL: 0.35, BEAR: 0.35, STRANGLE: 0.20},
            "adjustments": [{"detail": "bull put capped at 45%"}],
        }
        record["effective_source"] = "challenger_modified"
        record["orders"] = [
            {"strategy": BULL, "submitted": True, "summary": "Bull put spread ...",
             "fill": {"verdict": "filled", "atomic": True}},
        ]
    return record


# --- empty state -----------------------------------------------------------


def test_empty_log_gives_a_renderable_empty_state():
    feed = df.build_feed([])
    assert feed["has_data"] is False
    assert feed["state"] == "awaiting_first_cycle"
    assert feed["latest"] is None
    assert feed["series"] == []
    # Strategy labels are still present so the UI can render the shell.
    assert len(feed["strategies"]) == 3


# --- latest cycle ----------------------------------------------------------


def test_latest_carries_the_core_screen():
    feed = df.build_feed([a_cycle_record()])
    latest = feed["latest"]
    assert latest["portfolio"]["equity"] == 100000.0
    assert latest["market"]["iv_percentile_252d"] == 20.0
    assert latest["decision"]["proposed"][BULL] == 0.45
    assert latest["challenge"]["verdict"] == "MODIFY"
    assert latest["final_allocation"][BULL] == 0.35


def test_strategy_rows_show_the_change_arrow():
    feed = df.build_feed([a_cycle_record()])
    rows = {r["key"]: r for r in feed["latest"]["strategies"]}
    bull = rows[BULL]
    # current 0.25 -> final 0.35, a +0.10 change (the "25% -> 35%" arrow).
    assert bull["current_alloc"] == 0.25
    assert bull["final_alloc"] == 0.35
    assert bull["change"] == pytest.approx(0.10)
    assert bull["proposed_alloc"] == 0.45     # what the allocator wanted
    assert bull["reasoning"] == "trend intact"


def test_rows_carry_pnl_and_names():
    feed = df.build_feed([a_cycle_record()])
    rows = {r["key"]: r for r in feed["latest"]["strategies"]}
    assert rows[BULL]["name"] == "Bull put spread"
    assert rows[STRANGLE]["pnl_frac_of_max_loss"] == -0.06


def test_prefers_the_latest_decision_over_a_later_hold():
    """A market-closed skip after a real decision must not blank the screen."""
    decision = a_cycle_record("decision")
    skip = {"cycle_id": "skip", "asof": "2026-08-31", "status": "skipped_market_closed"}
    feed = df.build_feed([decision, skip])
    assert feed["latest"]["cycle_id"] == "decision"   # the real decision, not the skip
    assert len(feed["cycles"]) == 2                    # but both appear in the index


# --- series and metrics ----------------------------------------------------


def test_series_tracks_the_allocation_delta():
    records = [
        a_cycle_record("c1", actual_equity=1.01, delta=0.005),
        a_cycle_record("c2", actual_equity=1.03, delta=0.02),
    ]
    feed = df.build_feed(records)
    assert [p["cumulative_delta"] for p in feed["series"]] == [0.005, 0.02]


def test_headline_metrics_come_from_the_latest_record():
    feed = df.build_feed([a_cycle_record("c1", delta=0.005), a_cycle_record("c2", delta=0.02)])
    assert feed["metrics"]["allocation_delta"]["allocation_delta"] == 0.02


# --- simulated data stays flagged and separate -----------------------------


def test_shock_simulations_are_flagged_simulated():
    shocks = {
        "scenarios": [
            {"name": "vol_shock", "regime": "Volatility expansion",
             "expectation": "cut spreads", "final_allocation": {BULL: 0.3, BEAR: 0.3, STRANGLE: 0.0},
             "challenge": {"verdict": "APPROVE"}},
        ]
    }
    feed = df.build_feed([a_cycle_record()], shocks=shocks)
    assert feed["shock_simulations"]["simulated"] is True
    assert feed["shock_simulations"]["scenarios"][0]["regime"] == "Volatility expansion"


def test_correlation_is_carried_through():
    corr = {"keys": list(KEYS), "matrix": [[1, -0.88, -0.45], [-0.88, 1, 0.01], [-0.45, 0.01, 1]]}
    feed = df.build_feed([a_cycle_record()], correlation=corr)
    assert feed["correlation"]["matrix"][0][1] == -0.88


# --- serialisable ----------------------------------------------------------


def test_feed_is_json_serialisable():
    feed = df.build_feed([a_cycle_record()], correlation={"keys": list(KEYS), "matrix": [[1]]})
    json.dumps(feed)   # must not raise
