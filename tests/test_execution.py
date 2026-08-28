"""Execution tests (PRD 2.5, 2.6, 2.7).

The two cases that carry the most weight:

  test_idempotency_key_is_deterministic -- a timed-out retry must not open a
  second position.

  the interpret_fill leg-imbalance cases -- an unpaired short leg is the naked
  risk a defined-risk spread exists to prevent, and detecting it after
  submission is the whole point of fill verification.

Real OCC symbols from Alpaca's QQQ chain are used throughout, so the symbol
maths is checked against ground truth rather than a guess.
"""

from __future__ import annotations

import pytest

from alloc_agent import execution as ex
from alloc_agent import sizing
from alloc_agent.strategies import BULL_PUT_SPREAD, LONG_STRANGLE

BULL, STRANGLE = BULL_PUT_SPREAD.key, LONG_STRANGLE.key

# Two real symbols observed from the live Sep 11 QQQ chain.
SHORT_PUT = "QQQ260911P00702000"
LONG_PUT = "QQQ260911P00699000"


# --- OCC symbols -----------------------------------------------------------


def test_build_occ_matches_alpaca_chain():
    assert ex.build_occ_symbol("QQQ", "2026-09-11", "put", 702.0) == SHORT_PUT
    assert ex.build_occ_symbol("QQQ", "2026-09-11", "put", 699.0) == LONG_PUT
    assert ex.build_occ_symbol("QQQ", "2026-09-11", "call", 730.0) == "QQQ260911C00730000"


def test_occ_round_trips():
    for sym in (SHORT_PUT, LONG_PUT, "QQQ260911C00730000"):
        p = ex.parse_occ_symbol(sym)
        assert ex.build_occ_symbol(p["underlying"], p["expiry"], p["right"], p["strike"]) == sym


def test_parse_reads_the_real_symbol():
    p = ex.parse_occ_symbol(SHORT_PUT)
    assert p == {"underlying": "QQQ", "expiry": "2026-09-11", "right": "put", "strike": 702.0}


def test_occ_rejects_off_grid_strike():
    with pytest.raises(ValueError, match="grid"):
        ex.build_occ_symbol("QQQ", "2026-09-11", "put", 702.0001)


# --- helpers ---------------------------------------------------------------


def _mk_plan(key, delta, action, target):
    # Direct construction keeps the test independent of sizing's internals.
    return sizing.SizingPlan(
        strategy_key=key,
        target_alloc_frac=target,
        target_max_loss=target * 20_000.0,
        contracts=delta if delta > 0 else 0,
        contracts_current=0 if delta > 0 else -delta,
        contract_delta=delta,
        actual_max_loss=abs(delta) * 248.0,
        actual_alloc_frac=abs(delta) * 248.0 / 20_000.0,
        estimated_margin=abs(delta) * 248.0,
        action=action,
    )


def bull_put_legs():
    return (
        ex.ContractLeg(SHORT_PUT, "put", "sell", 702.0, "2026-09-11", mid=3.55),
        ex.ContractLeg(LONG_PUT, "put", "buy", 699.0, "2026-09-11", mid=3.03),
    )


# --- order construction ----------------------------------------------------


def test_credit_spread_limit_price_is_negative():
    """A bull put spread collects, so the net limit must be a credit."""
    spec = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs(), slippage=0.0)
    assert spec.is_credit
    assert spec.limit_price == pytest.approx(-(3.55 - 3.03))  # -0.52


def test_slippage_reduces_credit_never_flips_it():
    spec = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs(), slippage=0.05)
    # Accept less credit: -0.52 -> -0.47, still a credit.
    assert spec.limit_price == pytest.approx(-0.47)
    assert spec.is_credit


def test_slippage_cannot_turn_a_thin_credit_into_a_debit():
    thin = (
        ex.ContractLeg(SHORT_PUT, "put", "sell", 702.0, "2026-09-11", mid=3.04),
        ex.ContractLeg(LONG_PUT, "put", "buy", 699.0, "2026-09-11", mid=3.03),
    )
    spec = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), thin, slippage=0.10)
    assert spec.limit_price < 0  # net_cost -0.01, guard keeps it a credit


def test_debit_strangle_limit_price_is_positive():
    legs = (
        ex.ContractLeg("QQQ261009C00760000", "call", "buy", 760.0, "2026-10-09", mid=4.10),
        ex.ContractLeg("QQQ261009P00675000", "put", "buy", 675.0, "2026-10-09", mid=3.90),
    )
    plan = _mk_plan(STRANGLE, 2, "open", 0.15)
    spec = ex.build_order(plan, legs, slippage=0.0)
    assert not spec.is_credit
    assert spec.limit_price == pytest.approx(8.00)  # both bought


def test_legs_carry_open_intents():
    spec = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs())
    intents = {leg.side: leg.position_intent for leg in spec.legs}
    assert intents == {"sell": "sell_to_open", "buy": "buy_to_open"}


def test_reduce_reverses_each_leg():
    """Closing reverses side and switches to *_to_close intents."""
    spec = ex.build_order(_mk_plan(BULL, -1, "reduce", 0.30), bull_put_legs(), intent="reduce")
    by_symbol = {leg.occ_symbol: leg for leg in spec.legs}
    # The short put (sold to open) is bought to close.
    assert by_symbol[SHORT_PUT].side == "buy"
    assert by_symbol[SHORT_PUT].position_intent == "buy_to_close"
    assert by_symbol[LONG_PUT].side == "sell"
    assert by_symbol[LONG_PUT].position_intent == "sell_to_close"


def test_qty_is_the_absolute_delta():
    spec = ex.build_order(_mk_plan(BULL, 7, "open", 0.40), bull_put_legs())
    assert spec.qty == 7


def test_zero_delta_cannot_build_an_order():
    with pytest.raises(ValueError, match="nothing to trade"):
        ex.build_order(_mk_plan(BULL, 0, "hold", 0.40), bull_put_legs())


def test_scale_down_only_full_close_is_refused():
    """PRD 2.3 last-ditch guard: never slam a live strangle shut on a live target."""
    legs = (
        ex.ContractLeg("QQQ261009C00760000", "call", "buy", 760.0, "2026-10-09", mid=4.1),
        ex.ContractLeg("QQQ261009P00675000", "put", "buy", 675.0, "2026-10-09", mid=3.9),
    )
    plan = _mk_plan(STRANGLE, -2, "close", 0.15)  # target still positive
    with pytest.raises(ValueError, match="scale-down-only"):
        ex.build_order(plan, legs, intent="close")


# --- MCP kwargs ------------------------------------------------------------


def test_mcp_kwargs_shape_matches_place_option_order():
    spec = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs())
    kw = spec.to_mcp_kwargs()
    assert kw["order_class"] == "mleg"
    assert kw["type"] == "limit"
    assert kw["time_in_force"] == "day"
    assert kw["qty"] == "1"
    assert kw["client_order_id"].startswith("alloc-")
    assert len(kw["legs"]) == 2
    for leg in kw["legs"]:
        assert set(leg) == {"symbol", "ratio_qty", "side", "position_intent"}
        assert leg["ratio_qty"] == "1"
    # limit price serialised to 2dp string
    assert kw["limit_price"] == f"{spec.limit_price:.2f}"


# --- idempotency (PRD 2.5) -------------------------------------------------


def test_idempotency_key_is_deterministic():
    """Same structure and intent -> same id, so a timed-out retry is safe."""
    a = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs())
    b = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs())
    assert a.client_order_id == b.client_order_id


def test_different_strikes_get_a_different_key():
    """A re-picked structure is a different order and must not be deduped."""
    other = (
        ex.ContractLeg("QQQ260911P00701000", "put", "sell", 701.0, "2026-09-11", mid=3.4),
        ex.ContractLeg("QQQ260911P00698000", "put", "buy", 698.0, "2026-09-11", mid=2.9),
    )
    a = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs())
    b = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), other)
    assert a.client_order_id != b.client_order_id


def test_close_and_open_get_different_keys():
    a = ex.build_order(_mk_plan(BULL, 1, "open", 0.40), bull_put_legs())
    b = ex.build_order(_mk_plan(BULL, -1, "reduce", 0.30), bull_put_legs(), intent="reduce")
    assert a.client_order_id != b.client_order_id


# --- fill interpretation (PRD 2.5, 2.7) ------------------------------------


def test_clean_atomic_fill():
    order = {
        "id": "o1", "client_order_id": "alloc-x", "status": "filled",
        "qty": "1", "filled_qty": "1",
        "legs": [
            {"symbol": SHORT_PUT, "filled_qty": "1"},
            {"symbol": LONG_PUT, "filled_qty": "1"},
        ],
    }
    r = ex.interpret_fill(order)
    assert r.verdict == "filled"
    assert r.atomic is True
    assert not r.has_leg_imbalance
    assert r.is_terminal


def test_leg_imbalance_is_flagged_as_naked_risk():
    """The day-one question: one leg filled, the other did not."""
    order = {
        "id": "o2", "status": "partially_filled", "qty": "1", "filled_qty": "1",
        "legs": [
            {"symbol": SHORT_PUT, "filled_qty": "1"},  # short filled
            {"symbol": LONG_PUT, "filled_qty": "0"},   # long did not -> naked short
        ],
    }
    r = ex.interpret_fill(order)
    assert r.verdict == "partial_legs"
    assert r.has_leg_imbalance
    assert "unpaired" in r.detail


def test_filled_status_with_uneven_legs_is_caught():
    """Defensive: 'filled' should never co-occur with unequal legs, but if it
    does, the imbalance wins."""
    order = {
        "id": "o3", "status": "filled", "qty": "2", "filled_qty": "2",
        "legs": [
            {"symbol": SHORT_PUT, "filled_qty": "2"},
            {"symbol": LONG_PUT, "filled_qty": "1"},
        ],
    }
    r = ex.interpret_fill(order)
    assert r.verdict == "partial_legs"


def test_resting_order_is_working_not_failed():
    r = ex.interpret_fill({"id": "o4", "status": "new", "qty": "1", "filled_qty": "0", "legs": []})
    assert r.verdict == "working"
    assert not r.is_terminal


def test_rejected_order_carries_reason():
    r = ex.interpret_fill({"id": "o5", "status": "rejected", "reject_reason": "insufficient buying power"})
    assert r.verdict == "rejected"
    assert "buying power" in r.detail
    assert r.is_terminal


def test_balanced_partial_fill_is_working():
    """Both legs filled equally but short of the full size: fine, still working."""
    order = {
        "id": "o6", "status": "partially_filled", "qty": "10", "filled_qty": "4",
        "legs": [
            {"symbol": SHORT_PUT, "filled_qty": "4"},
            {"symbol": LONG_PUT, "filled_qty": "4"},
        ],
    }
    r = ex.interpret_fill(order)
    assert r.verdict == "working"
    assert r.atomic is True


def test_interpret_tolerates_missing_fields():
    """Reads a live account; must not crash on an unexpected shape."""
    r = ex.interpret_fill({})
    assert r.verdict == "unknown"
    assert r.filled_qty == 0
    assert r.atomic is None


def test_canceled_with_no_fill_is_unfilled():
    r = ex.interpret_fill({"id": "o7", "status": "canceled", "qty": "1", "filled_qty": "0"})
    assert r.verdict == "unfilled"


# --- chain snapshot -> leg -------------------------------------------------


def test_leg_from_live_quote_uses_mid():
    snap = {"latestQuote": {"bp": 3.52, "ap": 3.58}, "latestTrade": {"p": 3.55}}
    leg = ex.chain_leg_from_quote(snap, SHORT_PUT, "sell")
    assert leg.mid == pytest.approx(3.55)
    assert leg.strike == 702.0
    assert leg.right == "put"


def test_leg_falls_back_to_last_trade_when_quote_is_empty():
    snap = {"latestQuote": {}, "latestTrade": {"p": 3.40}}
    leg = ex.chain_leg_from_quote(snap, SHORT_PUT, "sell")
    assert leg.mid == pytest.approx(3.40)
