"""Evidence packet tests.

The load-bearing one is `test_packet_contains_no_verdicts`. PRD 2.1 is explicit
that this layer emits evidence and not conclusions, and the failure mode is
quiet: a single boolean like `thesis_holds` would reduce the allocator to
writing prose around it.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from alloc_agent.config import RISK
from alloc_agent.evidence import packet as pk
from alloc_agent.strategies import BULL_PUT_SPREAD, KEYS, LONG_STRANGLE
from conftest import ASOF, synthetic_bars, synthetic_vxn

EQUITY = 100_000.0
# The synthetic walk drifts, so strikes are derived from its real last close
# rather than assumed. Otherwise "2% below spot" can land above it.
SPOT = synthetic_bars()[-1]["c"]


def account(equity=EQUITY, buying_power=200_000.0):
    return pk.AccountSnapshot(
        equity=equity, buying_power=buying_power, options_buying_power=buying_power
    )


def bull_put_position(contracts=10, spot=SPOT, unrealized=250.0, opened="2026-08-24"):
    """A 10-lot 3-wide bull put spread, short strike roughly 2% below spot."""
    short_k, long_k = round(spot * 0.98), round(spot * 0.98) - 3
    expiry = (ASOF + dt.timedelta(days=10)).isoformat()
    return pk.PositionSnapshot(
        strategy_key=BULL_PUT_SPREAD.key,
        contracts=contracts,
        legs=(
            pk.LegSnapshot(
                occ_symbol=f"QQQ260907P00{short_k}000",
                right="put",
                action="sell",
                strike=float(short_k),
                expiry=expiry,
                contracts=contracts,
                delta=-0.25,
                mid=1.40,
                implied_vol=0.21,
                bid=1.35,
                ask=1.45,
            ),
            pk.LegSnapshot(
                occ_symbol=f"QQQ260907P00{long_k}000",
                right="put",
                action="buy",
                strike=float(long_k),
                expiry=expiry,
                contracts=contracts,
                delta=-0.18,
                mid=0.95,
                implied_vol=0.22,
                bid=0.90,
                ask=1.00,
            ),
        ),
        max_loss_per_contract=255.0,   # 3.00 width x 100 minus 45 credit
        entry_premium=45.0,
        opened_at=opened,
        unrealized_pnl=unrealized,
        equity_curve=(0.0, -120.0, 80.0, unrealized),
    )


def build(positions=None, acct=None, bars=None, vxn=None):
    return pk.build_packet(
        cycle_id="2026-08-28T10:00-04:00",
        symbol="QQQ",
        bars=bars or synthetic_bars(),
        vol_index_rows=vxn or synthetic_vxn(),
        positions=positions or {},
        account=acct or account(),
        correlation={"keys": list(KEYS), "matrix": [[1, -0.84, -0.64], [-0.84, 1, 0.15], [-0.64, 0.15, 1]]},
        asof=ASOF,
    )


# --- market ----------------------------------------------------------------


def test_market_evidence_is_populated(bars, vxn):
    m = pk.build_market_evidence(symbol="QQQ", bars=bars, vol_index_rows=vxn)
    assert m.spot == pytest.approx(bars[-1]["c"])
    assert m.spot_asof == bars[-1]["t"]
    assert m.returns["1d"] is not None
    assert m.realized_vol["21d"] > 0
    assert m.parkinson_vol_21d > 0
    assert 0.0 <= m.implied_vol_percentile_252d <= 100.0
    assert m.implied_vol_source == "CBOE VXN"


def test_implied_vol_is_converted_from_index_points(bars, vxn):
    """VXN prints 20.24 meaning 20.24% -- a raw 20.24 would be 2024% vol."""
    m = pk.build_market_evidence(symbol="QQQ", bars=bars, vol_index_rows=vxn)
    assert m.implied_vol == pytest.approx(vxn[-1]["close"] / 100.0)
    assert 0.01 < m.implied_vol < 3.0


def test_iv_rv_relationship_is_reported_not_judged(bars, vxn):
    m = pk.build_market_evidence(symbol="QQQ", bars=bars, vol_index_rows=vxn)
    assert set(m.iv_rv) == {"implied", "realized", "spread", "ratio"}
    assert m.iv_rv["spread"] == pytest.approx(m.iv_rv["implied"] - m.iv_rv["realized"])


def test_percentile_excludes_today_from_its_own_history(bars):
    """Today must not be in the sample it is ranked against."""
    vxn = synthetic_vxn()
    spike = list(vxn)
    spike[-1] = dict(spike[-1], close=max(r["close"] for r in vxn) * 2.0)
    m = pk.build_market_evidence(symbol="QQQ", bars=bars, vol_index_rows=spike)
    assert m.implied_vol_percentile_252d == 100.0


def test_market_evidence_rejects_a_single_bar(vxn):
    with pytest.raises(ValueError, match="at least two bars"):
        pk.build_market_evidence(symbol="QQQ", bars=[{"t": "x", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}], vol_index_rows=vxn)


# --- strategies ------------------------------------------------------------


def test_flat_strategy_keeps_its_thesis_and_reports_zero():
    """A cut strategy must stay visible, or it can never be funded again."""
    p = build()
    assert len(p.strategies) == 3
    for s in p.strategies:
        assert s.allocation_frac == 0.0
        assert s.contracts == 0
        assert s.max_loss_outstanding == 0.0
        assert s.unrealized_pnl is None
        assert s.thesis and s.invalidation and s.not_invalidation


def test_allocation_is_a_share_of_the_risk_budget():
    """PRD 2.2: max loss is the denominator, not capital deployed."""
    p = build({BULL_PUT_SPREAD.key: bull_put_position(contracts=10)})
    s = next(s for s in p.strategies if s.key == BULL_PUT_SPREAD.key)
    budget = EQUITY * RISK.total_budget_frac
    assert s.max_loss_outstanding == pytest.approx(2550.0)
    assert s.allocation_frac == pytest.approx(2550.0 / budget)


def test_pnl_is_expressed_against_risk_taken():
    p = build({BULL_PUT_SPREAD.key: bull_put_position(unrealized=255.0)})
    s = next(s for s in p.strategies if s.key == BULL_PUT_SPREAD.key)
    assert s.pnl_frac_of_max_loss == pytest.approx(255.0 / 2550.0)


def test_days_held_is_counted_from_entry():
    p = build({BULL_PUT_SPREAD.key: bull_put_position(opened="2026-08-24")})
    s = next(s for s in p.strategies if s.key == BULL_PUT_SPREAD.key)
    assert s.days_held == 4


def test_legs_carry_dte_and_strike_distance():
    p = build({BULL_PUT_SPREAD.key: bull_put_position()})
    s = next(s for s in p.strategies if s.key == BULL_PUT_SPREAD.key)
    assert len(s.legs) == 2
    for leg in s.legs:
        assert leg.dte == 10
        assert leg.strike_distance_pct < 0        # both puts sit below spot
        assert leg.strike_distance_sigma < 0
        assert leg.bid_ask_spread == pytest.approx(0.10)


def test_short_strike_distance_picks_the_nearest_strike():
    """The threatened strike is the closest one, whichever side it is on."""
    p = build({BULL_PUT_SPREAD.key: bull_put_position()})
    s = next(s for s in p.strategies if s.key == BULL_PUT_SPREAD.key)
    short_leg = next(leg for leg in s.legs if leg.action == "sell")
    assert s.short_strike_distance_sigma == pytest.approx(short_leg.strike_distance_sigma)


def test_min_dte_is_the_soonest_expiry():
    p = build({BULL_PUT_SPREAD.key: bull_put_position()})
    s = next(s for s in p.strategies if s.key == BULL_PUT_SPREAD.key)
    assert s.min_dte == 10


def test_drawdown_is_measured_against_max_loss():
    p = build({BULL_PUT_SPREAD.key: bull_put_position()})
    s = next(s for s in p.strategies if s.key == BULL_PUT_SPREAD.key)
    # curve (0, -120, 80, 250) shifted by max loss 2550: peak 2630, trough 2430
    assert s.drawdown_frac == pytest.approx(120.0 / 2550.0, rel=1e-6)


def test_unknown_strategy_key_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy keys"):
        build({"iron_condor": bull_put_position()})


# --- portfolio -------------------------------------------------------------


def test_risk_budget_arithmetic():
    p = build({BULL_PUT_SPREAD.key: bull_put_position(contracts=10)})
    port = p.portfolio
    assert port.risk_budget_total == pytest.approx(EQUITY * RISK.total_budget_frac)
    assert port.risk_budget_consumed == pytest.approx(2550.0)
    assert port.risk_budget_available == pytest.approx(port.risk_budget_total - 2550.0)
    assert port.risk_budget_utilisation == pytest.approx(2550.0 / port.risk_budget_total)


def test_buying_power_is_carried_separately_from_the_budget():
    """PRD 2.2: buying power is a hard gate, never the denominator."""
    p = build({BULL_PUT_SPREAD.key: bull_put_position()}, acct=account(buying_power=1234.0))
    assert p.portfolio.buying_power == 1234.0
    assert p.portfolio.risk_budget_total == pytest.approx(EQUITY * RISK.total_budget_frac)


def test_max_loss_against_equity_is_reported():
    p = build({BULL_PUT_SPREAD.key: bull_put_position()})
    assert p.portfolio.max_loss_as_frac_of_equity == pytest.approx(2550.0 / EQUITY)


# --- constraints and packet ------------------------------------------------


def test_constraints_are_stated_as_input():
    c = build().constraints
    assert c.total_budget_frac_of_equity == RISK.total_budget_frac
    assert c.per_strategy_max_frac_of_budget == RISK.per_strategy_max
    assert c.snap_to_zero_below_frac == RISK.snap_to_zero_below
    assert c.adjustment_threshold_frac == RISK.adjustment_threshold


def test_correlation_is_carried_through():
    p = build()
    assert p.correlation["keys"] == list(KEYS)
    assert p.correlation["matrix"][0][1] == pytest.approx(-0.84)


def test_stale_iv_reference_is_noted_not_hidden():
    stale = synthetic_vxn(end=ASOF - dt.timedelta(days=5))
    p = build(vxn=stale)
    assert any("behind spot" in n for n in p.notes)


def test_packet_serialises_to_json():
    p = build({BULL_PUT_SPREAD.key: bull_put_position()})
    parsed = json.loads(p.to_json())
    assert parsed["market"]["symbol"] == "QQQ"
    assert len(parsed["strategies"]) == 3
    assert parsed["portfolio"]["risk_budget_total"] > 0


def test_packet_contains_no_verdicts():
    """PRD 2.1: no booleans, no verdict-shaped keys anywhere in the payload.

    `bars_are_complete_sessions` is a provenance flag about the data, not a
    judgement about a strategy, so it is the one permitted exception.
    """
    payload = json.loads(build({BULL_PUT_SPREAD.key: bull_put_position()}).to_json())
    banned = ("holds", "valid", "should", "recommend", "verdict", "signal", "score")
    allowed_bools = {"bars_are_complete_sessions"}
    # Thesis text fields, carried through verbatim from the library. They
    # describe conditions in prose; nothing here evaluates them.
    thesis_fields = {"invalidation", "not_invalidation"}

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}"
                # Thesis prose legitimately contains these words; keys must not.
                if key not in thesis_fields:
                    assert not any(
                        b in key.lower() for b in banned
                    ), f"verdict-shaped key: {here}"
                if isinstance(value, bool):
                    assert key in allowed_bools, f"boolean verdict at {here}"
                walk(value, here)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(payload)
