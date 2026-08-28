"""Contract selection tests (PRD 2.3).

Run against a REAL QQQ put chain captured live from the Alpaca MCP
get_option_chain call (Sep 11 expiry, spot ~716.91). Selection is fixed, not
adaptive, so these assert the rules land on the contracts a human would pick by
hand from that chain.

The final test walks the whole pipeline on real data -- chain -> selection ->
sizing quote -> sizing plan -> order spec -> MCP kwargs -- which is the "one
strategy end to end" milestone minus the live place_option_order call.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from alloc_agent import execution as ex
from alloc_agent import selection as sel
from alloc_agent import sizing
from alloc_agent.strategies import BULL_PUT_SPREAD

# The chain was captured on 2026-08-28; from a Monday-08-31 decision this expiry
# is 11 DTE, inside the 7-14 band.
ASOF = dt.date(2026, 8, 31)
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "qqq_put_chain_sep11.json")


@pytest.fixture
def put_chain():
    with open(FIXTURE) as fh:
        return json.load(fh)


def test_fixture_flattens_to_candidates(put_chain):
    cands = sel.to_candidates(put_chain, asof=ASOF)
    assert len(cands) == 16
    assert all(c.right == "put" for c in cands)
    assert all(c.dte == 11 for c in cands)
    assert all(c.delta is not None and c.mid > 0 for c in cands)


def test_vertical_picks_the_25_delta_short_strike(put_chain):
    """Closest delta to 0.25 in this chain is the 702 put (-0.2486)."""
    selection = sel.select_vertical(put_chain, BULL_PUT_SPREAD, asof=ASOF)
    short = selection.short_candidate
    assert short.strike == 702.0
    assert abs(abs(short.delta) - 0.25) < 0.01


def test_vertical_long_strike_is_three_below(put_chain):
    """Bull put spec buys three strikes below the short; 702 -> 699."""
    selection = sel.select_vertical(put_chain, BULL_PUT_SPREAD, asof=ASOF)
    strikes = sorted(c.strike for c in selection.candidates)
    assert strikes == [699.0, 702.0]


def test_selected_legs_have_correct_sides(put_chain):
    selection = sel.select_vertical(put_chain, BULL_PUT_SPREAD, asof=ASOF)
    by_action = {l.action: l for l in selection.legs}
    assert by_action["sell"].strike == 702.0
    assert by_action["buy"].strike == 699.0
    assert by_action["sell"].occ_symbol == "QQQ260911P00702000"


def test_dispatch_matches_direct_selector(put_chain):
    a = sel.select(put_chain, BULL_PUT_SPREAD, asof=ASOF)
    b = sel.select_vertical(put_chain, BULL_PUT_SPREAD, asof=ASOF)
    assert a.legs == b.legs


def test_expiry_out_of_band_raises(put_chain):
    """From far in the future, the Sep 11 expiry is past the 14-day band."""
    with pytest.raises(sel.NoContractFound, match="DTE band"):
        sel.select_vertical(put_chain, BULL_PUT_SPREAD, asof=dt.date(2026, 8, 1))


def test_sizing_quote_carries_real_delta_and_spread(put_chain):
    selection = sel.select_vertical(put_chain, BULL_PUT_SPREAD, asof=ASOF)
    quote = sel.build_sizing_quote(selection, BULL_PUT_SPREAD)
    # 702 short delta from the live chain.
    assert quote.short_leg_delta == pytest.approx(-0.2486)
    # Widest leg bid-ask across the two legs (702: 0.06, 699: 0.09).
    assert quote.bid_ask_spread == pytest.approx(0.09, abs=1e-9)
    assert quote.strike_width == pytest.approx(3.0)


def test_full_pipeline_on_real_chain(put_chain):
    """chain -> selection -> quote -> plan -> order spec -> MCP kwargs."""
    strategy = BULL_PUT_SPREAD

    # 1. Fixed selection from the live chain.
    selection = sel.select_vertical(put_chain, strategy, asof=ASOF)

    # 2. Build the sizing quote and verify the chain (real deltas/spread).
    quote = sel.build_sizing_quote(selection, strategy)
    sizing.verify_chain(quote, strategy)  # must not raise

    # 3. Size a small test position: one contract.
    plan = sizing.size_strategy(
        strategy.key,
        target_alloc_frac=0.02,          # tiny, just enough for one contract
        risk_budget_total=20_000.0,
        quote=quote,
        contracts_current=0,
    )
    assert plan.contracts == 1
    assert plan.action == "open"

    # 4. Build the order spec.
    spec = ex.build_order(plan, selection.legs, slippage=0.02)
    assert spec.strategy_key == strategy.key
    assert spec.qty == 1
    assert spec.is_credit                # a bull put spread collects
    assert len(spec.legs) == 2

    # 5. The MCP kwargs are well-formed for place_option_order.
    kw = spec.to_mcp_kwargs()
    assert kw["order_class"] == "mleg"
    assert {l["symbol"] for l in kw["legs"]} == {
        "QQQ260911P00702000",
        "QQQ260911P00699000",
    }
    sides = {l["symbol"]: l["side"] for l in kw["legs"]}
    assert sides["QQQ260911P00702000"] == "sell"
    assert sides["QQQ260911P00699000"] == "buy"
    # Net credit around 0.50/share after slippage.
    assert -0.60 < float(kw["limit_price"]) < -0.40


def test_max_loss_is_defined_risk_on_real_chain(put_chain):
    """The 702/699 spread's max loss must be width minus credit."""
    selection = sel.select_vertical(put_chain, BULL_PUT_SPREAD, asof=ASOF)
    quote = sel.build_sizing_quote(selection, BULL_PUT_SPREAD)
    # short mid ~3.55, long mid ~3.025 -> credit ~0.525 -> $52.5
    # max loss = 300 - 52.5 = ~247.5
    assert 240.0 < quote.max_loss_per_contract < 255.0
