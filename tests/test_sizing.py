"""Sizing tests (PRD 2.1, 2.2).

Two things must not blur together here: the risk budget, which is denominated
in max loss, and buying power, which is a broker constraint. A plan can be
inside the budget and still unaffordable, and the two failures need different
messages because they call for different responses.
"""

from __future__ import annotations

import pytest

from alloc_agent import sizing
from alloc_agent.config import RISK
from alloc_agent.strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, LONG_STRANGLE

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key

BUDGET = 20_000.0   # 20% of a $100k account


def spread_quote(key=BULL, *, width=3.0, credit=45.0, dte=10, delta=-0.25, spread=0.05):
    return sizing.ChainQuote(
        strategy_key=key,
        max_loss_per_contract=width * 100.0 - credit,
        premium_per_contract=credit,
        strike_width=width,
        min_dte=dte,
        short_leg_delta=delta,
        bid_ask_spread=spread,
    )


def strangle_quote(*, debit=420.0, dte=30, spread=0.08):
    return sizing.ChainQuote(
        strategy_key=STRANGLE,
        max_loss_per_contract=debit,
        premium_per_contract=debit,
        strike_width=None,
        min_dte=dte,
        short_leg_delta=None,
        bid_ask_spread=spread,
    )


# --- chain verification (PRD 2.5) ------------------------------------------


def test_clean_quote_verifies():
    sizing.verify_chain(spread_quote(), BULL_PUT_SPREAD)
    sizing.verify_chain(strangle_quote(), LONG_STRANGLE)


def test_wrong_strategy_is_rejected():
    with pytest.raises(sizing.ChainRejected, match="quote is for"):
        sizing.verify_chain(spread_quote(key=BEAR), BULL_PUT_SPREAD)


def test_dte_outside_the_band_is_rejected():
    """A 30-day spread would be nearly inert across a four-day window."""
    with pytest.raises(sizing.ChainRejected, match="DTE 30 outside 7-14"):
        sizing.verify_chain(spread_quote(dte=30), BULL_PUT_SPREAD)


def test_short_dated_strangle_is_rejected():
    """Short-dated long options bleed aggressively; the strangle wants time."""
    with pytest.raises(sizing.ChainRejected, match="DTE 7 outside 25-35"):
        sizing.verify_chain(strangle_quote(dte=7), LONG_STRANGLE)


def test_max_loss_that_does_not_match_the_structure_is_rejected():
    """The defining check: is this actually the defined-risk trade it claims?"""
    bad = sizing.ChainQuote(
        strategy_key=BULL,
        max_loss_per_contract=1000.0,     # inconsistent with a 3-wide spread
        premium_per_contract=45.0,
        strike_width=3.0,
        min_dte=10,
        short_leg_delta=-0.25,
        bid_ask_spread=0.05,
    )
    with pytest.raises(sizing.ChainRejected, match="not what it claims to be"):
        sizing.verify_chain(bad, BULL_PUT_SPREAD)


def test_strangle_max_loss_must_equal_the_debit():
    bad = sizing.ChainQuote(
        strategy_key=STRANGLE,
        max_loss_per_contract=900.0,
        premium_per_contract=420.0,
        strike_width=None,
        min_dte=30,
        short_leg_delta=None,
        bid_ask_spread=0.08,
    )
    with pytest.raises(sizing.ChainRejected, match="must equal the debit"):
        sizing.verify_chain(bad, LONG_STRANGLE)


def test_drifted_short_delta_is_rejected():
    """A 45-delta short strike is a different trade from a 25-delta one."""
    with pytest.raises(sizing.ChainRejected, match="from the 0.25 target"):
        sizing.verify_chain(spread_quote(delta=-0.45), BULL_PUT_SPREAD)


def test_small_delta_drift_is_tolerated():
    sizing.verify_chain(spread_quote(delta=-0.30), BULL_PUT_SPREAD)


def test_wide_bid_ask_is_rejected():
    with pytest.raises(sizing.ChainRejected, match="bid-ask"):
        sizing.verify_chain(spread_quote(spread=0.40), BULL_PUT_SPREAD)


def test_credit_exceeding_width_is_rejected():
    bad = spread_quote(width=3.0, credit=350.0)
    with pytest.raises(sizing.ChainRejected, match="max loss per contract is"):
        sizing.verify_chain(bad, BULL_PUT_SPREAD)


# --- contract counts -------------------------------------------------------


def test_contracts_come_from_max_loss_not_capital():
    """PRD 2.2: allocation is a share of the max-loss budget."""
    quote = spread_quote()   # 255 max loss per contract
    plan = sizing.size_strategy(
        BULL, target_alloc_frac=0.40, risk_budget_total=BUDGET, quote=quote
    )
    assert plan.contracts == int(0.40 * BUDGET // 255.0)   # 31
    assert plan.actual_max_loss == pytest.approx(31 * 255.0)


def test_contracts_round_down_never_up():
    """Rounding up would put the position over its allocation."""
    quote = spread_quote()
    plan = sizing.size_strategy(
        BULL, target_alloc_frac=0.40, risk_budget_total=BUDGET, quote=quote
    )
    assert plan.actual_max_loss <= 0.40 * BUDGET


def test_actual_allocation_is_reported_alongside_target():
    quote = spread_quote()
    plan = sizing.size_strategy(
        BULL, target_alloc_frac=0.40, risk_budget_total=BUDGET, quote=quote
    )
    assert plan.actual_alloc_frac < plan.target_alloc_frac
    assert plan.actual_alloc_frac == pytest.approx(plan.actual_max_loss / BUDGET)


def test_allocation_too_small_for_one_contract_is_blocked():
    quote = strangle_quote(debit=5000.0)
    plan = sizing.size_strategy(
        STRANGLE, target_alloc_frac=0.10, risk_budget_total=BUDGET, quote=quote
    )
    assert plan.contracts == 0
    assert plan.is_blocked
    assert "short of" in plan.blocked_reason


def test_zero_target_is_not_blocked():
    """Targeting zero is a decision, not a failure to size."""
    plan = sizing.size_strategy(
        BULL, target_alloc_frac=0.0, risk_budget_total=BUDGET, quote=spread_quote()
    )
    assert plan.contracts == 0
    assert not plan.is_blocked


# --- scale-down-only granularity ------------------------------------------


def test_granularity_floor_is_derived_from_the_threshold():
    """One contract must move the allocation by no more than the threshold."""
    assert sizing.min_contracts_for_granularity(0.10) == 2
    assert sizing.min_contracts_for_granularity(0.45) == 9
    assert sizing.min_contracts_for_granularity(0.0) == 0


def test_strangle_is_lifted_to_a_meaningful_contract_count():
    """PRD 2.3: partial reduction has to be expressible."""
    quote = strangle_quote(debit=1800.0)   # 0.10 x 20000 = 2000 -> 1 contract
    plan = sizing.size_strategy(
        STRANGLE, target_alloc_frac=0.10, risk_budget_total=BUDGET, quote=quote
    )
    assert plan.contracts == 2
    assert not plan.is_blocked


def test_strangle_blocked_when_granularity_will_not_fit_the_cap():
    quote = strangle_quote(debit=5000.0)
    plan = sizing.size_strategy(
        STRANGLE, target_alloc_frac=0.40, risk_budget_total=BUDGET, quote=quote
    )
    assert plan.contracts == 0
    assert "meaningful" in plan.blocked_reason


def test_spreads_get_no_granularity_floor():
    """They can be closed fully, so a one-lot is a legitimate position."""
    quote = spread_quote(width=3.0, credit=45.0)
    plan = sizing.size_strategy(
        BULL, target_alloc_frac=0.015, risk_budget_total=BUDGET, quote=quote
    )
    assert plan.contracts == 1


# --- buying power ----------------------------------------------------------


def test_buying_power_caps_an_increase():
    quote = spread_quote()
    plan = sizing.size_strategy(
        BULL,
        target_alloc_frac=0.40,
        risk_budget_total=BUDGET,
        quote=quote,
        buying_power=1000.0,
    )
    assert plan.contracts == 3          # floor(1000 / 255)
    assert plan.estimated_margin <= 1000.0


def test_buying_power_never_blocks_a_reduction():
    """Reducing risk must not be gated on affordability."""
    quote = spread_quote()
    plan = sizing.size_strategy(
        BULL,
        target_alloc_frac=0.10,
        risk_budget_total=BUDGET,
        quote=quote,
        contracts_current=40,
        buying_power=0.0,
    )
    assert plan.action == "reduce"
    assert plan.contract_delta < 0
    assert not plan.is_blocked


def test_no_buying_power_for_even_one_contract_is_blocked():
    plan = sizing.size_strategy(
        BULL,
        target_alloc_frac=0.40,
        risk_budget_total=BUDGET,
        quote=spread_quote(),
        buying_power=10.0,
    )
    assert plan.contracts == 0
    assert "buying power" in plan.blocked_reason


def test_budget_and_buying_power_are_separate_constraints():
    """Inside the risk budget, still unaffordable. Both must be checked."""
    quote = spread_quote()
    inside_budget = sizing.size_strategy(
        BULL, target_alloc_frac=0.40, risk_budget_total=BUDGET, quote=quote
    )
    assert inside_budget.contracts == 31
    constrained = sizing.size_strategy(
        BULL,
        target_alloc_frac=0.40,
        risk_budget_total=BUDGET,
        quote=quote,
        buying_power=2000.0,
    )
    assert constrained.contracts == 7


# --- actions ---------------------------------------------------------------


@pytest.mark.parametrize(
    "current,target,expected",
    [
        (0, 0.40, "open"),
        (10, 0.40, "increase"),
        (40, 0.10, "reduce"),
        (10, 0.0, "close"),
        (0, 0.0, "hold"),
    ],
)
def test_action_is_classified(current, target, expected):
    plan = sizing.size_strategy(
        BULL,
        target_alloc_frac=target,
        risk_budget_total=BUDGET,
        quote=spread_quote(),
        contracts_current=current,
    )
    assert plan.action == expected


def test_hold_when_the_count_does_not_change():
    plan = sizing.size_strategy(
        BULL,
        target_alloc_frac=0.40,
        risk_budget_total=BUDGET,
        quote=spread_quote(),
        contracts_current=31,
    )
    assert plan.action == "hold"
    assert plan.contract_delta == 0


# --- portfolio ordering ----------------------------------------------------


def test_reductions_are_sized_before_increases():
    """Trimming one leg releases the margin an increase elsewhere needs.

    Sizing increases first would block a rotation the account can afford.
    """
    quotes = {BULL: spread_quote(BULL), BEAR: spread_quote(BEAR)}
    plans = sizing.size_portfolio(
        {BULL: 0.05, BEAR: 0.40},
        risk_budget_total=BUDGET,
        quotes=quotes,
        contracts_current={BULL: 30, BEAR: 0},
        buying_power=500.0,
    )
    assert plans[BULL].action == "reduce"
    # Freed margin from trimming the bull put funds the bear call.
    assert plans[BEAR].contracts > 1
    assert not plans[BEAR].is_blocked


def test_missing_quote_for_a_funded_strategy_is_rejected():
    with pytest.raises(sizing.ChainRejected, match="no live quote"):
        sizing.size_portfolio(
            {BULL: 0.40},
            risk_budget_total=BUDGET,
            quotes={},
            buying_power=BUDGET,
        )


def test_missing_quote_for_a_zero_allocation_is_fine():
    plans = sizing.size_portfolio(
        {BULL: 0.0},
        risk_budget_total=BUDGET,
        quotes={},
        buying_power=BUDGET,
    )
    assert plans == {}


def test_plan_serialises_for_the_cycle_log():
    plan = sizing.size_strategy(
        BULL, target_alloc_frac=0.40, risk_budget_total=BUDGET, quote=spread_quote()
    )
    payload = plan.to_dict()
    assert payload["contracts"] == 31
    assert payload["action"] == "open"
    assert payload["blocked_reason"] is None
