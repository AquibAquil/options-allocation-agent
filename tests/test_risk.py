"""Hard risk gate tests (PRD 2.2).

These gates are the last thing standing between a model's output and a real
order, so the cases that matter most are the adversarial ones: a proposal that
is over the cap, over budget, malformed, or shaped to slip a violation through
the adjustment threshold.
"""

from __future__ import annotations

import pytest

from alloc_agent import risk
from alloc_agent.config import RISK
from alloc_agent.strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, KEYS, LONG_STRANGLE

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key

FLAT = {BULL: 0.0, BEAR: 0.0, STRANGLE: 0.0}


def alloc(bull=0.0, bear=0.0, strangle=0.0):
    return {BULL: bull, BEAR: bear, STRANGLE: strangle}


def rules(result):
    return {a.rule for a in result.adjustments}


# --- malformed output (PRD 2.5: hold, do not trade) ------------------------


def test_missing_strategy_is_rejected():
    with pytest.raises(risk.MalformedAllocation, match="missing strategy keys"):
        risk.apply_gates({BULL: 0.4, BEAR: 0.3}, FLAT)


def test_unknown_strategy_is_rejected():
    with pytest.raises(risk.MalformedAllocation, match="unknown strategy keys"):
        risk.apply_gates({**alloc(0.3, 0.3, 0.3), "iron_condor": 0.1}, FLAT)


def test_negative_allocation_is_rejected():
    with pytest.raises(risk.MalformedAllocation, match="negative"):
        risk.apply_gates(alloc(-0.1, 0.3, 0.3), FLAT)


def test_non_finite_allocation_is_rejected():
    with pytest.raises(risk.MalformedAllocation, match="not finite"):
        risk.apply_gates(alloc(float("nan"), 0.3, 0.3), FLAT)
    with pytest.raises(risk.MalformedAllocation, match="not finite"):
        risk.apply_gates(alloc(float("inf"), 0.3, 0.3), FLAT)


def test_non_numeric_allocation_is_rejected():
    with pytest.raises(risk.MalformedAllocation, match="must be a number"):
        risk.apply_gates({BULL: "0.4", BEAR: 0.3, STRANGLE: 0.3}, FLAT)


def test_boolean_is_not_accepted_as_a_number():
    """True would silently become 1.0, a 100% allocation."""
    with pytest.raises(risk.MalformedAllocation, match="must be a number"):
        risk.apply_gates({BULL: True, BEAR: 0.3, STRANGLE: 0.3}, FLAT)


# --- per-strategy cap ------------------------------------------------------


def test_over_cap_is_capped():
    result = risk.apply_gates(alloc(0.80, 0.10, 0.10), FLAT)
    assert result.final[BULL] == pytest.approx(RISK.per_strategy_max)
    assert "per_strategy_cap" in rules(result)


def test_at_cap_is_untouched():
    result = risk.apply_gates(alloc(RISK.per_strategy_max, 0.30, 0.20), FLAT)
    assert result.final[BULL] == pytest.approx(RISK.per_strategy_max)
    assert "per_strategy_cap" not in rules(result)


def test_cap_holds_even_when_every_strategy_is_over():
    result = risk.apply_gates(alloc(0.9, 0.9, 0.9), FLAT)
    for key in KEYS:
        assert result.final[key] <= RISK.per_strategy_max + 1e-12


# --- snap to zero ----------------------------------------------------------


def test_sub_threshold_snaps_to_zero():
    result = risk.apply_gates(alloc(0.40, 0.40, 0.05), FLAT)
    assert result.final[STRANGLE] == 0.0
    assert "snap_to_zero" in rules(result)


def test_at_threshold_survives():
    result = risk.apply_gates(alloc(0.40, 0.40, RISK.snap_to_zero_below), FLAT)
    assert result.final[STRANGLE] == pytest.approx(RISK.snap_to_zero_below)


def test_zero_is_not_an_adjustment():
    """Proposing zero is a decision, not a violation to be logged."""
    result = risk.apply_gates(alloc(0.40, 0.40, 0.0), FLAT)
    assert result.final[STRANGLE] == 0.0
    assert "snap_to_zero" not in rules(result)


# --- the 2.2 / 2.3 conflict ------------------------------------------------


def test_scale_down_only_position_floors_instead_of_closing():
    """PRD 2.3 over 2.2: a live strangle cannot be closed by drift."""
    result = risk.apply_gates(alloc(0.40, 0.40, 0.04), alloc(strangle=0.20))
    assert result.final[STRANGLE] == pytest.approx(RISK.snap_to_zero_below)
    assert "scale_down_only_floor" in rules(result)


def test_scale_down_only_still_reaches_zero_on_an_explicit_target():
    """An explicit cut on invalidation must still close it."""
    result = risk.apply_gates(alloc(0.40, 0.40, 0.0), alloc(strangle=0.20))
    assert result.final[STRANGLE] == 0.0
    assert "scale_down_only_floor" not in rules(result)


def test_scale_down_only_floor_does_not_apply_when_flat():
    """Nothing to protect: a flat strangle proposed sub-threshold stays flat."""
    result = risk.apply_gates(alloc(0.40, 0.40, 0.04), FLAT)
    assert result.final[STRANGLE] == 0.0


def test_spreads_are_not_floored():
    """Only scale-down-only strategies get the floor."""
    result = risk.apply_gates(alloc(0.04, 0.40, 0.40), alloc(bull=0.20))
    assert result.final[BULL] == 0.0
    assert "scale_down_only_floor" not in rules(result)


def test_literal_2_2_can_be_restored(monkeypatch):
    monkeypatch.setattr(risk, "SNAP_TO_ZERO_OVERRIDES_SCALE_DOWN", True)
    result = risk.apply_gates(alloc(0.40, 0.40, 0.04), alloc(strangle=0.20))
    assert result.final[STRANGLE] == 0.0


# --- total budget ----------------------------------------------------------


def test_over_budget_is_scaled_proportionally():
    result = risk.apply_gates(alloc(0.45, 0.45, 0.45), FLAT)
    assert sum(result.final.values()) <= 1.0 + 1e-12
    assert "total_budget" in rules(result)
    # Proportional: equal inputs stay equal.
    assert result.final[BULL] == pytest.approx(result.final[BEAR])


def test_exactly_full_budget_is_not_scaled():
    result = risk.apply_gates(alloc(0.40, 0.40, 0.20), FLAT)
    assert "total_budget" not in rules(result)
    assert sum(result.final.values()) == pytest.approx(1.0)


def test_under_budget_is_left_alone():
    """Holding back is a legitimate decision, not a shortfall to top up."""
    result = risk.apply_gates(alloc(0.20, 0.20, 0.10), FLAT)
    assert sum(result.final.values()) == pytest.approx(0.50)
    assert not result.adjustments


def test_rescaling_never_pushes_a_survivor_under_the_floor():
    """The step-5 re-snap is unreachable under the current bounds.

    After step 3 every surviving allocation is at least `floor`. For scaling to
    push one under, it needs s/(s + others) < floor, i.e. s < others/9 at a 10%
    floor; with a 45% cap and two other strategies, others <= 0.90 forces
    s < 0.10, which step 3 already snapped. Asserted rather than assumed,
    because the cap and the strategy count are both configurable.
    """
    floor, cap = RISK.snap_to_zero_below, RISK.per_strategy_max
    others_max = cap * (len(KEYS) - 1)
    assert floor >= others_max / (1.0 / floor - 1.0), (
        "bounds now allow scaling to push an allocation under the floor; "
        "the step-5 re-snap is live and needs direct test coverage"
    )

    for proposal in (alloc(0.45, 0.45, 0.105), alloc(0.45, 0.45, 0.11), alloc(0.44, 0.44, 0.13)):
        result = risk.apply_gates(proposal, FLAT)
        assert sum(result.final.values()) <= 1.0 + 1e-12
        assert "snap_to_zero_after_scaling" not in rules(result)
        for value in result.final.values():
            assert value == 0.0 or value >= floor - 1e-12


# --- adjustment threshold --------------------------------------------------


def test_small_change_does_not_trade():
    current = alloc(0.30, 0.30, 0.20)
    result = risk.apply_gates(alloc(0.33, 0.30, 0.20), current)
    assert result.final[BULL] == pytest.approx(0.30)
    assert BULL not in result.traded_keys
    assert "adjustment_threshold" in rules(result)


def test_change_at_the_threshold_trades():
    """0.35 - 0.30 is 0.04999999999999999 in binary floating point.

    Without a tolerance the boundary case silently does not trade, which is the
    opposite of what PRD 2.2 says: smaller changes do not trade, so a change of
    exactly the threshold must.
    """
    current = alloc(0.30, 0.30, 0.20)
    result = risk.apply_gates(alloc(0.30 + RISK.adjustment_threshold, 0.30, 0.20), current)
    assert BULL in result.traded_keys
    assert result.final[BULL] == pytest.approx(0.35)


def test_change_just_under_the_threshold_does_not_trade():
    current = alloc(0.30, 0.30, 0.20)
    result = risk.apply_gates(
        alloc(0.30 + RISK.adjustment_threshold - 0.001, 0.30, 0.20), current
    )
    assert BULL not in result.traded_keys


def test_forced_reduction_bypasses_the_threshold():
    """PRD 2.2: hard risk reductions ignore the adjustment threshold."""
    current = alloc(0.30, 0.30, 0.20)
    result = risk.apply_gates(
        alloc(0.28, 0.30, 0.20), current, forced_reductions=frozenset({BULL})
    )
    assert BULL in result.traded_keys
    assert result.final[BULL] == pytest.approx(0.28)


def test_forced_flag_does_not_licence_a_small_increase():
    """The bypass is for reductions only."""
    current = alloc(0.30, 0.30, 0.20)
    result = risk.apply_gates(
        alloc(0.32, 0.30, 0.20), current, forced_reductions=frozenset({BULL})
    )
    assert BULL not in result.traded_keys
    assert result.final[BULL] == pytest.approx(0.30)


def test_threshold_cannot_preserve_a_cap_violation():
    """A tiny change that is still over the cap must not survive as 'too small'.

    This is the gate-ordering case: comparing against current before applying
    the cap would let an illegal allocation persist indefinitely.
    """
    current = alloc(bull=0.60)  # already above the cap
    result = risk.apply_gates(alloc(0.62, 0.20, 0.10), current)
    assert result.final[BULL] == pytest.approx(RISK.per_strategy_max)
    assert BULL in result.traded_keys


# --- result bookkeeping ----------------------------------------------------


def test_clean_proposal_passes_through_untouched():
    result = risk.apply_gates(alloc(0.40, 0.35, 0.15), FLAT)
    assert result.final == pytest.approx(alloc(0.40, 0.35, 0.15))
    assert not result.modified
    assert set(result.traded_keys) == {BULL, BEAR, STRANGLE}


def test_result_records_what_was_proposed():
    proposal = alloc(0.80, 0.10, 0.10)
    result = risk.apply_gates(proposal, FLAT)
    assert result.proposed == pytest.approx(proposal)
    assert result.final[BULL] != proposal[BULL]


def test_result_serialises_for_the_cycle_log():
    result = risk.apply_gates(alloc(0.80, 0.10, 0.10), FLAT)
    payload = result.to_dict()
    assert payload["proposed"][BULL] == 0.80
    assert payload["adjustments"][0]["rule"] == "per_strategy_cap"
    assert payload["adjustments"][0]["detail"]


def test_every_adjustment_names_a_rule_and_a_reason():
    result = risk.apply_gates(alloc(0.9, 0.9, 0.02), FLAT)
    assert result.adjustments
    for adjustment in result.adjustments:
        assert adjustment.rule
        assert adjustment.detail
        assert adjustment.strategy_key in KEYS


def test_final_allocation_always_satisfies_every_hard_limit():
    """Sweep: whatever goes in, the output is inside the box."""
    proposals = [
        alloc(0.9, 0.9, 0.9),
        alloc(1.0, 0.0, 0.0),
        alloc(0.05, 0.05, 0.05),
        alloc(0.44, 0.44, 0.44),
        alloc(0.0, 0.0, 0.0),
        alloc(0.5, 0.3, 0.11),
    ]
    for proposal in proposals:
        for current in (FLAT, alloc(0.2, 0.2, 0.2), alloc(0.45, 0.45, 0.10)):
            final = risk.apply_gates(proposal, current).final
            assert sum(final.values()) <= 1.0 + 1e-9, (proposal, current)
            for key, value in final.items():
                assert 0.0 <= value <= RISK.per_strategy_max + 1e-9, (key, proposal)
