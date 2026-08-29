"""Allocation delta and rejection rate tests (PRD 2.8)."""

from __future__ import annotations

import pytest

from alloc_agent.benchmark import AllocationDelta, CycleReturn, RejectionTracker
from alloc_agent.strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, KEYS, LONG_STRANGLE

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key


def ret(bull=0.0, bear=0.0, strangle=0.0):
    return {BULL: bull, BEAR: bear, STRANGLE: strangle}


def wt(bull=0.0, bear=0.0, strangle=0.0):
    return {BULL: bull, BEAR: bear, STRANGLE: strangle}


# --- allocation delta ------------------------------------------------------


def test_equal_weights_have_zero_delta():
    """If the AI holds equal weights, it cannot beat equal weight."""
    d = AllocationDelta()
    third = 1.0 / 3
    d.record(CycleReturn("c1", ret(0.05, -0.02, 0.01), wt(third, third, third)))
    assert d.cumulative_delta == pytest.approx(0.0, abs=1e-12)


def test_overweighting_a_winner_beats_equal_weight():
    d = AllocationDelta()
    # Bull put is the only winner; AI is all-in on it, equal weight is not.
    d.record(CycleReturn("c1", ret(0.06, -0.03, -0.03), wt(bull=1.0)))
    assert d.actual_total_return == pytest.approx(0.06)
    assert d.equal_weight_total_return == pytest.approx(0.0)
    assert d.cumulative_delta > 0


def test_overweighting_a_loser_loses_to_equal_weight():
    d = AllocationDelta()
    d.record(CycleReturn("c1", ret(-0.06, 0.03, 0.03), wt(bull=1.0)))
    assert d.cumulative_delta < 0
    assert not d.summary()["beats_equal_weight"]


def test_returns_compound_not_sum():
    """+2% then -2% is not flat; the curve must compound."""
    d = AllocationDelta()
    d.record(CycleReturn("c1", ret(bull=0.02), wt(bull=1.0)))
    d.record(CycleReturn("c2", ret(bull=-0.02), wt(bull=1.0)))
    assert d.actual_total_return == pytest.approx(1.02 * 0.98 - 1.0)
    assert d.actual_total_return < 0  # a genuine loss, not zero


def test_delta_is_reported_regardless_of_sign():
    d = AllocationDelta()
    d.record(CycleReturn("c1", ret(-0.06, 0.03, 0.03), wt(bull=1.0)))
    summary = d.summary()
    assert "allocation_delta" in summary
    assert summary["allocation_delta"] < 0  # honest reporting of a loss


def test_history_accumulates_per_cycle():
    d = AllocationDelta()
    d.record(CycleReturn("c1", ret(bull=0.01), wt(bull=0.5)))
    d.record(CycleReturn("c2", ret(bull=0.01), wt(bull=0.5)))
    assert len(d.history) == 2
    assert d.history[-1]["cumulative_delta"] == d.cumulative_delta


# --- rejection tracker -----------------------------------------------------


def test_rejection_rate_counts_modify_and_reject():
    t = RejectionTracker()
    for v in ("APPROVE", "APPROVE", "MODIFY", "REJECT"):
        t.record(v)
    assert t.total == 4
    assert t.rejection_rate == pytest.approx(0.5)   # 2 of 4 not rubber-stamped
    assert t.approval_rate == pytest.approx(0.5)


def test_all_approvals_is_visible_as_high_approval():
    """PRD: if the challenger approves nearly everything, say so."""
    t = RejectionTracker()
    for _ in range(10):
        t.record("APPROVE")
    assert t.rejection_rate == 0.0
    assert t.summary()["approval_rate"] == 1.0


def test_empty_tracker_has_zero_rate():
    assert RejectionTracker().rejection_rate == 0.0


def test_unknown_verdict_is_rejected():
    with pytest.raises(ValueError, match="unknown verdict"):
        RejectionTracker().record("MAYBE")
