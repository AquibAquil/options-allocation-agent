"""Allocator and Challenger tests (PRD 2.1).

All offline: a FakeClient returns canned JSON, so prompt construction, response
parsing, validation, and the challenge-resolution logic are all exercised
without a network call or a credential. The one thing not tested here is the
quality of the model's judgement -- that is not a unit-testable property.

The load-bearing cases:
  - no confidence score can enter the pipeline (PRD is explicit)
  - malformed model output raises ModelUnavailable, which upstream means "hold
    last valid allocation, do not trade" (PRD 2.5)
  - REJECT holds the current allocation rather than trading on an unsupported
    proposal
"""

from __future__ import annotations

import datetime as dt

import pytest

from alloc_agent import allocator as al
from alloc_agent import challenger as ch
from alloc_agent.evidence import packet as pk
from alloc_agent.llm import ModelResponse, ModelUnavailable
from alloc_agent.strategies import BULL_PUT_SPREAD, BEAR_CALL_SPREAD, KEYS, LONG_STRANGLE
from conftest import ASOF, synthetic_bars, synthetic_vxn

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key


class FakeClient:
    """Records the last call and returns a preloaded payload."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.last_system = None
        self.last_user = None
        self.last_schema = None

    def complete_json(self, *, system, user, schema, max_tokens=16000):
        self.last_system = system
        self.last_user = user
        self.last_schema = schema
        return ModelResponse(data=self.payload, raw_text="{}", model="fake-model", usage={})


class RaisingClient:
    def complete_json(self, *, system, user, schema, max_tokens=16000):
        raise ModelUnavailable("simulated model failure")


def a_packet():
    return pk.build_packet(
        cycle_id="2026-08-31T10:00-04:00",
        symbol="QQQ",
        bars=synthetic_bars(),
        vol_index_rows=synthetic_vxn(),
        positions={},
        account=pk.AccountSnapshot(equity=100_000.0, buying_power=200_000.0),
        correlation={"keys": list(KEYS), "matrix": [[1, -0.88, -0.45], [-0.88, 1, 0.01], [-0.45, 0.01, 1]]},
        asof=ASOF,
    )


def alloc_payload(bull=0.40, bear=0.10, strangle=0.15, extra=None):
    payload = {
        "allocations": {BULL: bull, BEAR: bear, STRANGLE: strangle},
        "reasoning": {
            BULL: "Trend intact, IV above RV, short strike 2 sigma away.",
            BEAR: "Weak upside momentum but thin premium; small share.",
            STRANGLE: "IV percentile low, protection cheap; held not cut.",
        },
        "portfolio_rationale": "Spreads strongly negatively correlated; budget not concentrated.",
    }
    if extra:
        payload.update(extra)
    return payload


def challenge_payload(verdict="APPROVE", bull=0.40, bear=0.10, strangle=0.15):
    return {
        "verdict": verdict,
        "critique": "Shares track the evidence; no performance chasing detected.",
        "evidence_cited": ["iv_rv.spread", "short_strike_distance_sigma"],
        "modified_allocations": {BULL: bull, BEAR: bear, STRANGLE: strangle},
    }


# --- allocator: prompt -----------------------------------------------------


def test_prompt_carries_theses_and_evidence():
    packet = a_packet()
    user = al.build_user_prompt(packet)
    # The theses travel in the packet, so the model sees facts and theses together.
    assert "not_invalidation" in user
    assert "bull_put_spread" in user
    assert "correlation" in user


def test_system_prompt_states_the_hard_rules():
    sys = al.SYSTEM_PROMPT
    assert "do not" in sys.lower() and "forecast" in sys.lower()
    assert "confidence" in sys.lower()
    assert "not_invalidation" in sys.lower()
    assert "maximum loss" in sys.lower()


def test_schema_has_no_confidence_field():
    props = al.ALLOCATION_SCHEMA["properties"]
    assert "confidence" not in props
    assert set(props["allocations"]["properties"]) == set(KEYS)


# --- allocator: parsing ----------------------------------------------------


def test_valid_proposal_parses():
    packet = a_packet()
    client = FakeClient(alloc_payload())
    proposal = al.Allocator(client).propose(packet)
    assert proposal.allocations == {BULL: 0.40, BEAR: 0.10, STRANGLE: 0.15}
    assert set(proposal.reasoning) == set(KEYS)
    assert proposal.portfolio_rationale


def test_proposal_uses_the_allocation_schema():
    client = FakeClient(alloc_payload())
    al.Allocator(client).propose(a_packet())
    assert client.last_schema is al.ALLOCATION_SCHEMA


def test_missing_strategy_key_is_rejected():
    bad = alloc_payload()
    del bad["allocations"][STRANGLE]
    with pytest.raises(ModelUnavailable, match="missing keys"):
        al.parse_allocation(bad)


def test_unknown_strategy_key_is_rejected():
    bad = alloc_payload()
    bad["allocations"]["iron_condor"] = 0.1
    with pytest.raises(ModelUnavailable, match="unknown keys"):
        al.parse_allocation(bad)


def test_allocation_out_of_range_is_rejected():
    with pytest.raises(ModelUnavailable, match="out of"):
        al.parse_allocation(alloc_payload(bull=1.5))


def test_non_numeric_allocation_is_rejected():
    bad = alloc_payload()
    bad["allocations"][BULL] = "lots"
    with pytest.raises(ModelUnavailable, match="not a number"):
        al.parse_allocation(bad)


def test_boolean_allocation_is_rejected():
    bad = alloc_payload()
    bad["allocations"][BULL] = True
    with pytest.raises(ModelUnavailable, match="not a number"):
        al.parse_allocation(bad)


def test_missing_reasoning_is_rejected():
    bad = alloc_payload()
    del bad["reasoning"][BEAR]
    with pytest.raises(ModelUnavailable, match="reasoning"):
        al.parse_allocation(bad)


def test_smuggled_confidence_field_is_rejected():
    """PRD: a confidence number must never enter the pipeline, under any name."""
    with pytest.raises(ModelUnavailable, match="confidence"):
        al.parse_allocation(alloc_payload(extra={"confidence": 0.9}))


def test_model_failure_becomes_hold_signal():
    """PRD 2.5: a failed model call is ModelUnavailable -> caller holds."""
    with pytest.raises(ModelUnavailable):
        al.Allocator(RaisingClient()).propose(a_packet())


def test_allocator_allows_holding_budget_back():
    """Summing to less than 1.0 is legitimate, not an error."""
    proposal = al.parse_allocation(alloc_payload(bull=0.20, bear=0.10, strangle=0.05))
    assert sum(proposal.allocations.values()) == pytest.approx(0.35)


# --- challenger: parsing ---------------------------------------------------


def test_approve_parses_and_flags_approved():
    result = ch.parse_challenge(challenge_payload("APPROVE"))
    assert result.approved
    assert not result.modified and not result.rejected
    assert result.evidence_cited


def test_modify_carries_corrected_allocations():
    result = ch.parse_challenge(challenge_payload("MODIFY", bull=0.35))
    assert result.modified
    assert result.modified_allocations[BULL] == 0.35


def test_reject_parses():
    assert ch.parse_challenge(challenge_payload("REJECT")).rejected


def test_unknown_verdict_is_rejected():
    bad = challenge_payload()
    bad["verdict"] = "MAYBE"
    with pytest.raises(ModelUnavailable, match="verdict"):
        ch.parse_challenge(bad)


def test_challenge_without_cited_evidence_is_rejected():
    bad = challenge_payload()
    bad["evidence_cited"] = []
    with pytest.raises(ModelUnavailable, match="cited evidence"):
        ch.parse_challenge(bad)


def test_challenge_confidence_field_is_rejected():
    bad = challenge_payload()
    bad["confidence"] = 0.8
    with pytest.raises(ModelUnavailable, match="confidence"):
        ch.parse_challenge(bad)


def test_challenger_system_prompt_names_its_bias_and_blind_spot():
    sys = ch.SYSTEM_PROMPT.lower()
    assert "performance chasing" in sys
    assert "not_invalidation" in sys        # the blind-spot guard
    assert "tend to agree" in sys           # its own agreement bias


def test_challenger_review_end_to_end():
    packet = a_packet()
    proposal = al.Allocator(FakeClient(alloc_payload())).propose(packet)
    result = ch.Challenger(FakeClient(challenge_payload("APPROVE"))).review(packet, proposal)
    assert result.approved


# --- challenge resolution --------------------------------------------------


def test_approve_uses_the_proposal():
    proposal = al.parse_allocation(alloc_payload(0.40, 0.10, 0.15))
    result = ch.parse_challenge(challenge_payload("APPROVE", 0.40, 0.10, 0.15))
    effective, source = ch.resolve_effective_allocation(proposal, result, {BULL: 0, BEAR: 0, STRANGLE: 0})
    assert effective == {BULL: 0.40, BEAR: 0.10, STRANGLE: 0.15}
    assert source == "allocator_proposal"


def test_modify_uses_the_challengers_numbers():
    proposal = al.parse_allocation(alloc_payload(0.45, 0.10, 0.15))
    result = ch.parse_challenge(challenge_payload("MODIFY", 0.35, 0.10, 0.15))
    effective, source = ch.resolve_effective_allocation(proposal, result, {BULL: 0.30, BEAR: 0.10, STRANGLE: 0.15})
    assert effective[BULL] == 0.35
    assert source == "challenger_modified"


def test_reject_holds_the_current_allocation():
    """PRD: on REJECT, do not trade on an unsupported proposal -- hold."""
    proposal = al.parse_allocation(alloc_payload(0.45, 0.10, 0.15))
    result = ch.parse_challenge(challenge_payload("REJECT"))
    current = {BULL: 0.30, BEAR: 0.10, STRANGLE: 0.20}
    effective, source = ch.resolve_effective_allocation(proposal, result, current)
    assert effective == current
    assert source == "held_on_reject"


# --- integration with the risk gates --------------------------------------


def test_resolved_allocation_feeds_the_gates():
    """The whole point: allocator -> challenger -> gates, end to end offline."""
    from alloc_agent import risk

    packet = a_packet()
    proposal = al.Allocator(FakeClient(alloc_payload(0.80, 0.10, 0.05))).propose(packet)
    # Allocator over-allocated bull to 0.80; challenger approves as-is.
    result = ch.Challenger(FakeClient(challenge_payload("APPROVE", 0.80, 0.10, 0.05))).review(packet, proposal)
    effective, _ = ch.resolve_effective_allocation(proposal, result, {BULL: 0, BEAR: 0, STRANGLE: 0})

    gated = risk.apply_gates(effective, {BULL: 0, BEAR: 0, STRANGLE: 0})
    # The 0.80 is capped to the 0.45 per-strategy limit, and the sub-threshold
    # strangle (0.05) snaps to zero -- the gates catch what the models let by.
    assert gated.final[BULL] == pytest.approx(risk.RISK.per_strategy_max)
    assert gated.final[STRANGLE] == 0.0
