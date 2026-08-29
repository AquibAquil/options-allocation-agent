"""Shock simulation tests (PRD 4.4).

The model's actual decision is not unit-testable (and deliberately not asserted).
What IS tested: every scenario builds a valid packet that genuinely represents
its regime, the packets carry no verdicts, and run_scenario drives the real
decision path (with a fake allocator) and records the result -- including
holding cleanly when a model call fails.
"""

from __future__ import annotations

import json

import pytest

from alloc_agent import shocks
from alloc_agent.llm import ModelResponse, ModelUnavailable
from alloc_agent.strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, KEYS, LONG_STRANGLE

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, *, system, user, schema, max_tokens=16000):
        return ModelResponse(data=self.payload, raw_text="{}", model="fake", usage={})


class RaisingClient:
    def complete_json(self, *, system, user, schema, max_tokens=16000):
        raise ModelUnavailable("simulated failure")


def alloc_payload(bull=0.35, bear=0.35, strangle=0.20):
    return {
        "allocations": {BULL: bull, BEAR: bear, STRANGLE: strangle},
        "reasoning": {k: "reason" for k in KEYS},
        "portfolio_rationale": "rationale",
    }


def challenge_payload(verdict="APPROVE"):
    return {
        "verdict": verdict,
        "critique": "c",
        "evidence_cited": ["x"],
        "modified_allocations": {BULL: 0.35, BEAR: 0.35, STRANGLE: 0.20},
    }


# --- scenarios build valid, representative packets -------------------------


def test_all_scenarios_build():
    scenarios = shocks.all_scenarios()
    assert len(scenarios) == 6
    names = {s.name for s in scenarios}
    assert "strangle_bleeds_protection_cheap" in names
    assert "strangle_protection_expensive" in names


def test_every_packet_is_labelled_simulated():
    for s in shocks.all_scenarios():
        assert any("SIMULATED" in n for n in s.packet.notes)
        assert "simulated" in s.packet.market.implied_vol_source.lower()


def test_baseline_is_quiet_and_cheap():
    p = shocks.baseline_quiet().packet
    assert p.market.implied_vol_percentile_252d == 20.0
    assert p.market.iv_rv["spread"] > 0          # IV above RV: premium exists


def test_vol_shock_has_high_iv_and_rv_above_iv():
    p = shocks.volatility_shock().packet
    assert p.market.implied_vol_percentile_252d >= 90.0
    assert p.market.iv_rv["realized"] > p.market.iv_rv["implied"]   # premium gone


def test_selloff_breaches_the_bull_put_short_strike():
    p = shocks.directional_selloff_breach().packet
    bull = next(s for s in p.strategies if s.key == BULL)
    assert abs(bull.short_strike_distance_sigma) < 0.25    # breached
    assert bull.pnl_frac_of_max_loss < 0                   # underwater


def test_meltup_breaches_the_bear_call_short_strike():
    p = shocks.melt_up_breach().packet
    bear = next(s for s in p.strategies if s.key == BEAR)
    assert abs(bear.short_strike_distance_sigma) < 0.25


def test_bleeding_case_is_cheap_and_losing():
    """The marquee contrast: strangle deep in loss, but IV cheap."""
    p = shocks.strangle_bleeds_but_protection_cheap().packet
    strangle = next(s for s in p.strategies if s.key == STRANGLE)
    assert strangle.pnl_frac_of_max_loss < -0.3     # bleeding
    assert p.market.implied_vol_percentile_252d < 25.0   # protection cheap


def test_expensive_case_is_costly_and_winning():
    """The counter-case: strangle in profit, but IV expensive."""
    p = shocks.strangle_protection_expensive().packet
    strangle = next(s for s in p.strategies if s.key == STRANGLE)
    assert strangle.pnl_frac_of_max_loss > 0           # not losing
    assert p.market.implied_vol_percentile_252d > 80.0  # protection expensive


def test_packets_carry_no_verdicts():
    """Same discipline as the live packet: evidence, not conclusions."""
    banned = ("holds", "should", "recommend", "verdict", "signal", "score")
    thesis_fields = {"invalidation", "not_invalidation"}
    for s in shocks.all_scenarios():
        payload = json.loads(s.packet.to_json())

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k not in thesis_fields:
                        assert not any(b in k.lower() for b in banned), k
                    if isinstance(v, bool):
                        assert k == "bars_are_complete_sessions"
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)


# --- run_scenario drives the real decision path ----------------------------


def test_run_scenario_records_the_full_decision():
    from alloc_agent.allocator import Allocator
    from alloc_agent.challenger import Challenger

    scenario = shocks.baseline_quiet()
    record = shocks.run_scenario(
        scenario,
        Allocator(FakeClient(alloc_payload())),
        Challenger(FakeClient(challenge_payload("APPROVE"))),
    )
    assert record["simulated"] is True
    assert record["proposal"]["allocations"][BULL] == 0.35
    assert record["challenge"]["verdict"] == "APPROVE"
    assert set(record["final_allocation"]) == set(KEYS)


def test_run_scenario_applies_gates():
    from alloc_agent.allocator import Allocator
    from alloc_agent.challenger import Challenger

    scenario = shocks.baseline_quiet()
    # Allocator over-caps bull; gate must trim to the per-strategy max.
    record = shocks.run_scenario(
        scenario,
        Allocator(FakeClient(alloc_payload(bull=0.90, bear=0.05, strangle=0.05))),
        Challenger(FakeClient(challenge_payload("APPROVE"))),
    )
    from alloc_agent.config import RISK
    assert record["final_allocation"][BULL] == pytest.approx(RISK.per_strategy_max)


def test_run_scenario_holds_on_model_failure():
    from alloc_agent.allocator import Allocator
    from alloc_agent.challenger import Challenger

    record = shocks.run_scenario(
        shocks.baseline_quiet(),
        Allocator(RaisingClient()),
        Challenger(FakeClient(challenge_payload())),
    )
    assert "error" in record
    assert "final_allocation" not in record
