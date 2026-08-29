"""Shock simulations (PRD 4.4, mitigation for PRD 3).

The live window is late-August quiet: the allocator's correct decision may be
obvious and unchanging, so the four days may never exercise the judgement the
system exists for. These simulations feed the REAL allocator, challenger, and
risk gates a set of synthetic evidence packets crafted to represent regimes the
window will not produce, and record what the AI does.

This is explicitly allowed (PRD 4.4, confirmed by mentors) as a deliverable, not
a contingency. It is SIMULATION, never scored P&L: paper equity remains the
only P&L record. Every artifact this writes is labelled a simulation.

The scenarios are chosen around the judgements that matter, above all the
strangle's: hold it while it bleeds if protection is still cheap, but cut it
when protection has become expensive. A rule cannot tell those apart; that is
the whole wedge.

Each scenario carries an `expectation` in plain words -- what good judgement
looks like. It is reported next to the actual decision as a narrative aid, NOT
asserted as a pass/fail: model judgement is not deterministic, and the honest
deliverable shows what the allocator did and lets the reader judge.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .evidence import metrics
from .evidence.packet import (
    Constraints,
    EvidencePacket,
    MarketEvidence,
    PortfolioEvidence,
    StrategyEvidence,
)
from .strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, BY_KEY, KEYS, LONG_STRANGLE

ASOF = dt.date(2026, 8, 31)
EQUITY = 100_000.0
RISK_BUDGET = 20_000.0  # 20% of equity, as max loss

# A representative correlation matrix (the real revaluation shape).
CORRELATION = {
    "keys": list(KEYS),
    "matrix": [[1.0, -0.88, -0.45], [-0.88, 1.0, 0.01], [-0.45, 0.01, 1.0]],
}


@dataclass(frozen=True)
class StratState:
    """High-level per-strategy state for a scenario."""

    key: str
    alloc_frac: float = 0.0
    contracts: int = 0
    pnl_frac: float | None = None          # unrealised P&L as a fraction of max loss
    short_sigma: float | None = None       # nearest short strike distance, in sigma
    min_dte: int | None = None
    drawdown: float | None = None
    days_held: int | None = None


def _strategy_evidence(state: StratState) -> StrategyEvidence:
    s = BY_KEY[state.key]
    max_loss = state.alloc_frac * RISK_BUDGET
    pnl = None if state.pnl_frac is None else state.pnl_frac * max_loss
    return StrategyEvidence(
        key=s.key,
        name=s.name,
        thesis=s.thesis,
        invalidation=s.invalidation,
        not_invalidation=s.not_invalidation,
        exit_behaviour=s.exit_behaviour.value,
        vol_exposure=s.vol_exposure.value,
        direction=s.direction,
        allocation_frac=state.alloc_frac,
        max_loss_outstanding=max_loss,
        contracts=state.contracts,
        unrealized_pnl=pnl,
        pnl_frac_of_max_loss=state.pnl_frac,
        drawdown_frac=state.drawdown,
        days_held=state.days_held,
        legs=(),
        short_strike_distance_sigma=state.short_sigma,
        min_dte=state.min_dte,
    )


@dataclass(frozen=True)
class Scenario:
    name: str
    regime: str
    description: str
    expectation: str
    packet: EvidencePacket
    current_allocation: dict[str, float]


def _packet(
    *,
    cycle_id: str,
    spot: float,
    returns: dict,
    rv: float,
    iv: float,
    iv_pct: float,
    states: list[StratState],
) -> EvidencePacket:
    realized = {"5d": rv, "10d": rv, "21d": rv, "63d": rv}
    market = MarketEvidence(
        symbol="QQQ",
        spot=spot,
        spot_asof=ASOF.isoformat(),
        returns=returns,
        realized_vol=realized,
        parkinson_vol_21d=rv,
        implied_vol=iv,
        implied_vol_source="CBOE VXN (simulated)",
        implied_vol_asof=ASOF.isoformat(),
        implied_vol_percentile_252d=iv_pct,
        implied_vol_percentile_full=iv_pct,
        iv_rv=metrics.vol_risk_premium(iv, rv),
    )
    strategies = tuple(_strategy_evidence(s) for s in states)
    consumed = sum(s.max_loss_outstanding for s in strategies)
    portfolio = PortfolioEvidence(
        equity=EQUITY,
        buying_power=EQUITY * 2,
        options_buying_power=EQUITY,
        risk_budget_total=RISK_BUDGET,
        risk_budget_consumed=consumed,
        risk_budget_available=RISK_BUDGET - consumed,
        risk_budget_utilisation=consumed / RISK_BUDGET,
        total_max_loss_outstanding=consumed,
        max_loss_as_frac_of_equity=consumed / EQUITY,
    )
    return EvidencePacket(
        asof=ASOF.isoformat(),
        cycle_id=cycle_id,
        market=market,
        strategies=strategies,
        portfolio=portfolio,
        correlation=CORRELATION,
        constraints=Constraints.current(),
        notes=("SIMULATED regime, not live market data",),
    )


def _current(states: list[StratState]) -> dict[str, float]:
    held = {s.key: s.alloc_frac for s in states}
    return {k: held.get(k, 0.0) for k in KEYS}


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key


def baseline_quiet() -> Scenario:
    states = [
        StratState(BULL, 0.35, 20, pnl_frac=0.05, short_sigma=-1.9, min_dte=9),
        StratState(BEAR, 0.35, 20, pnl_frac=0.03, short_sigma=1.9, min_dte=9),
        StratState(STRANGLE, 0.20, 6, pnl_frac=-0.06, short_sigma=None, min_dte=28),
    ]
    return Scenario(
        name="baseline_quiet",
        regime="Quiet, low IV (the live window)",
        description=(
            "Low realised vol, IV in the 20th percentile, both spreads' short "
            "strikes ~1.9 sigma away and mildly profitable, strangle bleeding "
            "gently. This is roughly what the four-day live window looks like."
        ),
        expectation=(
            "Fund both spreads to harvest the IV>RV premium; keep the strangle "
            "funded despite its small loss because protection is cheap (IV 20th "
            "pct) -- do NOT cut it for bleeding."
        ),
        packet=_packet(
            cycle_id="shock-baseline_quiet",
            spot=716.0,
            returns={"1d": 0.001, "5d": 0.004, "10d": 0.006, "21d": 0.012, "63d": 0.03},
            rv=0.145,
            iv=0.175,
            iv_pct=20.0,
            states=states,
        ),
        current_allocation=_current(states),
    )


def volatility_shock() -> Scenario:
    states = [
        StratState(BULL, 0.35, 20, pnl_frac=-0.55, short_sigma=-1.1, min_dte=7, drawdown=0.6),
        StratState(BEAR, 0.35, 20, pnl_frac=-0.40, short_sigma=1.2, min_dte=7, drawdown=0.45),
        StratState(STRANGLE, 0.20, 6, pnl_frac=0.9, short_sigma=None, min_dte=26),
    ]
    return Scenario(
        name="volatility_shock",
        regime="Volatility expansion",
        description=(
            "IV spikes to the 92nd percentile, realised vol jumps above implied, "
            "both spreads' short strikes are now ~1.1-1.2 sigma away and deep in "
            "the red, the long strangle has paid off sharply."
        ),
        expectation=(
            "Cut both short-vol spreads hard -- their short strikes are "
            "threatened and RV now exceeds IV, so the premium no longer pays for "
            "the risk. The strangle is the winner, but IV at the 92nd pct means "
            "adding fresh convexity is expensive; holding the existing position "
            "is reasonable, chasing it less so."
        ),
        packet=_packet(
            cycle_id="shock-volatility_shock",
            spot=690.0,
            returns={"1d": -0.03, "5d": -0.05, "10d": -0.04, "21d": -0.02, "63d": 0.01},
            rv=0.34,
            iv=0.30,
            iv_pct=92.0,
            states=states,
        ),
        current_allocation=_current(states),
    )


def directional_selloff_breach() -> Scenario:
    states = [
        StratState(BULL, 0.40, 24, pnl_frac=-0.85, short_sigma=0.1, min_dte=6, drawdown=0.9),
        StratState(BEAR, 0.30, 17, pnl_frac=0.4, short_sigma=2.6, min_dte=6),
        StratState(STRANGLE, 0.20, 6, pnl_frac=0.7, short_sigma=None, min_dte=25),
    ]
    return Scenario(
        name="directional_selloff_breach",
        regime="Sharp selloff, bull put breached",
        description=(
            "A hard down move puts spot essentially at the bull put's short "
            "strike (0.1 sigma away) with the position deeply underwater. The "
            "bear call is now far out of the money and profitable; the strangle "
            "has gained on the move."
        ),
        expectation=(
            "Cut the bull put decisively -- its short strike is breached, which "
            "is genuine invalidation, not noise. This is the case a defined-risk "
            "spread caps but does not welcome. The bear call is safe; the "
            "strangle earned its keep."
        ),
        packet=_packet(
            cycle_id="shock-directional_selloff_breach",
            spot=678.0,
            returns={"1d": -0.045, "5d": -0.06, "10d": -0.05, "21d": -0.03, "63d": -0.01},
            rv=0.28,
            iv=0.27,
            iv_pct=78.0,
            states=states,
        ),
        current_allocation=_current(states),
    )


def melt_up_breach() -> Scenario:
    states = [
        StratState(BULL, 0.30, 17, pnl_frac=0.4, short_sigma=-2.6, min_dte=6),
        StratState(BEAR, 0.40, 24, pnl_frac=-0.85, short_sigma=0.1, min_dte=6, drawdown=0.9),
        StratState(STRANGLE, 0.20, 6, pnl_frac=0.6, short_sigma=None, min_dte=25),
    ]
    return Scenario(
        name="melt_up_breach",
        regime="Sharp rally, bear call breached",
        description=(
            "The mirror of the selloff: a sharp rally puts spot at the bear "
            "call's short strike, that position deep in the red, while the bull "
            "put is now safe and the strangle has gained."
        ),
        expectation=(
            "Cut the bear call -- its short strike is breached. The bull put is "
            "safe and profitable; the strangle profited from the move."
        ),
        packet=_packet(
            cycle_id="shock-melt_up_breach",
            spot=752.0,
            returns={"1d": 0.045, "5d": 0.06, "10d": 0.05, "21d": 0.035, "63d": 0.02},
            rv=0.24,
            iv=0.24,
            iv_pct=70.0,
            states=states,
        ),
        current_allocation=_current(states),
    )


def strangle_bleeds_but_protection_cheap() -> Scenario:
    """The marquee case: bleeding is NOT invalidation while protection is cheap."""
    states = [
        StratState(BULL, 0.35, 20, pnl_frac=0.08, short_sigma=-2.0, min_dte=9),
        StratState(BEAR, 0.35, 20, pnl_frac=0.06, short_sigma=2.0, min_dte=9),
        StratState(STRANGLE, 0.20, 6, pnl_frac=-0.45, short_sigma=None, min_dte=20, drawdown=0.5, days_held=6),
    ]
    return Scenario(
        name="strangle_bleeds_protection_cheap",
        regime="Strangle bleeding, IV still cheap",
        description=(
            "The strangle has lost nearly half its premium over six quiet days "
            "and is the worst performer in the book. But IV sits at the 18th "
            "percentile -- protection is cheap, and the conditions for an "
            "expansion have not resolved."
        ),
        expectation=(
            "HOLD the strangle. Cutting it here is the exact error the system "
            "exists to avoid: selling convexity for a loss right before it might "
            "pay, on the basis of P&L alone. Its not_invalidation says bleeding "
            "is what it does while it waits, and protection is still cheap."
        ),
        packet=_packet(
            cycle_id="shock-strangle_bleeds_protection_cheap",
            spot=714.0,
            returns={"1d": -0.001, "5d": 0.002, "10d": 0.005, "21d": 0.008, "63d": 0.02},
            rv=0.13,
            iv=0.16,
            iv_pct=18.0,
            states=states,
        ),
        current_allocation=_current(states),
    )


def strangle_protection_expensive() -> Scenario:
    """The counter-case: cut the strangle when protection is genuinely expensive."""
    states = [
        StratState(BULL, 0.35, 20, pnl_frac=0.05, short_sigma=-1.8, min_dte=9),
        StratState(BEAR, 0.35, 20, pnl_frac=0.04, short_sigma=1.8, min_dte=9),
        StratState(STRANGLE, 0.20, 6, pnl_frac=0.15, short_sigma=None, min_dte=22, days_held=4),
    ]
    return Scenario(
        name="strangle_protection_expensive",
        regime="Strangle in profit, but IV now expensive",
        description=(
            "The strangle is modestly profitable, but IV has risen to the 86th "
            "percentile -- protection is now expensive, so the position is "
            "paying a high price for the same convexity going forward."
        ),
        expectation=(
            "Reduce the strangle -- this is its GENUINE invalidation (protection "
            "no longer cheap), which is different from cutting it for a loss. "
            "The contrast with the bleeding-but-cheap case is the whole point: "
            "P&L is not the trigger, the IV percentile is."
        ),
        packet=_packet(
            cycle_id="shock-strangle_protection_expensive",
            spot=718.0,
            returns={"1d": 0.002, "5d": 0.006, "10d": 0.01, "21d": 0.015, "63d": 0.028},
            rv=0.19,
            iv=0.29,
            iv_pct=86.0,
            states=states,
        ),
        current_allocation=_current(states),
    )


SCENARIOS = (
    baseline_quiet,
    volatility_shock,
    directional_selloff_breach,
    melt_up_breach,
    strangle_bleeds_but_protection_cheap,
    strangle_protection_expensive,
)


def all_scenarios() -> list[Scenario]:
    return [build() for build in SCENARIOS]


# ---------------------------------------------------------------------------
# Running a scenario through the real decision path
# ---------------------------------------------------------------------------


def run_scenario(scenario: Scenario, allocator, challenger) -> dict:
    """Run one scenario through the real allocator, challenger, and gates.

    Returns a record for the report/artifact. Any model failure is caught and
    recorded (a hold), so one flaky call does not abort the whole run.
    """
    from . import risk
    from .challenger import resolve_effective_allocation
    from .llm import ModelUnavailable

    record: dict = {
        "name": scenario.name,
        "regime": scenario.regime,
        "description": scenario.description,
        "expectation": scenario.expectation,
        "current_allocation": scenario.current_allocation,
        "simulated": True,
    }
    try:
        proposal = allocator.propose(scenario.packet)
    except ModelUnavailable as exc:
        record["error"] = f"allocator unavailable: {exc}"
        return record
    record["proposal"] = proposal.to_dict()

    try:
        challenge = challenger.review(scenario.packet, proposal)
    except ModelUnavailable as exc:
        record["error"] = f"challenger unavailable: {exc}"
        return record
    record["challenge"] = challenge.to_dict()

    effective, source = resolve_effective_allocation(
        proposal, challenge, scenario.current_allocation
    )
    gated = risk.apply_gates(effective, scenario.current_allocation)
    record["effective_source"] = source
    record["final_allocation"] = gated.final
    record["gate_adjustments"] = [a.rule for a in gated.adjustments]
    return record
