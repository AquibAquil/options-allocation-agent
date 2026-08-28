"""The evidence packet (PRD 2.1).

Emits evidence, not verdicts. Nothing in this module decides whether a thesis
holds; it assembles the facts the allocator needs in order to decide, and hands
over the thesis text unchanged alongside them.

Three sections, matching PRD 2.1:

  market      one underlying, shared by all three strategies
  strategies  per-strategy state and option-level detail
  portfolio   exposure, risk budget consumed against available, buying power

plus the precomputed correlation matrix and the constraints. The constraints
are restated here as INPUT so the allocator knows the shape of the box it is
proposing into. They are enforced separately on its output, because you cannot
bound a proposal that does not exist yet (PRD 2.2).

Live position and chain data arrives through MCP. This module never calls it:
it takes plain snapshots, which keeps assembly pure and testable before any
credential exists.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import RISK
from ..strategies import BY_KEY, KEYS, Strategy
from . import metrics


# ---------------------------------------------------------------------------
# Inputs supplied by the MCP layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegSnapshot:
    """One option leg as the chain currently reports it."""

    occ_symbol: str
    right: str              # "put" | "call"
    action: str             # "sell" | "buy"
    strike: float
    expiry: str             # ISO date
    contracts: int
    delta: float | None
    mid: float | None       # per share
    implied_vol: float | None
    bid: float | None = None
    ask: float | None = None
    open_interest: int | None = None


@dataclass(frozen=True)
class PositionSnapshot:
    """A strategy's live position. Absent entirely when flat."""

    strategy_key: str
    contracts: int
    legs: tuple[LegSnapshot, ...]
    max_loss_per_contract: float
    entry_premium: float           # per contract, credit positive / debit positive
    opened_at: str                 # ISO date
    unrealized_pnl: float          # dollars, whole position
    equity_curve: tuple[float, ...] = ()   # position value by cycle, for drawdown


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    buying_power: float
    options_buying_power: float | None = None


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketEvidence:
    symbol: str
    spot: float
    spot_asof: str
    returns: dict[str, float | None]
    realized_vol: dict[str, float | None]
    parkinson_vol_21d: float | None
    implied_vol: float
    implied_vol_source: str
    implied_vol_asof: str
    implied_vol_percentile_252d: float
    implied_vol_percentile_full: float
    iv_rv: dict[str, float]
    bars_are_complete_sessions: bool = True


@dataclass(frozen=True)
class LegEvidence:
    occ_symbol: str
    right: str
    action: str
    strike: float
    expiry: str
    dte: int
    contracts: int
    delta: float | None
    mid: float | None
    implied_vol: float | None
    strike_distance_pct: float
    strike_distance_sigma: float | None
    bid_ask_spread: float | None


@dataclass(frozen=True)
class StrategyEvidence:
    key: str
    name: str
    thesis: str
    invalidation: str
    not_invalidation: str
    exit_behaviour: str
    vol_exposure: str
    direction: str

    # Allocation and risk
    allocation_frac: float          # share of the risk budget currently held
    max_loss_outstanding: float     # dollars
    contracts: int

    # Performance
    unrealized_pnl: float | None
    pnl_frac_of_max_loss: float | None
    drawdown_frac: float | None
    days_held: int | None

    # Structure
    legs: tuple[LegEvidence, ...]
    short_strike_distance_sigma: float | None
    min_dte: int | None


@dataclass(frozen=True)
class PortfolioEvidence:
    equity: float
    buying_power: float
    options_buying_power: float | None
    risk_budget_total: float           # dollars of permitted max loss
    risk_budget_consumed: float
    risk_budget_available: float
    risk_budget_utilisation: float     # consumed / total
    total_max_loss_outstanding: float
    max_loss_as_frac_of_equity: float


@dataclass(frozen=True)
class Constraints:
    """Restated as input so the allocator proposes inside the right box.

    Enforcement happens on output, in the risk gates, regardless of what is
    written here.
    """

    total_budget_frac_of_equity: float
    per_strategy_max_frac_of_budget: float
    snap_to_zero_below_frac: float
    adjustment_threshold_frac: float
    allocations_must_sum_to_at_most: float = 1.0

    @classmethod
    def current(cls) -> "Constraints":
        return cls(
            total_budget_frac_of_equity=RISK.total_budget_frac,
            per_strategy_max_frac_of_budget=RISK.per_strategy_max,
            snap_to_zero_below_frac=RISK.snap_to_zero_below,
            adjustment_threshold_frac=RISK.adjustment_threshold,
        )


@dataclass(frozen=True)
class EvidencePacket:
    asof: str
    cycle_id: str
    market: MarketEvidence
    strategies: tuple[StrategyEvidence, ...]
    portfolio: PortfolioEvidence
    correlation: dict
    constraints: Constraints
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_market_evidence(
    *,
    symbol: str,
    bars: list[dict],
    vol_index_rows: list[dict],
    vol_index_name: str = "VXN",
) -> MarketEvidence:
    """Underlying facts, from completed daily bars and the IV reference series.

    Volatility is anchored to completed bars (PRD 2.7): on the Basic plan the
    most recent 15 minutes of historical bars cannot be pulled, so a partial
    session must never enter a volatility calculation.
    """
    if len(bars) < 2:
        raise ValueError("need at least two bars")

    closes = np.array([float(b["c"]) for b in bars], dtype=float)
    highs = np.array([float(b["h"]) for b in bars], dtype=float)
    lows = np.array([float(b["l"]) for b in bars], dtype=float)
    log_returns = np.diff(np.log(closes))

    realized = metrics.realized_volatility(log_returns)
    # The 21-day window is the reference for the IV/RV relationship: long
    # enough to be stable, short enough to describe the current regime.
    rv_reference = realized.get("21d") or realized.get("10d") or realized.get("5d")
    if rv_reference is None:
        raise ValueError("no realised volatility window has enough data")

    iv_closes = np.array([r["close"] for r in vol_index_rows], dtype=float)
    iv_now = float(iv_closes[-1]) / 100.0  # index points are vol percentage points

    return MarketEvidence(
        symbol=symbol,
        spot=float(closes[-1]),
        spot_asof=str(bars[-1]["t"])[:10],
        returns=metrics.multi_horizon_returns(closes),
        realized_vol=realized,
        parkinson_vol_21d=metrics.parkinson_volatility(highs, lows, window=21),
        implied_vol=iv_now,
        implied_vol_source=f"CBOE {vol_index_name}",
        implied_vol_asof=vol_index_rows[-1]["date"],
        implied_vol_percentile_252d=metrics.percentile_rank(
            metrics.trailing_window(iv_closes[:-1], 252), iv_closes[-1]
        ),
        implied_vol_percentile_full=metrics.percentile_rank(
            iv_closes[:-1], iv_closes[-1]
        ),
        iv_rv=metrics.vol_risk_premium(iv_now, rv_reference),
    )


def _dte(expiry: str, asof: dt.date) -> int:
    return (dt.date.fromisoformat(expiry) - asof).days


def build_leg_evidence(
    leg: LegSnapshot, *, spot: float, annual_vol: float, asof: dt.date
) -> LegEvidence:
    dte = _dte(leg.expiry, asof)
    spread = (
        float(leg.ask - leg.bid)
        if leg.bid is not None and leg.ask is not None
        else None
    )
    distance_sigma = (
        metrics.sigma_distance(spot, leg.strike, annual_vol, dte) if dte > 0 else None
    )
    return LegEvidence(
        occ_symbol=leg.occ_symbol,
        right=leg.right,
        action=leg.action,
        strike=leg.strike,
        expiry=leg.expiry,
        dte=dte,
        contracts=leg.contracts,
        delta=leg.delta,
        mid=leg.mid,
        implied_vol=leg.implied_vol,
        strike_distance_pct=float(leg.strike / spot - 1.0),
        strike_distance_sigma=distance_sigma,
        bid_ask_spread=spread,
    )


def build_strategy_evidence(
    strategy: Strategy,
    position: PositionSnapshot | None,
    *,
    spot: float,
    annual_vol: float,
    risk_budget_total: float,
    asof: dt.date,
) -> StrategyEvidence:
    """One strategy's state. A flat strategy still appears, at zero.

    A strategy the allocator has cut to nothing must remain visible with its
    thesis attached, or it can never be funded again.
    """
    base = dict(
        key=strategy.key,
        name=strategy.name,
        thesis=strategy.thesis,
        invalidation=strategy.invalidation,
        not_invalidation=strategy.not_invalidation,
        exit_behaviour=strategy.exit_behaviour.value,
        vol_exposure=strategy.vol_exposure.value,
        direction=strategy.direction,
    )

    if position is None or position.contracts == 0:
        return StrategyEvidence(
            **base,
            allocation_frac=0.0,
            max_loss_outstanding=0.0,
            contracts=0,
            unrealized_pnl=None,
            pnl_frac_of_max_loss=None,
            drawdown_frac=None,
            days_held=None,
            legs=(),
            short_strike_distance_sigma=None,
            min_dte=None,
        )

    max_loss = position.max_loss_per_contract * position.contracts
    legs = tuple(
        build_leg_evidence(leg, spot=spot, annual_vol=annual_vol, asof=asof)
        for leg in position.legs
    )

    short_legs = [leg for leg in legs if leg.action == "sell"]
    short_distance = None
    if short_legs:
        finite = [
            leg.strike_distance_sigma
            for leg in short_legs
            if leg.strike_distance_sigma is not None
        ]
        # The threatened strike is the nearest one, whichever side it sits on.
        short_distance = min(finite, key=abs) if finite else None

    drawdown = None
    if len(position.equity_curve) >= 2 and max_loss > 0:
        # Express the position's path as a fraction of max loss remaining, so a
        # drawdown is measured against risk rather than against an arbitrary base.
        curve = np.array(position.equity_curve, dtype=float) + max_loss
        if np.all(curve > 0):
            drawdown = metrics.max_drawdown(curve)

    return StrategyEvidence(
        **base,
        allocation_frac=(max_loss / risk_budget_total) if risk_budget_total > 0 else 0.0,
        max_loss_outstanding=max_loss,
        contracts=position.contracts,
        unrealized_pnl=position.unrealized_pnl,
        pnl_frac_of_max_loss=(
            position.unrealized_pnl / max_loss if max_loss > 0 else None
        ),
        drawdown_frac=drawdown,
        days_held=(asof - dt.date.fromisoformat(position.opened_at)).days,
        legs=legs,
        short_strike_distance_sigma=short_distance,
        min_dte=min((leg.dte for leg in legs), default=None),
    )


def build_portfolio_evidence(
    account: AccountSnapshot, strategies: tuple[StrategyEvidence, ...]
) -> PortfolioEvidence:
    budget_total = account.equity * RISK.total_budget_frac
    consumed = sum(s.max_loss_outstanding for s in strategies)
    return PortfolioEvidence(
        equity=account.equity,
        buying_power=account.buying_power,
        options_buying_power=account.options_buying_power,
        risk_budget_total=budget_total,
        risk_budget_consumed=consumed,
        risk_budget_available=budget_total - consumed,
        risk_budget_utilisation=(consumed / budget_total) if budget_total > 0 else 0.0,
        total_max_loss_outstanding=consumed,
        max_loss_as_frac_of_equity=(
            consumed / account.equity if account.equity > 0 else 0.0
        ),
    )


def build_packet(
    *,
    cycle_id: str,
    symbol: str,
    bars: list[dict],
    vol_index_rows: list[dict],
    positions: dict[str, PositionSnapshot | None],
    account: AccountSnapshot,
    correlation: dict,
    asof: dt.date | None = None,
    notes: tuple[str, ...] = (),
) -> EvidencePacket:
    asof = asof or dt.date.today()
    market = build_market_evidence(
        symbol=symbol, bars=bars, vol_index_rows=vol_index_rows
    )

    # Strike distances are measured against the current regime, not the
    # long-run average, so a quiet tape does not make every strike look safe.
    annual_vol = (
        market.realized_vol.get("21d")
        or market.realized_vol.get("10d")
        or market.implied_vol
    )
    risk_budget_total = account.equity * RISK.total_budget_frac

    unknown = set(positions) - set(KEYS)
    if unknown:
        raise ValueError(f"positions contains unknown strategy keys: {sorted(unknown)}")

    strategies = tuple(
        build_strategy_evidence(
            BY_KEY[key],
            positions.get(key),
            spot=market.spot,
            annual_vol=annual_vol,
            risk_budget_total=risk_budget_total,
            asof=asof,
        )
        for key in KEYS
    )

    extra_notes = list(notes)
    if market.implied_vol_asof < market.spot_asof:
        extra_notes.append(
            f"IV reference ({market.implied_vol_source}) is as of "
            f"{market.implied_vol_asof}, behind spot at {market.spot_asof}"
        )

    return EvidencePacket(
        asof=asof.isoformat(),
        cycle_id=cycle_id,
        market=market,
        strategies=strategies,
        portfolio=build_portfolio_evidence(account, strategies),
        correlation=correlation,
        constraints=Constraints.current(),
        notes=tuple(extra_notes),
    )
