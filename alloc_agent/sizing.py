"""Sizing: target percentages to contract counts (PRD 2.1, 2.2).

Own code, not the LLM. The allocator never sees a contract count and is never
asked to produce one; it proposes a share of the risk budget, and this module
turns that into an integer number of contracts using the max-loss definition,
then checks buying power separately.

Buying power is a HARD GATE, not the denominator. Margin requirement is a
broker constraint independent of how risk is defined, so a plan can be inside
the risk budget and still be unaffordable. Both are checked, and the failure
reasons are distinct.

Chain verification happens before every order (PRD 2.5). Option conditions
change between decision and execution, so nothing here trusts a premium or a
delta that was read at decision time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RISK
from .strategies import BY_KEY, ExitBehaviour, Strategy


@dataclass(frozen=True)
class ChainQuote:
    """A live structure as the chain reports it, at execution time."""

    strategy_key: str
    max_loss_per_contract: float
    premium_per_contract: float       # credit collected or debit paid, positive
    strike_width: float | None        # None for the strangle
    min_dte: int
    short_leg_delta: float | None
    bid_ask_spread: float             # widest leg, per share
    open_interest_min: int | None = None


@dataclass(frozen=True)
class SizingPlan:
    strategy_key: str
    target_alloc_frac: float
    target_max_loss: float
    contracts: int
    contracts_current: int
    contract_delta: int
    actual_max_loss: float
    actual_alloc_frac: float
    estimated_margin: float
    action: str                        # open | increase | reduce | close | hold
    blocked_reason: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked_reason is not None

    def to_dict(self) -> dict:
        return {
            "strategy_key": self.strategy_key,
            "target_alloc_frac": self.target_alloc_frac,
            "target_max_loss": self.target_max_loss,
            "contracts": self.contracts,
            "contracts_current": self.contracts_current,
            "contract_delta": self.contract_delta,
            "actual_max_loss": self.actual_max_loss,
            "actual_alloc_frac": self.actual_alloc_frac,
            "estimated_margin": self.estimated_margin,
            "action": self.action,
            "blocked_reason": self.blocked_reason,
        }


class ChainRejected(ValueError):
    """The live chain does not match what the decision was made on."""


def verify_chain(
    quote: ChainQuote,
    strategy: Strategy,
    *,
    max_bid_ask: float = 0.15,
    delta_tolerance: float = 0.10,
) -> None:
    """Reject a structure the live chain no longer supports (PRD 2.5).

    Raises rather than returning a flag: a failed verification must stop the
    order, not be weighed against anything.
    """
    if quote.strategy_key != strategy.key:
        raise ChainRejected(f"quote is for {quote.strategy_key}, not {strategy.key}")

    if not strategy.dte_min <= quote.min_dte <= strategy.dte_max:
        raise ChainRejected(
            f"{strategy.key}: DTE {quote.min_dte} outside "
            f"{strategy.dte_min}-{strategy.dte_max}"
        )

    if quote.max_loss_per_contract <= 0:
        raise ChainRejected(
            f"{strategy.key}: max loss per contract is {quote.max_loss_per_contract}"
        )

    if quote.premium_per_contract <= 0:
        raise ChainRejected(f"{strategy.key}: premium is {quote.premium_per_contract}")

    if strategy.collects_premium:
        if quote.strike_width is None:
            raise ChainRejected(f"{strategy.key}: credit spread has no strike width")
        expected = quote.strike_width * 100.0 - quote.premium_per_contract
        if not math.isclose(quote.max_loss_per_contract, expected, rel_tol=1e-6):
            raise ChainRejected(
                f"{strategy.key}: max loss {quote.max_loss_per_contract:.2f} does not "
                f"match width minus credit {expected:.2f}; the structure is not what "
                "it claims to be"
            )
    elif not math.isclose(
        quote.max_loss_per_contract, quote.premium_per_contract, rel_tol=1e-6
    ):
        raise ChainRejected(
            f"{strategy.key}: max loss must equal the debit paid, got "
            f"{quote.max_loss_per_contract:.2f} vs {quote.premium_per_contract:.2f}"
        )

    # A short leg that has drifted far from its target delta is a different
    # trade from the one the thesis describes.
    target = next(
        (leg.target_delta for leg in strategy.legs if leg.action == "sell"), None
    )
    if target is not None and quote.short_leg_delta is not None:
        drift = abs(abs(quote.short_leg_delta) - target)
        if drift > delta_tolerance:
            raise ChainRejected(
                f"{strategy.key}: short leg delta {abs(quote.short_leg_delta):.3f} "
                f"is {drift:.3f} from the {target:.2f} target"
            )

    if quote.bid_ask_spread > max_bid_ask:
        raise ChainRejected(
            f"{strategy.key}: widest leg bid-ask {quote.bid_ask_spread:.2f} exceeds "
            f"{max_bid_ask:.2f}; crossing it would cost more than the edge"
        )


def min_contracts_for_granularity(alloc_frac: float) -> int:
    """Smallest position where a one-contract change is not coarser than the
    adjustment threshold.

    PRD 2.3 requires the strangle to hold enough contracts per leg that partial
    reduction is meaningful, without naming a number. This derives one instead
    of inventing it: at n contracts holding `alloc_frac` of the budget, one
    contract moves the allocation by alloc_frac/n, and that step should be no
    larger than the 5 percentage point threshold below which the system does
    not trade at all.
    """
    if alloc_frac <= 0:
        return 0
    return max(1, math.ceil(alloc_frac / RISK.adjustment_threshold))


def estimate_margin(quote: ChainQuote, contracts: int) -> float:
    """Margin a defined-risk structure is expected to hold.

    For a credit spread this is max loss; for a long strangle it is the debit,
    paid in full. Both are the theoretical values.

    PRD 2.7 flags what paper actually holds for a defined-risk spread as an
    open question to resolve on day one with a single small order. Until that
    is measured this is an estimate, and the estimate is deliberately the
    conservative reading rather than an optimistic one.
    """
    return quote.max_loss_per_contract * max(contracts, 0)


def size_strategy(
    strategy_key: str,
    *,
    target_alloc_frac: float,
    risk_budget_total: float,
    quote: ChainQuote,
    contracts_current: int = 0,
    buying_power: float = float("inf"),
) -> SizingPlan:
    """Convert one target allocation into a contract count and an action."""
    strategy = BY_KEY[strategy_key]
    verify_chain(quote, strategy)

    if risk_budget_total <= 0:
        raise ValueError("risk_budget_total must be positive")

    target_max_loss = target_alloc_frac * risk_budget_total
    contracts = int(math.floor(target_max_loss / quote.max_loss_per_contract))

    blocked: str | None = None

    # A scale-down-only strategy must not be reduced below the point where a
    # further reduction is expressible (PRD 2.3).
    if (
        strategy.exit_behaviour is ExitBehaviour.SCALE_DOWN_ONLY
        and target_alloc_frac > 0
        and contracts > 0
    ):
        floor_contracts = min_contracts_for_granularity(target_alloc_frac)
        if contracts < floor_contracts:
            affordable = int(
                math.floor(risk_budget_total * RISK.per_strategy_max / quote.max_loss_per_contract)
            )
            if floor_contracts <= affordable:
                contracts = floor_contracts
            else:
                blocked = (
                    f"{strategy.name} needs {floor_contracts} contracts for a "
                    f"reduction to be meaningful but only {affordable} fit the "
                    f"{RISK.per_strategy_max:.0%} per-strategy cap"
                )
                contracts = 0

    if target_alloc_frac > 0 and contracts == 0 and blocked is None:
        blocked = (
            f"{strategy.name}: {target_alloc_frac:.1%} of a "
            f"${risk_budget_total:,.0f} budget is ${target_max_loss:,.0f}, short of "
            f"${quote.max_loss_per_contract:,.2f} for one contract"
        )

    actual_max_loss = contracts * quote.max_loss_per_contract
    margin = estimate_margin(quote, contracts)
    delta = contracts - contracts_current

    # Buying power gates INCREASES only. Never block a reduction on affordability.
    if delta > 0 and blocked is None:
        incremental = estimate_margin(quote, delta)
        if incremental > buying_power:
            affordable_delta = int(math.floor(buying_power / quote.max_loss_per_contract))
            contracts = contracts_current + max(affordable_delta, 0)
            actual_max_loss = contracts * quote.max_loss_per_contract
            margin = estimate_margin(quote, contracts)
            delta = contracts - contracts_current
            if delta <= 0:
                blocked = (
                    f"{strategy.name}: buying power ${buying_power:,.0f} does not "
                    f"cover one more contract at ${quote.max_loss_per_contract:,.2f}"
                )

    if delta == 0:
        action = "hold"
    elif contracts_current == 0:
        action = "open"
    elif contracts == 0:
        action = "close"
    elif delta > 0:
        action = "increase"
    else:
        action = "reduce"

    return SizingPlan(
        strategy_key=strategy_key,
        target_alloc_frac=target_alloc_frac,
        target_max_loss=target_max_loss,
        contracts=contracts,
        contracts_current=contracts_current,
        contract_delta=delta,
        actual_max_loss=actual_max_loss,
        actual_alloc_frac=actual_max_loss / risk_budget_total,
        estimated_margin=margin,
        action=action,
        blocked_reason=blocked,
    )


def size_portfolio(
    allocations: dict[str, float],
    *,
    risk_budget_total: float,
    quotes: dict[str, ChainQuote],
    contracts_current: dict[str, int] | None = None,
    buying_power: float,
) -> dict[str, SizingPlan]:
    """Size every strategy, spending buying power on reductions first.

    Ordering matters: closing or trimming a position releases margin that an
    increase elsewhere may need. Doing increases first can block a rotation
    that the account can actually afford.
    """
    contracts_current = contracts_current or {}
    plans: dict[str, SizingPlan] = {}
    remaining = buying_power

    def rank(item: tuple[str, float]) -> float:
        key, target = item
        held = contracts_current.get(key, 0)
        quote = quotes.get(key)
        if quote is None or held == 0:
            return 1.0
        target_contracts = math.floor(
            target * risk_budget_total / quote.max_loss_per_contract
        )
        return 0.0 if target_contracts < held else 1.0

    for key, target in sorted(allocations.items(), key=rank):
        quote = quotes.get(key)
        if quote is None:
            if target > 0:
                raise ChainRejected(f"{key}: no live quote for a non-zero allocation")
            continue

        plan = size_strategy(
            key,
            target_alloc_frac=target,
            risk_budget_total=risk_budget_total,
            quote=quote,
            contracts_current=contracts_current.get(key, 0),
            buying_power=remaining,
        )
        plans[key] = plan

        if plan.contract_delta > 0:
            remaining -= estimate_margin(quote, plan.contract_delta)
        elif plan.contract_delta < 0:
            remaining += estimate_margin(quote, -plan.contract_delta)

    return plans
