"""Order construction and fill interpretation (PRD 2.5, 2.6).

Own code, deliberately split from the MCP call. This module turns a sizing plan
plus chosen contracts into an exact, validated order spec, and turns the order
status the broker returns back into a verdict. It never places anything itself.

Why the split. PRD 2.6 puts execution through the Alpaca MCP server, and those
tools are invoked by the agent at runtime, not by library code. So the testable,
deterministic work -- building the legs, the net limit price, the idempotency
key, and reading the fill -- lives here, and the single side-effecting step
(calling place_option_order with the spec this produces) is the agent's action.
Everything in this file runs and is tested offline.

Two hard requirements from the PRD live here:

  Idempotency (PRD 2.5). Every order carries a deterministic client_order_id.
  If a submission times out, retrying with the same id is safe: the broker
  rejects the duplicate rather than opening a second position. A transient
  software problem must not double a position.

  Fill verification (PRD 2.5, 2.7). After submission the agent checks the
  order's ACTUAL status before doing anything else. interpret_fill turns that
  status into a verdict, and specifically answers the day-one question of
  whether a multi-leg order filled atomically or leg by leg -- an unpaired
  short leg is the risk a spread is supposed to preclude.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass

from .sizing import ChainQuote, SizingPlan
from .strategies import BY_KEY, ExitBehaviour


# ---------------------------------------------------------------------------
# OCC symbols
# ---------------------------------------------------------------------------


def build_occ_symbol(underlying: str, expiry: str, right: str, strike: float) -> str:
    """Construct an OCC option symbol, matching what Alpaca's chain returns.

    QQQ, 2026-09-11, put, 699.0 -> "QQQ260911P00699000"

    Root is left unpadded, which is how Alpaca renders QQQ (and the exact form
    the chain endpoint returned). Strike is in thousandths, zero-padded to 8.
    """
    date = dt.date.fromisoformat(expiry)
    cp = {"call": "C", "put": "P"}.get(right)
    if cp is None:
        raise ValueError(f"right must be call or put, got {right!r}")
    strike_thousandths = round(strike * 1000)
    if abs(strike_thousandths - strike * 1000) > 1e-6:
        raise ValueError(f"strike {strike} is finer than the $0.001 OCC grid")
    return f"{underlying.upper()}{date:%y%m%d}{cp}{strike_thousandths:08d}"


def parse_occ_symbol(symbol: str) -> dict:
    """Inverse of build_occ_symbol, for validating chain-returned symbols.

    Splits on the single C/P that separates the date from the strike.
    """
    for i, ch in enumerate(symbol):
        if ch in ("C", "P") and i >= 7:
            root, date_str, right_ch, strike_str = (
                symbol[: i - 6],
                symbol[i - 6 : i],
                ch,
                symbol[i + 1 :],
            )
            return {
                "underlying": root,
                "expiry": dt.datetime.strptime(date_str, "%y%m%d").date().isoformat(),
                "right": "call" if right_ch == "C" else "put",
                "strike": int(strike_str) / 1000.0,
            }
    raise ValueError(f"not a parseable OCC symbol: {symbol!r}")


# ---------------------------------------------------------------------------
# Order spec
# ---------------------------------------------------------------------------

_OPEN_INTENT = {"buy": "buy_to_open", "sell": "sell_to_open"}
_CLOSE_INTENT = {"buy": "buy_to_close", "sell": "sell_to_close"}


@dataclass(frozen=True)
class OrderLeg:
    occ_symbol: str
    side: str              # buy | sell
    ratio_qty: int         # per-leg ratio; 1 for all three strategies here
    position_intent: str   # buy_to_open | sell_to_open | buy_to_close | sell_to_close
    reference_mid: float   # per-share mid at construction, for the net price

    def to_mcp_leg(self) -> dict:
        return {
            "symbol": self.occ_symbol,
            "ratio_qty": str(self.ratio_qty),
            "side": self.side,
            "position_intent": self.position_intent,
        }


@dataclass(frozen=True)
class OrderSpec:
    strategy_key: str
    intent: str                     # open | close | increase | reduce
    qty: int                        # strategy multiplier (number of structures)
    legs: tuple[OrderLeg, ...]
    limit_price: float              # net; negative = credit, positive = debit
    client_order_id: str
    order_type: str = "limit"
    time_in_force: str = "day"

    @property
    def order_class(self) -> str:
        return "mleg" if len(self.legs) > 1 else "simple"

    @property
    def is_credit(self) -> bool:
        return self.limit_price < 0

    def to_mcp_kwargs(self) -> dict:
        """Exactly the kwargs for the place_option_order MCP tool.

        Multi-leg: qty is the strategy multiplier; each leg carries its own
        side and ratio. limit_price is the net, signed as the tool documents
        (negative = credit received, positive = debit paid).
        """
        return {
            "qty": str(self.qty),
            "legs": [leg.to_mcp_leg() for leg in self.legs],
            "order_class": "mleg",
            "type": self.order_type,
            "limit_price": f"{self.limit_price:.2f}",
            "time_in_force": self.time_in_force,
            "client_order_id": self.client_order_id,
        }

    def human_summary(self) -> str:
        verb = "credit" if self.is_credit else "debit"
        lines = [
            f"{BY_KEY[self.strategy_key].name}  ({self.intent}, {self.qty}x)",
        ]
        for leg in self.legs:
            lines.append(
                f"  {leg.side:4} {leg.ratio_qty}x {leg.occ_symbol} "
                f"@ ~{leg.reference_mid:.2f}  [{leg.position_intent}]"
            )
        lines.append(f"  net {verb}: {abs(self.limit_price):.2f}/share")
        return "\n".join(lines)


@dataclass(frozen=True)
class ContractLeg:
    """One leg the agent has picked from the live chain for an order."""

    occ_symbol: str
    right: str
    action: str            # buy | sell
    strike: float
    expiry: str
    mid: float             # per-share mid, from the live snapshot
    bid: float | None = None   # for marketable pricing
    ask: float | None = None


def _client_order_id(strategy_key: str, intent: str, qty: int, legs: tuple[ContractLeg, ...]) -> str:
    """Deterministic idempotency key (PRD 2.5).

    Same intent and same structure -> same id, so a timed-out submission is
    safe to retry. A different structure (re-picked strikes) -> different id,
    because it is genuinely a different order.
    """
    payload = "|".join(
        f"{leg.action}:{leg.occ_symbol}" for leg in legs
    )
    digest = hashlib.sha1(
        f"{strategy_key}|{intent}|{qty}|{payload}".encode()
    ).hexdigest()[:10]
    return f"alloc-{strategy_key}-{intent}-{digest}"


def build_order(
    plan: SizingPlan,
    contracts: tuple[ContractLeg, ...],
    *,
    intent: str | None = None,
    slippage: float = 0.02,
    marketable: bool = False,
) -> OrderSpec:
    """Build a validated multi-leg order spec from a sizing plan and chosen legs.

    `slippage` (per share, non-negative) makes the net price marginally
    marketable in the correct direction: accept slightly less credit, or pay
    slightly more debit. It never flips the sign of the net.

    `marketable` prices each leg on the side that fills against resting liquidity
    -- buy at the ask, sell at the bid -- instead of at mid. A debit structure
    (long strangle) fills at mid regardless because you are buying; a CREDIT
    spread priced at mid rests until a buyer meets it, so it must be priced
    marketably or it never fills. Falls back to mid for any leg missing a quote.
    """
    strategy = BY_KEY[plan.strategy_key]
    if slippage < 0:
        raise ValueError("slippage must be non-negative")

    # The substantive reason first: a plan with no delta is nothing to trade,
    # whatever action label it carries.
    qty = abs(plan.contract_delta)
    if qty == 0:
        raise ValueError(f"{plan.strategy_key}: nothing to trade (delta 0)")
    if not contracts:
        raise ValueError("no contracts supplied")

    intent = intent or plan.action
    if intent not in ("open", "close", "increase", "reduce"):
        raise ValueError(f"unexpected intent {intent!r}")

    # Closing a scale-down-only strategy fully is forbidden unless the plan
    # explicitly targets zero (PRD 2.3). The gate enforces this upstream; this
    # is a last-ditch guard so a bug cannot slam a strangle shut at the worst
    # possible moment.
    if (
        strategy.exit_behaviour is ExitBehaviour.SCALE_DOWN_ONLY
        and intent == "close"
        and plan.target_alloc_frac > 0
    ):
        raise ValueError(
            f"{strategy.name} is scale-down-only; refusing to fully close while "
            f"target allocation is {plan.target_alloc_frac:.1%}, not zero"
        )

    closing = intent in ("close", "reduce")
    intent_map = _CLOSE_INTENT if closing else _OPEN_INTENT

    legs: list[OrderLeg] = []
    net_cost = 0.0  # Sigma(buy) - Sigma(sell); negative => net credit
    for c in contracts:
        if c.mid <= 0:
            raise ValueError(f"{c.occ_symbol}: non-positive mid {c.mid}")
        # Reducing/closing reverses each leg's side relative to the open.
        side = c.action
        if closing:
            side = "sell" if c.action == "buy" else "buy"

        # Price this leg. Marketable crosses the spread on the side that fills.
        price = c.mid
        if marketable and c.bid is not None and c.ask is not None and c.bid > 0 and c.ask > 0:
            price = c.ask if side == "buy" else c.bid

        legs.append(
            OrderLeg(
                occ_symbol=c.occ_symbol,
                side=side,
                ratio_qty=1,
                position_intent=intent_map[side],
                reference_mid=c.mid,
            )
        )
        net_cost += price if side == "buy" else -price

    limit_price = round(net_cost + slippage, 2)
    # Guard the sign: a credit structure must not round into a debit limit.
    if net_cost < 0 and limit_price >= 0:
        limit_price = round(net_cost, 2)

    return OrderSpec(
        strategy_key=plan.strategy_key,
        intent=intent,
        qty=qty,
        legs=tuple(legs),
        limit_price=limit_price,
        client_order_id=_client_order_id(plan.strategy_key, intent, qty, contracts),
    )


# ---------------------------------------------------------------------------
# Fill interpretation (PRD 2.5, 2.7)
# ---------------------------------------------------------------------------

# Statuses that mean the order is done moving and will not fill further.
_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}
_RESTING = {"new", "accepted", "pending_new", "accepted_for_bidding", "held", "partially_filled"}


@dataclass(frozen=True)
class FillReport:
    order_id: str | None
    client_order_id: str | None
    status: str
    verdict: str              # filled | partial_legs | working | unfilled | rejected | unknown
    filled_qty: int
    ordered_qty: int
    leg_fills: tuple[tuple[str, int], ...]
    atomic: bool | None       # True=all legs moved together, False=leg imbalance, None=n/a
    detail: str

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    @property
    def has_leg_imbalance(self) -> bool:
        return self.atomic is False


def interpret_fill(order: dict) -> FillReport:
    """Turn a broker order object into a verdict (PRD 2.5, 2.7).

    Answers, for a multi-leg order, whether the legs filled together. Unequal
    leg fills mean an unpaired short leg -- naked risk the spread was supposed
    to define away -- and that is the single most important thing to detect
    after submitting.

    Written defensively: broker payloads vary and this reads a live account, so
    every field access tolerates absence rather than trusting a shape.
    """
    status = str(order.get("status", "unknown")).lower()
    order_id = order.get("id")
    client_order_id = order.get("client_order_id")

    def _int(value) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    ordered_qty = _int(order.get("qty"))
    filled_qty = _int(order.get("filled_qty"))

    legs = order.get("legs") or []
    leg_fills = tuple((str(leg.get("symbol", "?")), _int(leg.get("filled_qty"))) for leg in legs)

    atomic: bool | None = None
    if leg_fills:
        distinct = {qty for _, qty in leg_fills}
        atomic = len(distinct) == 1

    if status == "filled":
        verdict = "filled"
        detail = f"filled {filled_qty}/{ordered_qty}"
        if atomic is False:
            # Should not co-occur with 'filled'; surface loudly if it does.
            verdict = "partial_legs"
            detail = f"status filled but leg fills differ: {leg_fills}"
    elif status == "partially_filled":
        if atomic is False:
            verdict = "partial_legs"
            detail = f"legs filled unevenly: {leg_fills} -- possible unpaired leg"
        else:
            verdict = "working"
            detail = f"partially filled {filled_qty}/{ordered_qty}, legs balanced"
    elif status == "rejected":
        verdict = "rejected"
        detail = str(order.get("reject_reason") or order.get("reason") or "rejected")
    elif status in _RESTING:
        verdict = "working"
        detail = f"resting ({status}), {filled_qty}/{ordered_qty} filled"
    elif status in ("canceled", "expired", "done_for_day"):
        verdict = "unfilled" if filled_qty == 0 else "working"
        detail = f"{status}, {filled_qty}/{ordered_qty} filled"
    else:
        verdict = "unknown"
        detail = f"unrecognised status {status!r}"

    return FillReport(
        order_id=order_id,
        client_order_id=client_order_id,
        status=status,
        verdict=verdict,
        filled_qty=filled_qty,
        ordered_qty=ordered_qty,
        leg_fills=leg_fills,
        atomic=atomic,
        detail=detail,
    )


def chain_leg_from_quote(quote_snapshot: dict, occ_symbol: str, action: str) -> ContractLeg:
    """Build a ContractLeg from one entry of a live option-chain snapshot.

    Mid is the average of the live bid and ask; if only a last trade is present
    (thin or after hours), it falls back to that so a spec can still be built.
    """
    parsed = parse_occ_symbol(occ_symbol)
    q = quote_snapshot.get("latestQuote") or {}
    bid, ask = q.get("bp"), q.get("ap")
    if bid and ask and bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
    else:
        trade = quote_snapshot.get("latestTrade") or {}
        mid = trade.get("p") or 0.0
    return ContractLeg(
        occ_symbol=occ_symbol,
        right=parsed["right"],
        action=action,
        strike=parsed["strike"],
        expiry=parsed["expiry"],
        mid=float(mid),
    )
