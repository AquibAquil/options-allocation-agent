"""Contract selection from a live option chain (PRD 2.3).

Own code, deterministic. Strike and expiry selection is FIXED, not adaptive:
adaptivity lives in the allocator, and fixed selection keeps positions
comparable and the evidence clean. So there is no optimisation here, only rules:

  vertical spread -- the expiry nearest the target DTE inside the band; the
  short strike whose delta is closest to the target; the long strike a fixed
  number of strikes further out of the money.

  strangle -- the expiry nearest its (longer) target DTE; the call and put
  strikes whose deltas are closest to the target on each side.

Input is the raw snapshot map returned by the get_option_chain MCP tool
(symbol -> snapshot). Nothing here calls MCP; the agent pulls the chain and
passes it in, which keeps selection pure and testable against captured chains.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .execution import ContractLeg, parse_occ_symbol
from .strategies import Strategy


class NoContractFound(ValueError):
    """The chain does not contain a contract meeting the fixed selection rule."""


@dataclass(frozen=True)
class Candidate:
    occ_symbol: str
    right: str
    strike: float
    expiry: str
    dte: int
    delta: float | None
    mid: float
    bid: float | None
    ask: float | None
    implied_vol: float | None

    @property
    def bid_ask_spread(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


def _mid(snapshot: dict) -> float:
    q = snapshot.get("latestQuote") or {}
    bid, ask = q.get("bp"), q.get("ap")
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return float((snapshot.get("latestTrade") or {}).get("p") or 0.0)


def to_candidates(chain: dict, *, asof: dt.date) -> list[Candidate]:
    """Flatten a chain snapshot map into typed candidates with DTE."""
    out: list[Candidate] = []
    for symbol, snap in (chain.get("snapshots") or chain).items():
        try:
            parsed = parse_occ_symbol(symbol)
        except ValueError:
            continue
        q = snap.get("latestQuote") or {}
        greeks = snap.get("greeks") or {}
        expiry = parsed["expiry"]
        out.append(
            Candidate(
                occ_symbol=symbol,
                right=parsed["right"],
                strike=parsed["strike"],
                expiry=expiry,
                dte=(dt.date.fromisoformat(expiry) - asof).days,
                delta=greeks.get("delta"),
                mid=_mid(snap),
                bid=q.get("bp"),
                ask=q.get("ap"),
                implied_vol=snap.get("impliedVolatility"),
            )
        )
    return out


def _pick_expiry(candidates: list[Candidate], strategy: Strategy) -> int:
    """The single expiry nearest the target DTE, inside the strategy's band."""
    in_band = {
        c.dte for c in candidates if strategy.dte_min <= c.dte <= strategy.dte_max
    }
    if not in_band:
        raise NoContractFound(
            f"{strategy.key}: no expiry in the {strategy.dte_min}-{strategy.dte_max} "
            f"DTE band"
        )
    return min(in_band, key=lambda d: abs(d - strategy.dte_target))


def _nearest_delta(
    candidates: list[Candidate], right: str, dte: int, target_abs_delta: float
) -> Candidate:
    pool = [
        c
        for c in candidates
        if c.right == right and c.dte == dte and c.delta is not None and c.mid > 0
    ]
    if not pool:
        raise NoContractFound(f"no {right} with a delta at {dte} DTE")
    return min(pool, key=lambda c: abs(abs(c.delta) - target_abs_delta))


@dataclass(frozen=True)
class Selection:
    """The chosen structure: candidates (rich) plus execution-ready legs.

    The candidates retain delta, bid, ask and IV so the sizing quote can be
    built with the real short-leg delta and bid-ask spread that chain
    verification checks against. The legs carry only what an order needs.
    """

    strategy_key: str
    candidates: tuple[Candidate, ...]
    legs: tuple[ContractLeg, ...]

    @property
    def short_candidate(self) -> Candidate | None:
        return next((c for c, l in zip(self.candidates, self.legs) if l.action == "sell"), None)

    @property
    def widest_bid_ask(self) -> float:
        spreads = [c.bid_ask_spread for c in self.candidates if c.bid_ask_spread is not None]
        return max(spreads) if spreads else 0.0

    @property
    def min_dte(self) -> int:
        return min(c.dte for c in self.candidates)


def _leg(c: Candidate, action: str) -> ContractLeg:
    return ContractLeg(
        c.occ_symbol, c.right, action, c.strike, c.expiry, c.mid, bid=c.bid, ask=c.ask
    )


def select_vertical(
    chain: dict, strategy: Strategy, *, asof: dt.date, strike_spacing: float = 1.0
) -> Selection:
    """Short leg nearest the target delta; long leg a fixed width further OTM.

    Raises NoContractFound if the fixed rule cannot be satisfied -- the agent
    then holds rather than improvising a different structure (PRD 2.5).
    """
    short_spec = next(l for l in strategy.legs if l.action == "sell")
    long_spec = next(l for l in strategy.legs if l.action == "buy")
    right = short_spec.right

    candidates = to_candidates(chain, asof=asof)
    dte = _pick_expiry(candidates, strategy)
    short = _nearest_delta(candidates, right, dte, short_spec.target_delta)

    # Long strike is a fixed number of strikes further out of the money.
    steps = abs(long_spec.strikes_away)
    direction = -1.0 if right == "put" else 1.0
    target_long_strike = short.strike + direction * steps * strike_spacing

    long_pool = [c for c in candidates if c.right == right and c.dte == dte and c.mid > 0]
    long = min(long_pool, key=lambda c: abs(c.strike - target_long_strike))
    if long.occ_symbol == short.occ_symbol:
        raise NoContractFound(
            f"{strategy.key}: long strike collapsed onto the short at {short.strike}"
        )

    return Selection(
        strategy_key=strategy.key,
        candidates=(short, long),
        legs=(_leg(short, "sell"), _leg(long, "buy")),
    )


def select_strangle(chain: dict, strategy: Strategy, *, asof: dt.date) -> Selection:
    """Long call and long put, each nearest its target delta at one expiry."""
    call_spec = next(l for l in strategy.legs if l.right == "call")
    put_spec = next(l for l in strategy.legs if l.right == "put")

    candidates = to_candidates(chain, asof=asof)
    dte = _pick_expiry(candidates, strategy)
    call = _nearest_delta(candidates, "call", dte, call_spec.target_delta)
    put = _nearest_delta(candidates, "put", dte, put_spec.target_delta)

    return Selection(
        strategy_key=strategy.key,
        candidates=(call, put),
        legs=(_leg(call, "buy"), _leg(put, "buy")),
    )


def select(chain: dict, strategy: Strategy, *, asof: dt.date, strike_spacing: float = 1.0) -> Selection:
    """Dispatch to the right selector by strategy shape."""
    if strategy.collects_premium:
        return select_vertical(chain, strategy, asof=asof, strike_spacing=strike_spacing)
    return select_strangle(chain, strategy, asof=asof)


def build_sizing_quote(selection: Selection, strategy: Strategy):
    """Build the sizing ChainQuote from a selection, with real deltas.

    Uses the candidates' live deltas and the widest leg bid-ask, so the sizing
    layer's chain verification (delta drift, spread width, defined-risk maths)
    runs against real numbers rather than placeholders.
    """
    from .sizing import ChainQuote

    if strategy.collects_premium:
        short = next(c for c, l in zip(selection.candidates, selection.legs) if l.action == "sell")
        long = next(c for c, l in zip(selection.candidates, selection.legs) if l.action == "buy")
        width = abs(short.strike - long.strike)
        premium = (short.mid - long.mid) * 100.0
        max_loss = width * 100.0 - premium
        short_delta = short.delta
    else:
        premium = sum(c.mid for c in selection.candidates) * 100.0
        max_loss = premium
        width = None
        short_delta = None

    return ChainQuote(
        strategy_key=strategy.key,
        max_loss_per_contract=max_loss,
        premium_per_contract=premium,
        strike_width=width,
        min_dte=selection.min_dte,
        short_leg_delta=short_delta,
        bid_ask_spread=selection.widest_bid_ask,
    )
