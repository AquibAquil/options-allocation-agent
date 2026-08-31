"""Shadow book for the allocation-delta benchmark (PRD 2.8; fixes the PRD 3 bias).

The allocation delta compares the AI's weighting against equal weight on the
SAME three strategies (PRD 2.8): both portfolios are evaluated on a common
per-strategy return series r_i, so the delta isolates the weight decision. For
that to be honest, r_i must exist for every strategy every cycle -- including one
the AI is not holding, because equal weight would be.

Taking r_i = 0 for an unheld strategy (the earlier shortcut) biases the delta
toward the AI: when it declines to fund a strategy that loses, equal weight wears
the loss and the AI does not, a positive delta earned for a trivial reason
(exactly the risk called out in PRD Part 3).

This module removes that. It keeps a shadow position per strategy -- the fixed
selection (PRD 2.3), held across cycles so time decay accrues, which is the
spreads' whole return mechanism -- and marks it each cycle by Black-Scholes
revaluation from the underlying and its volatility. Same methodology as the
correlation precompute (evidence/bs.py), so nothing new is assumed. It needs no
option chain: spot and vol are already in the evidence packet.

These are MODELLED returns for an analytical metric. Paper equity remains the
only scored P&L (PRD 1.5); the allocation delta is a separate, honest read on
whether the weighting beat equal weight.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .evidence import bs
from .strategies import BY_KEY, KEYS

STRIKE_SPACING = 1.0
SPREAD_WIDTH_STRIKES = 3


def _round_strike(k: float) -> float:
    return round(k / STRIKE_SPACING) * STRIKE_SPACING


@dataclass
class ShadowPosition:
    """A fixed-selection structure, opened once and marked forward."""

    strategy_key: str
    opened_asof: dt.date
    dte_at_open: int
    # spread: (short_k, long_k, right); strangle: (call_k, put_k)
    strikes: tuple[float, ...]
    right: str | None
    entry: float          # per share: credit collected (spread) or debit paid (strangle)
    max_loss: float       # per share
    last_pnl: float = 0.0  # per share, mark-to-market P&L at the previous mark

    def dte_now(self, asof: dt.date) -> int:
        return self.dte_at_open - (asof - self.opened_asof).days


def _open_spread(key: str, spot: float, vol: float, asof: dt.date) -> ShadowPosition:
    strat = BY_KEY[key]
    right = next(l.right for l in strat.legs if l.action == "sell")
    dte = strat.dte_target
    t = dte / bs.TRADING_DAYS
    short_k = _round_strike(bs.strike_for_delta(spot, vol, t, right, 0.25))
    offset = -SPREAD_WIDTH_STRIKES if right == "put" else SPREAD_WIDTH_STRIKES
    long_k = short_k + offset * STRIKE_SPACING
    width = SPREAD_WIDTH_STRIKES * STRIKE_SPACING
    credit = bs.price(spot, short_k, vol, t, right) - bs.price(spot, long_k, vol, t, right)
    credit = max(credit, 0.01)
    max_loss = max(width - credit, 0.01)
    return ShadowPosition(key, asof, dte, (short_k, long_k), right, credit, max_loss)


def _open_strangle(key: str, spot: float, vol: float, asof: dt.date) -> ShadowPosition:
    strat = BY_KEY[key]
    dte = strat.dte_target
    t = dte / bs.TRADING_DAYS
    call_k = _round_strike(bs.strike_for_delta(spot, vol, t, "call", 0.175))
    put_k = _round_strike(bs.strike_for_delta(spot, vol, t, "put", 0.175))
    debit = bs.price(spot, call_k, vol, t, "call") + bs.price(spot, put_k, vol, t, "put")
    debit = max(debit, 0.01)
    return ShadowPosition(key, asof, dte, (call_k, put_k), None, debit, debit)


def _open(key: str, spot: float, vol: float, asof: dt.date) -> ShadowPosition:
    return (_open_spread if BY_KEY[key].collects_premium else _open_strangle)(key, spot, vol, asof)


def _mark_pnl(pos: ShadowPosition, spot: float, vol: float, asof: dt.date) -> float:
    """Mark-to-market P&L per share, floored at the defined risk."""
    dte = pos.dte_now(asof)
    t = max(dte, 0) / bs.TRADING_DAYS
    if BY_KEY[pos.strategy_key].collects_premium:
        short_k, long_k = pos.strikes
        cost = bs.price(spot, short_k, vol, t, pos.right) - bs.price(spot, long_k, vol, t, pos.right)
        pnl = pos.entry - cost               # collected credit, owe current cost
    else:
        call_k, put_k = pos.strikes
        value = bs.price(spot, call_k, vol, t, "call") + bs.price(spot, put_k, vol, t, "put")
        pnl = value - pos.entry              # paid debit, now worth value
    return max(pnl, -pos.max_loss)


class ShadowBook:
    """Per-strategy shadow positions, marked each cycle to yield r_i for all."""

    def __init__(self) -> None:
        self.positions: dict[str, ShadowPosition] = {}

    def to_dict(self) -> dict:
        return {
            key: {
                "strategy_key": p.strategy_key,
                "opened_asof": p.opened_asof.isoformat(),
                "dte_at_open": p.dte_at_open,
                "strikes": list(p.strikes),
                "right": p.right,
                "entry": p.entry,
                "max_loss": p.max_loss,
                "last_pnl": p.last_pnl,
            }
            for key, p in self.positions.items()
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ShadowBook":
        book = cls()
        for key, v in (data or {}).items():
            book.positions[key] = ShadowPosition(
                strategy_key=v["strategy_key"],
                opened_asof=dt.date.fromisoformat(v["opened_asof"]),
                dte_at_open=v["dte_at_open"],
                strikes=tuple(v["strikes"]),
                right=v.get("right"),
                entry=v["entry"],
                max_loss=v["max_loss"],
                last_pnl=v.get("last_pnl", 0.0),
            )
        return book

    def mark(self, *, spot: float, annual_vol: float, asof: dt.date) -> dict[str, float]:
        """Return each strategy's per-unit-max-loss return since the last mark.

        Opens a position the first time it sees a strategy (return 0 that cycle),
        rolls it to a fresh fixed selection once it decays past the strategy's DTE
        band, and otherwise revalues it. Volatility is floored so a degenerate
        input cannot blow up the pricer.
        """
        vol = max(annual_vol, 0.02)
        returns: dict[str, float] = {}
        for key in KEYS:
            pos = self.positions.get(key)
            if pos is None:
                self.positions[key] = _open(key, spot, vol, asof)
                returns[key] = 0.0
                continue

            # Roll when the structure has decayed out of its DTE band.
            if pos.dte_now(asof) < BY_KEY[key].dte_min:
                final = _mark_pnl(pos, spot, vol, asof)
                returns[key] = (final - pos.last_pnl) / pos.max_loss
                self.positions[key] = _open(key, spot, vol, asof)
                continue

            pnl = _mark_pnl(pos, spot, vol, asof)
            returns[key] = (pnl - pos.last_pnl) / pos.max_loss
            pos.last_pnl = pnl
        return returns
