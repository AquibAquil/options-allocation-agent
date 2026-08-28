"""The verified strategy library (PRD 2.3).

The agent does not compose strategies at runtime (PRD 2.9). It selects among
these three and decides how much risk budget each deserves.

Each entry carries a written thesis. The thesis is the thing the allocator
reconciles evidence against; it is prompt input, not decoration. The
invalidation text matters more than the entry text, because the failure mode
this system exists to avoid is cutting a strategy that is behaving exactly as
its thesis says it should.

Strike and expiry selection is FIXED, not adaptive (PRD 2.3). Adaptivity lives
in the allocator. Fixed selection keeps positions comparable and evidence clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExitBehaviour(str, Enum):
    # Accrues steadily; closing early prices in what is left to earn. Clean to cut.
    CLOSE_FULLY = "close_fully"
    # Pays in a burst after bleeding. Closing fully at the worst moment turns a
    # maybe into a certain loss. Reduce contract count instead.
    SCALE_DOWN_ONLY = "scale_down_only"


class VolExposure(str, Enum):
    SHORT = "short"
    LONG = "long"


@dataclass(frozen=True)
class LegSpec:
    right: str                          # "put" | "call"
    action: str                         # "sell" | "buy"
    target_delta: float | None = None   # absolute delta; None when strike-relative
    strikes_away: int | None = None     # offset from the short strike, in strikes


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    legs: tuple[LegSpec, ...]
    dte_min: int
    dte_max: int
    vol_exposure: VolExposure
    direction: str
    collects_premium: bool
    exit_behaviour: ExitBehaviour
    thesis: str
    invalidation: str
    not_invalidation: str

    @property
    def dte_target(self) -> int:
        return (self.dte_min + self.dte_max) // 2


BULL_PUT_SPREAD = Strategy(
    key="bull_put_spread",
    name="Bull put spread",
    legs=(
        LegSpec(right="put", action="sell", target_delta=0.25),
        LegSpec(right="put", action="buy", strikes_away=-3),
    ),
    dte_min=7,
    dte_max=14,
    vol_exposure=VolExposure.SHORT,
    direction="bullish/neutral",
    collects_premium=True,
    exit_behaviour=ExitBehaviour.CLOSE_FULLY,
    thesis=(
        "Sells downside insurance at a strike the underlying is unlikely to reach "
        "within the holding period, and earns the volatility risk premium plus time "
        "decay for bearing that risk. Works when the trend is intact or merely "
        "directionless, when implied volatility sits above realised volatility by "
        "enough to pay for the risk taken, and when the short strike sits far enough "
        "below spot that ordinary daily movement does not threaten it."
    ),
    invalidation=(
        "Spot approaching or through the short strike; realised volatility rising to "
        "meet or exceed implied, which removes the premium being paid for the risk; a "
        "directional break lower that is large relative to the strike buffer rather "
        "than inside it."
    ),
    not_invalidation=(
        "Mark-to-market losses while spot remains comfortably above the short strike. A "
        "credit spread is expected to show unrealised losses whenever the underlying "
        "moves against it, and to recover them through decay if the strike holds."
    ),
)

BEAR_CALL_SPREAD = Strategy(
    key="bear_call_spread",
    name="Bear call spread",
    legs=(
        LegSpec(right="call", action="sell", target_delta=0.25),
        LegSpec(right="call", action="buy", strikes_away=3),
    ),
    dte_min=7,
    dte_max=14,
    vol_exposure=VolExposure.SHORT,
    direction="bearish/neutral",
    collects_premium=True,
    exit_behaviour=ExitBehaviour.CLOSE_FULLY,
    thesis=(
        "The mirror of the bull put spread. Sells upside at a strike the underlying is "
        "unlikely to reach, earning the volatility risk premium and time decay. Works "
        "when upside momentum is absent or exhausted, when implied volatility exceeds "
        "realised by enough to pay for the risk, and when the short strike sits far "
        "enough above spot to absorb ordinary movement."
    ),
    invalidation=(
        "Spot approaching or through the short strike; a momentum expansion higher that "
        "is large relative to the strike buffer; realised volatility rising to meet "
        "implied."
    ),
    not_invalidation=(
        "Mark-to-market losses during a drift higher that leaves the short strike "
        "intact. Direction moving against the position is the risk being paid for, not "
        "evidence the thesis has failed."
    ),
)

LONG_STRANGLE = Strategy(
    key="long_strangle",
    name="Long strangle",
    legs=(
        LegSpec(right="call", action="buy", target_delta=0.175),
        LegSpec(right="put", action="buy", target_delta=0.175),
    ),
    dte_min=25,
    dte_max=35,
    vol_exposure=VolExposure.LONG,
    direction="neutral",
    collects_premium=False,
    exit_behaviour=ExitBehaviour.SCALE_DOWN_ONLY,
    thesis=(
        "Buys convexity. Pays a known premium for a payoff that is worthless unless the "
        "underlying moves substantially in either direction, or unless implied "
        "volatility expands. Works when protection is cheap relative to its own history "
        "and when conditions capable of producing an expansion still exist. Its entire "
        "payoff profile is a long stretch of small losses punctuated by a large gain. It "
        "is the only strategy in this library positioned for a volatility shock, and it "
        "is the reason the other two do not all lose together."
    ),
    invalidation=(
        "Protection is no longer cheap: implied volatility percentile has risen into the "
        "upper part of its own range, so the position is paying a high price for the "
        "same convexity. Or the conditions that could produce an expansion have "
        "demonstrably resolved. Or time decay has consumed enough of the remaining life "
        "that the position can no longer pay off within its expiry."
    ),
    not_invalidation=(
        "LOSING MONEY. This is the critical case. Bleeding is what this strategy does "
        "while it waits, and a run of losing days is the ordinary state of a long "
        "volatility position in a quiet tape, not evidence against it. Cutting it after "
        "consecutive losses is buying volatility and selling it immediately before it "
        "pays. Anchor the judgement to whether protection is still cheap and whether the "
        "conditions for expansion still exist, never to the P&L line."
    ),
)

LIBRARY: tuple[Strategy, ...] = (BULL_PUT_SPREAD, BEAR_CALL_SPREAD, LONG_STRANGLE)
BY_KEY: dict[str, Strategy] = {s.key: s for s in LIBRARY}
KEYS: tuple[str, ...] = tuple(s.key for s in LIBRARY)


def max_loss_per_contract(
    strategy_key: str,
    *,
    premium: float,
    strike_width: float | None = None,
) -> float:
    """Max loss for one contract, in dollars (PRD 2.2).

    Defined-risk across all three, which is what makes a single risk-budget
    denominator coherent.

    spreads:  (strike width x 100) - premium collected
    strangle: premium paid
    """
    strategy = BY_KEY[strategy_key]
    if strategy.collects_premium:
        if strike_width is None:
            raise ValueError(f"{strategy_key} requires strike_width")
        loss = strike_width * 100.0 - premium
        if loss <= 0:
            raise ValueError(
                f"{strategy_key}: credit {premium} exceeds width {strike_width}x100; "
                "chain data is wrong"
            )
        return loss
    if premium <= 0:
        raise ValueError(f"{strategy_key}: debit must be positive, got {premium}")
    return premium
