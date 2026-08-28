"""Hard risk gates (PRD 2.2).

Enforced on the allocator's output, before execution, regardless of what was
proposed. The limits are also stated in the input packet as constraints, but
enforcement happens HERE, on output: you cannot bound a proposal that does not
exist yet.

Nothing in this module asks a model anything. Every rule is arithmetic, applied
in a fixed order, and every change it makes is recorded with the rule that
caused it so the cycle log can show what the allocator wanted and what it got.

Order of operations matters and is not arbitrary:

  1. reject malformed output (negative, non-finite, unknown key)
  2. cap each strategy at its maximum share of the budget
  3. snap sub-threshold allocations to zero
  4. scale down proportionally if the total exceeds the budget
  5. re-snap, because scaling can push a strategy under the threshold
  6. apply the adjustment threshold against the CURRENT allocation

Step 6 is last because it compares the finished target against what is
currently held. Doing it earlier would let a proposal that violates a hard
limit survive on the grounds that it was a small change.

The scale-down-only conflict
----------------------------
PRD 2.2 says snap to zero below 10% of the risk budget. PRD 2.3 says the long
strangle is scale-down-only, because it pays in a burst after bleeding and
closing it fully converts a maybe into a certain loss at the worst moment.
For a strategy drifting under 10%, those two rules disagree.

Resolved here in favour of 2.3, which is the more specific rule and the one
carrying a stated reason: a scale-down-only strategy holding a live position
floors at the snap threshold instead of snapping to zero. It still reaches zero
when the allocator explicitly targets zero, so the position remains closable on
a genuine invalidation -- it just cannot be closed by drift.

Set SNAP_TO_ZERO_OVERRIDES_SCALE_DOWN to True to follow 2.2 literally instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import RISK
from .strategies import BY_KEY, KEYS, ExitBehaviour

SNAP_TO_ZERO_OVERRIDES_SCALE_DOWN = False

# Allocations are compared at percentage-point scale, where binary floating
# point cannot represent the endpoints exactly: 0.35 - 0.30 is 0.049999999999,
# which would make a change of exactly the threshold fail to trade. PRD 2.2
# says SMALLER changes do not trade, so the boundary itself must.
_EPS = 1e-9


@dataclass(frozen=True)
class Adjustment:
    strategy_key: str
    rule: str
    before: float
    after: float
    detail: str

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass(frozen=True)
class GateResult:
    """What the allocator proposed, what will actually be traded, and why."""

    proposed: dict[str, float]
    final: dict[str, float]
    current: dict[str, float]
    adjustments: tuple[Adjustment, ...]
    traded_keys: tuple[str, ...]

    @property
    def modified(self) -> bool:
        return bool(self.adjustments)

    def to_dict(self) -> dict:
        return {
            "proposed": self.proposed,
            "final": self.final,
            "current": self.current,
            "traded_keys": list(self.traded_keys),
            "adjustments": [
                {
                    "strategy_key": a.strategy_key,
                    "rule": a.rule,
                    "before": a.before,
                    "after": a.after,
                    "detail": a.detail,
                }
                for a in self.adjustments
            ],
        }


class MalformedAllocation(ValueError):
    """The allocator returned something that cannot be gated, only rejected.

    PRD 2.5: on malformed model output, hold the last valid allocation and do
    not trade. A transient software problem must not cause a portfolio change.
    """


def validate_proposal(proposed: dict[str, float]) -> dict[str, float]:
    missing = set(KEYS) - set(proposed)
    unknown = set(proposed) - set(KEYS)
    if unknown:
        raise MalformedAllocation(f"unknown strategy keys: {sorted(unknown)}")
    if missing:
        raise MalformedAllocation(f"missing strategy keys: {sorted(missing)}")

    out: dict[str, float] = {}
    for key in KEYS:
        value = proposed[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MalformedAllocation(f"{key}: allocation must be a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise MalformedAllocation(f"{key}: allocation is not finite ({value})")
        if value < 0:
            raise MalformedAllocation(f"{key}: allocation is negative ({value})")
        out[key] = value
    return out


def _snap(
    allocations: dict[str, float],
    current: dict[str, float],
    adjustments: list[Adjustment],
    rule: str,
) -> dict[str, float]:
    floor = RISK.snap_to_zero_below
    out = dict(allocations)
    for key, value in allocations.items():
        if value <= 0.0 or value >= floor:
            continue

        scale_down_only = (
            BY_KEY[key].exit_behaviour is ExitBehaviour.SCALE_DOWN_ONLY
            and not SNAP_TO_ZERO_OVERRIDES_SCALE_DOWN
        )
        if scale_down_only and current.get(key, 0.0) > 0.0:
            out[key] = floor
            adjustments.append(
                Adjustment(
                    strategy_key=key,
                    rule="scale_down_only_floor",
                    before=value,
                    after=floor,
                    detail=(
                        f"{BY_KEY[key].name} is scale-down-only and holds a live "
                        f"position; floored at {floor:.0%} of budget rather than "
                        f"snapped to zero (PRD 2.3 over 2.2)"
                    ),
                )
            )
            continue

        out[key] = 0.0
        adjustments.append(
            Adjustment(
                strategy_key=key,
                rule=rule,
                before=value,
                after=0.0,
                detail=f"below the {floor:.0%} minimum; snapped to zero",
            )
        )
    return out


def apply_gates(
    proposed: dict[str, float],
    current: dict[str, float],
    *,
    forced_reductions: frozenset[str] = frozenset(),
) -> GateResult:
    """Enforce every hard limit on a proposal, then decide what actually trades.

    `current` is the live allocation, as a share of the risk budget.
    `forced_reductions` names strategies whose reduction is a hard risk action
    and therefore bypasses the adjustment threshold (PRD 2.2).
    """
    proposed = validate_proposal(proposed)
    current = {key: float(current.get(key, 0.0)) for key in KEYS}
    adjustments: list[Adjustment] = []

    # 2. per-strategy cap
    cap = RISK.per_strategy_max
    allocations = {}
    for key, value in proposed.items():
        if value > cap:
            adjustments.append(
                Adjustment(
                    strategy_key=key,
                    rule="per_strategy_cap",
                    before=value,
                    after=cap,
                    detail=f"exceeds the {cap:.0%} per-strategy maximum",
                )
            )
            allocations[key] = cap
        else:
            allocations[key] = value

    # 3. snap
    allocations = _snap(allocations, current, adjustments, "snap_to_zero")

    # 4. total budget
    total = sum(allocations.values())
    if total > 1.0:
        factor = 1.0 / total
        for key, value in list(allocations.items()):
            if value <= 0.0:
                continue
            scaled = value * factor
            adjustments.append(
                Adjustment(
                    strategy_key=key,
                    rule="total_budget",
                    before=value,
                    after=scaled,
                    detail=(
                        f"total proposed {total:.1%} of budget; scaled down "
                        f"proportionally to fit 100%"
                    ),
                )
            )
            allocations[key] = scaled

        # 5. re-snap. Unreachable under the current bounds -- with a 45% cap
        # and three strategies, s_final < floor requires s < others/9 while
        # others <= 0.90 forces s < floor, which step 3 already snapped. Kept
        # as a guard because the cap and the strategy count are both
        # configurable, and asserted as an invariant in the tests.
        allocations = _snap(allocations, current, adjustments, "snap_to_zero_after_scaling")

    # 6. adjustment threshold, against what is currently held
    threshold = RISK.adjustment_threshold
    traded: list[str] = []
    for key in KEYS:
        target, held = allocations[key], current[key]
        change = target - held
        if abs(change) < threshold - _EPS:
            is_forced = key in forced_reductions and change < 0
            if not is_forced:
                if abs(change) > 0:
                    adjustments.append(
                        Adjustment(
                            strategy_key=key,
                            rule="adjustment_threshold",
                            before=target,
                            after=held,
                            detail=(
                                f"change of {change:+.1%} is below the "
                                f"{threshold:.0%} threshold; holding at {held:.1%}"
                            ),
                        )
                    )
                allocations[key] = held
                continue
        traded.append(key)

    return GateResult(
        proposed=proposed,
        final=allocations,
        current=current,
        adjustments=tuple(adjustments),
        traded_keys=tuple(traded),
    )
