"""Allocation delta and rejection rate (PRD 2.8).

The headline metric of the whole experiment: does adaptive AI allocation beat
equal weighting across the same three strategies? Both are reported regardless
of what they show -- a video saying the allocator lost to equal weight and
explaining why is more credible than pretending otherwise.

The allocation delta is coherent because both portfolios trade the SAME three
strategies with the SAME fixed entry and exit timing. The only thing that
differs is the weights, so over any period:

    return_actual = sum_i  w_actual_i  * r_i
    return_equal  = sum_i  (1/n)       * r_i
    delta         = return_actual - return_equal = sum_i (w_actual_i - 1/n) * r_i

where r_i is strategy i's return over the period per unit of risk budget. Because
selection is fixed and deterministic, r_i is knowable for every strategy each
cycle -- from the marked P&L of a held position, or from marking the
fixed-selection shadow of an unheld one. This module does the accounting; the
orchestrator supplies r_i and the weights.

Weights here are shares of the risk budget (max loss), the same denominator the
allocator allocates in, which is what makes the equal-weight benchmark
comparable in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .strategies import KEYS


@dataclass(frozen=True)
class CycleReturn:
    """One cycle's realised per-strategy returns and the weights in force.

    `returns` is per unit of risk budget over the period just ended. `weights`
    is the actual allocation (share of budget) held over that period.
    """

    cycle_id: str
    returns: dict[str, float]
    weights: dict[str, float]

    def actual_return(self) -> float:
        return sum(self.weights.get(k, 0.0) * self.returns.get(k, 0.0) for k in KEYS)

    def equal_weight_return(self) -> float:
        w = 1.0 / len(KEYS)
        return sum(w * self.returns.get(k, 0.0) for k in KEYS)


@dataclass
class AllocationDelta:
    """Compounding equity curves for the actual and equal-weight portfolios.

    Returns compound rather than sum: a portfolio that is up 2% then down 2% is
    not flat, and the allocation delta must reflect that or it overstates a
    volatile strategy's contribution.
    """

    actual_equity: float = 1.0
    equal_equity: float = 1.0
    history: list[dict] = field(default_factory=list)

    def record(self, cycle: CycleReturn) -> dict:
        r_actual = cycle.actual_return()
        r_equal = cycle.equal_weight_return()
        self.actual_equity *= 1.0 + r_actual
        self.equal_equity *= 1.0 + r_equal
        entry = {
            "cycle_id": cycle.cycle_id,
            "actual_return": r_actual,
            "equal_weight_return": r_equal,
            "actual_equity": self.actual_equity,
            "equal_weight_equity": self.equal_equity,
            "cumulative_delta": self.cumulative_delta,
        }
        self.history.append(entry)
        return entry

    @property
    def actual_total_return(self) -> float:
        return self.actual_equity - 1.0

    @property
    def equal_weight_total_return(self) -> float:
        return self.equal_equity - 1.0

    @property
    def cumulative_delta(self) -> float:
        """Actual total return minus equal-weight total return, to date."""
        return self.actual_total_return - self.equal_weight_total_return

    def summary(self) -> dict:
        return {
            "actual_total_return": self.actual_total_return,
            "equal_weight_total_return": self.equal_weight_total_return,
            "allocation_delta": self.cumulative_delta,
            "cycles": len(self.history),
            "beats_equal_weight": self.cumulative_delta > 0,
        }

    def to_dict(self) -> dict:
        return {
            "actual_equity": self.actual_equity,
            "equal_equity": self.equal_equity,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AllocationDelta":
        obj = cls(
            actual_equity=data.get("actual_equity", 1.0),
            equal_equity=data.get("equal_equity", 1.0),
        )
        obj.history = list(data.get("history", []))
        return obj


@dataclass
class RejectionTracker:
    """Share of proposals the challenger blocked or modified (PRD 2.8).

    Reported regardless of what it shows. If the challenger approves nearly
    everything, that is the finding, not something to hide -- one model reviewing
    another tends toward agreement, and the number makes that visible.
    """

    approve: int = 0
    modify: int = 0
    reject: int = 0

    def record(self, verdict: str) -> None:
        verdict = verdict.upper()
        if verdict == "APPROVE":
            self.approve += 1
        elif verdict == "MODIFY":
            self.modify += 1
        elif verdict == "REJECT":
            self.reject += 1
        else:
            raise ValueError(f"unknown verdict {verdict!r}")

    @property
    def total(self) -> int:
        return self.approve + self.modify + self.reject

    @property
    def rejection_rate(self) -> float:
        """Share blocked or modified -- i.e. not rubber-stamped."""
        if self.total == 0:
            return 0.0
        return (self.modify + self.reject) / self.total

    @property
    def approval_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.approve / self.total

    def summary(self) -> dict:
        return {
            "total_reviews": self.total,
            "approve": self.approve,
            "modify": self.modify,
            "reject": self.reject,
            "rejection_rate": self.rejection_rate,
            "approval_rate": self.approval_rate,
        }
