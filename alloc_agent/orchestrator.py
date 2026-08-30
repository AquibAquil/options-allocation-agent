"""The decision cycle (PRD 2.5, 2.8).

One cycle: gather evidence, ask the allocator, let the challenger attack it,
resolve the verdict, enforce the hard gates, size, and execute -- logging every
step. Runs twice a trading day on Alpaca's clock, never a hardcoded time.

The organising principle of this module is PRD 2.5's failure rule: HOLD THE LAST
VALID ALLOCATION. A model call that fails or returns malformed output, a broker
call that errors, an order whose status cannot be confirmed -- none of these may
move the portfolio. A transient software problem must not cause a trade. Every
stage is wrapped so that the fallback is always "do nothing and log why".

The cycle is pure given a BrokerGateway, an Allocator, and a Challenger, so the
whole flow is tested offline with fakes. The only live seam is the gateway.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field

from . import config, risk
from .allocator import Allocator, AllocationProposal
from .benchmark import AllocationDelta, CycleReturn, RejectionTracker
from .shadow import ShadowBook
from .challenger import Challenger, ChallengeResult, resolve_effective_allocation
from .evidence import packet as pk
from .gateway import BrokerError, BrokerGateway, verify_after_submit
from .llm import ModelUnavailable
from .selection import NoContractFound, build_sizing_quote, select
from .sizing import ChainRejected, size_strategy
from .strategies import BY_KEY, KEYS
from .execution import build_order

# Cycle outcomes. "traded" and "held" are both successes; "held" means the
# system correctly decided (or was forced) not to move.
TRADED = "traded"
HELD_NO_CHANGE = "held_no_change"
HELD_ON_FAILURE = "held_on_failure"
SKIPPED_MARKET_CLOSED = "skipped_market_closed"
DRY_RUN = "dry_run"


@dataclass
class CycleResult:
    cycle_id: str
    asof: str
    status: str
    reason: str = ""
    packet: dict | None = None
    proposal: dict | None = None
    challenge: dict | None = None
    effective_allocation: dict | None = None
    effective_source: str = ""
    gate_result: dict | None = None
    sizing: dict = field(default_factory=dict)
    orders: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def held(self) -> bool:
        return self.status in (HELD_NO_CHANGE, HELD_ON_FAILURE, SKIPPED_MARKET_CLOSED)

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionCycle:
    """Holds the cross-cycle state: last valid allocation, benchmark, logs."""

    def __init__(
        self,
        allocator: Allocator,
        challenger: Challenger,
        *,
        correlation: dict,
        symbol: str = config.UNDERLYING,
        log_dir: str = config.LOG_DIR,
    ):
        self.allocator = allocator
        self.challenger = challenger
        self.correlation = correlation
        self.symbol = symbol
        self.log_dir = log_dir

        # Cross-cycle state.
        self.last_valid_allocation: dict[str, float] = {k: 0.0 for k in KEYS}
        self.delta = AllocationDelta()
        self.rejections = RejectionTracker()
        self.shadow = ShadowBook()

    # -- the cycle ----------------------------------------------------------

    def run_cycle(
        self,
        gateway: BrokerGateway,
        *,
        cycle_id: str,
        asof: dt.date | None = None,
        dry_run: bool = False,
    ) -> CycleResult:
        """Run one decision cycle.

        dry_run drives the entire pipeline with live components -- evidence,
        allocator, challenger, gates, sizing against the live chain -- but never
        places an order: it records what it WOULD trade. It also proceeds when
        the market is closed, so the full system can be validated and demoed off
        hours. Nothing it does can move the account.
        """
        asof = asof or dt.date.today()
        result = CycleResult(cycle_id=cycle_id, asof=asof.isoformat(), status=HELD_ON_FAILURE)

        # 0. Trade only when the market is open, on the broker's clock. A dry run
        # skips this gate on purpose -- it places nothing regardless.
        if not dry_run:
            try:
                if not gateway.is_market_open():
                    result.status = SKIPPED_MARKET_CLOSED
                    result.reason = "market closed"
                    return self._finish(result)
            except BrokerError as exc:
                result.reason = f"clock check failed: {exc}"
                return self._finish(result)

        # 1. Evidence. Any failure here holds.
        try:
            packet = self._gather_evidence(gateway, cycle_id=cycle_id, asof=asof)
        except (BrokerError, ValueError) as exc:
            result.reason = f"evidence gathering failed: {exc}"
            return self._finish(result)
        result.packet = packet.to_dict()

        current = {se.key: se.allocation_frac for se in packet.strategies}

        # 2. Allocator. A failed or malformed call holds (PRD 2.5).
        try:
            proposal = self.allocator.propose(packet)
        except ModelUnavailable as exc:
            result.reason = f"allocator unavailable: {exc}"
            return self._finish(result)
        result.proposal = proposal.to_dict()

        # 3. Challenger. Same rule.
        try:
            challenge = self.challenger.review(packet, proposal)
        except ModelUnavailable as exc:
            result.reason = f"challenger unavailable: {exc}"
            return self._finish(result)
        result.challenge = challenge.to_dict()
        self.rejections.record(challenge.verdict)

        # 4. Resolve verdict -> effective allocation.
        effective, source = resolve_effective_allocation(proposal, challenge, current)
        result.effective_allocation = effective
        result.effective_source = source

        # 5. Hard risk gates, on the resolved output.
        forced = self._forced_reductions(packet)
        gated = risk.apply_gates(effective, current, forced_reductions=forced)
        result.gate_result = gated.to_dict()

        # 6. Size and 7. execute the strategies that trade.
        result.sizing, result.orders = self._size_and_execute(
            gateway, packet, gated, asof=asof, dry_run=dry_run
        )

        traded = any(o.get("submitted") for o in result.orders)
        would_trade = any(o.get("dry_run") for o in result.orders)
        if dry_run:
            result.status = DRY_RUN
            result.reason = f"dry run: {len(result.orders)} order(s) planned, none placed"
        elif traded:
            result.status = TRADED
        elif gated.traded_keys:
            result.status = HELD_ON_FAILURE
            result.reason = result.reason or "sizing/execution produced no fills"
        else:
            result.status = HELD_NO_CHANGE
            result.reason = "no change exceeded the adjustment threshold"

        # A cycle that reached a valid gated allocation updates the anchor, even
        # if nothing traded -- that IS the current valid target.
        self.last_valid_allocation = dict(gated.final)
        return self._finish(result, packet=packet)

    # -- stages -------------------------------------------------------------

    def _gather_evidence(
        self, gateway: BrokerGateway, *, cycle_id: str, asof: dt.date
    ) -> pk.EvidencePacket:
        from .data import vol_index

        account = gateway.account()
        positions = gateway.positions()
        bars = gateway.daily_bars(self.symbol, days=config.CORRELATION_LOOKBACK_DAYS)
        vxn = vol_index.load("VXN", refresh=True)

        return pk.build_packet(
            cycle_id=cycle_id,
            symbol=self.symbol,
            bars=bars,
            vol_index_rows=vxn,
            positions=positions,
            account=account,
            correlation=self.correlation,
            asof=asof,
        )

    def _forced_reductions(self, packet: pk.EvidencePacket) -> frozenset[str]:
        """Strategies whose short strike has been breached are hard risk
        reductions and bypass the adjustment threshold (PRD 2.2).

        Uses only evidence the packet already computed -- the nearest short
        strike's distance in sigma. A strike at or through spot is a breach.
        """
        forced = set()
        for se in packet.strategies:
            if se.contracts == 0:
                continue
            d = se.short_strike_distance_sigma
            if d is not None and abs(d) < 0.25:
                forced.add(se.key)
        return frozenset(forced)

    def _size_and_execute(
        self,
        gateway: BrokerGateway,
        packet: pk.EvidencePacket,
        gated,
        *,
        asof: dt.date,
        dry_run: bool = False,
    ) -> tuple[dict, list]:
        sizing_out: dict = {}
        orders: list = []
        budget = packet.portfolio.risk_budget_total
        current_contracts = {
            se.key: se.contracts for se in packet.strategies
        }

        for key in gated.traded_keys:
            target = gated.final[key]
            strategy = BY_KEY[key]
            entry: dict = {"strategy": key, "target_alloc": target}

            # Pull the chain, narrowed to the strategy's expiry window so the
            # target-DTE contracts are actually in the response (an unfiltered
            # chain is truncated by the server's limit and can miss them), then
            # apply fixed selection.
            exp_gte = (asof + dt.timedelta(days=strategy.dte_min)).isoformat()
            exp_lte = (asof + dt.timedelta(days=strategy.dte_max)).isoformat()
            try:
                chain = gateway.option_chain(
                    self.symbol,
                    expiration_date_gte=exp_gte,
                    expiration_date_lte=exp_lte,
                )
                selection = select(chain, strategy, asof=asof)
                quote = build_sizing_quote(selection, strategy)
                plan = size_strategy(
                    key,
                    target_alloc_frac=target,
                    risk_budget_total=budget,
                    quote=quote,
                    contracts_current=current_contracts.get(key, 0),
                    buying_power=packet.portfolio.buying_power,
                )
            except (BrokerError, NoContractFound, ChainRejected, ValueError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                sizing_out[key] = entry
                continue

            entry["plan"] = plan.to_dict()
            sizing_out[key] = entry

            if plan.is_blocked or plan.contract_delta == 0:
                continue

            # Build the order. A dry run stops here, recording the intended
            # trade without placing it.
            try:
                spec = build_order(plan, selection.legs, intent=plan.action)
            except ValueError as exc:
                orders.append(
                    {"strategy": key, "submitted": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            if dry_run:
                orders.append(
                    {
                        "strategy": key,
                        "submitted": False,
                        "dry_run": True,
                        "client_order_id": spec.client_order_id,
                        "spec": spec.to_mcp_kwargs(),
                        "summary": spec.human_summary(),
                    }
                )
                continue

            # Place the order; verify actual status after.
            try:
                submit = gateway.place_order(spec)
                report = verify_after_submit(gateway, spec, submit)
                orders.append(
                    {
                        "strategy": key,
                        "submitted": True,
                        "client_order_id": spec.client_order_id,
                        "spec": spec.to_mcp_kwargs(),
                        "fill": {
                            "verdict": report.verdict,
                            "status": report.status,
                            "atomic": report.atomic,
                            "leg_imbalance": report.has_leg_imbalance,
                            "detail": report.detail,
                        },
                    }
                )
            except (BrokerError, ValueError) as exc:
                orders.append(
                    {"strategy": key, "submitted": False, "error": f"{type(exc).__name__}: {exc}"}
                )

        return sizing_out, orders

    # -- metrics and logging -----------------------------------------------

    def _record_metrics(self, packet: pk.EvidencePacket) -> dict:
        """Update the allocation delta and rejection rate for this cycle.

        Per-strategy return r_i comes from the shadow book: a fixed-selection
        structure per strategy, held across cycles and Black-Scholes revalued, so
        r_i exists for every strategy whether or not the real portfolio holds it.
        Both the actual and equal-weight portfolios are evaluated on this common
        r_i, so the delta isolates the weight decision (PRD 2.8) and no longer
        favours the AI merely for declining to fund a losing strategy (PRD 3).
        Weights are the real allocation shares.
        """
        annual_vol = (
            packet.market.realized_vol.get("21d")
            or packet.market.realized_vol.get("10d")
            or packet.market.implied_vol
        )
        returns = self.shadow.mark(
            spot=packet.market.spot,
            annual_vol=annual_vol,
            asof=dt.date.fromisoformat(packet.asof),
        )
        weights = {se.key: se.allocation_frac for se in packet.strategies}

        cycle_return = CycleReturn(cycle_id=packet.cycle_id, returns=returns, weights=weights)
        delta_entry = self.delta.record(cycle_return)
        return {
            "this_cycle": delta_entry,
            "allocation_delta": self.delta.summary(),
            "rejection": self.rejections.summary(),
        }

    def _finish(self, result: CycleResult, packet: pk.EvidencePacket | None = None) -> CycleResult:
        if packet is not None:
            result.metrics = self._record_metrics(packet)
        else:
            result.metrics = {
                "allocation_delta": self.delta.summary(),
                "rejection": self.rejections.summary(),
            }
        self._write_log(result)
        return result

    def _write_log(self, result: CycleResult) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "cycles.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), default=str) + "\n")
        return path
