"""Demo-screen data feed (supports PRD 4.3).

The decision cycle writes a verbose record per cycle to logs/cycles.jsonl -- the
full evidence packet, proposal, challenge, gates, sizing, orders, metrics. That
is the right thing for the audit log, but too much for a screen that has to be
read in thirty seconds.

This module distils those records into one compact, stable JSON the demo screen
can consume directly, so the UI parses one tidy file instead of the raw log. It
is pure (a list of cycle records in, a dict out) and tested; the companion
script reads the files and writes artifacts/demo_feed.json.

It never invents anything: every field traces to a cycle record, and simulated
data (shock scenarios) is kept in its own section and clearly flagged, so the
scored paper-equity P&L is never confused with a simulation.
"""

from __future__ import annotations

import datetime as dt

from .strategies import KEYS, LIBRARY

SCHEMA_VERSION = 1


def strategies_meta() -> list[dict]:
    """Static labels for the three strategies (names, structure, one-liners)."""
    out = []
    for s in LIBRARY:
        thesis_first = s.thesis.split(". ")[0].rstrip(".") + "."
        out.append(
            {
                "key": s.key,
                "name": s.name,
                "direction": s.direction,
                "vol_exposure": s.vol_exposure.value,
                "exit_behaviour": s.exit_behaviour.value,
                "dte_band": [s.dte_min, s.dte_max],
                "thesis_short": thesis_first,
            }
        )
    return out


def _market(packet: dict) -> dict:
    m = packet.get("market", {}) if packet else {}
    return {
        "symbol": m.get("symbol"),
        "spot": m.get("spot"),
        "spot_asof": m.get("spot_asof"),
        "realized_vol_21d": (m.get("realized_vol") or {}).get("21d"),
        "implied_vol": m.get("implied_vol"),
        "implied_vol_source": m.get("implied_vol_source"),
        "iv_percentile_252d": m.get("implied_vol_percentile_252d"),
        "iv_rv": m.get("iv_rv"),
    }


def _portfolio(packet: dict) -> dict:
    p = packet.get("portfolio", {}) if packet else {}
    return {
        "equity": p.get("equity"),
        "buying_power": p.get("buying_power"),
        "risk_budget_total": p.get("risk_budget_total"),
        "risk_budget_utilisation": p.get("risk_budget_utilisation"),
        "max_loss_as_frac_of_equity": p.get("max_loss_as_frac_of_equity"),
    }


def _strategy_rows(record: dict) -> list[dict]:
    """One row per strategy: current alloc, final alloc, the change, and P&L.

    This is the strategy table plus the AI-decision arrow (`current -> final`).
    """
    packet = record.get("packet") or {}
    by_key = {s.get("key"): s for s in (packet.get("strategies") or [])}
    final = (record.get("gate_result") or {}).get("final") or {}
    proposed = (record.get("proposal") or {}).get("allocations") or {}
    reasoning = (record.get("proposal") or {}).get("reasoning") or {}

    rows = []
    for meta in strategies_meta():
        key = meta["key"]
        ev = by_key.get(key, {})
        current = ev.get("allocation_frac", 0.0)
        final_alloc = final.get(key, current)
        proposed_alloc = proposed.get(key)
        rows.append(
            {
                "key": key,
                "name": meta["name"],
                "current_alloc": current,
                "proposed_alloc": proposed_alloc,
                "final_alloc": final_alloc,
                "change": (final_alloc - current) if final_alloc is not None else 0.0,
                "contracts": ev.get("contracts", 0),
                "pnl_frac_of_max_loss": ev.get("pnl_frac_of_max_loss"),
                "unrealized_pnl": ev.get("unrealized_pnl"),
                "short_strike_distance_sigma": ev.get("short_strike_distance_sigma"),
                "reasoning": reasoning.get(key),
            }
        )
    return rows


def _orders(record: dict) -> list[dict]:
    out = []
    for o in record.get("orders") or []:
        out.append(
            {
                "strategy": o.get("strategy"),
                "submitted": o.get("submitted", False),
                "dry_run": o.get("dry_run", False),
                "summary": o.get("summary"),
                "fill": o.get("fill"),
                "error": o.get("error"),
            }
        )
    return out


def build_latest(record: dict) -> dict:
    """The core demo screen for one cycle."""
    challenge = record.get("challenge") or {}
    proposal = record.get("proposal") or {}
    gate = record.get("gate_result") or {}
    return {
        "cycle_id": record.get("cycle_id"),
        "asof": record.get("asof"),
        "status": record.get("status"),
        "reason": record.get("reason"),
        "is_dry_run": record.get("status") == "dry_run",
        "market": _market(record.get("packet") or {}),
        "portfolio": _portfolio(record.get("packet") or {}),
        "strategies": _strategy_rows(record),
        "decision": {
            "proposed": proposal.get("allocations"),
            "reasoning": proposal.get("reasoning"),
            "portfolio_rationale": proposal.get("portfolio_rationale"),
            "model": proposal.get("model"),
        },
        "challenge": {
            "verdict": challenge.get("verdict"),
            "critique": challenge.get("critique"),
            "evidence_cited": challenge.get("evidence_cited"),
            "modified_allocations": challenge.get("modified_allocations"),
        },
        "final_allocation": gate.get("final"),
        "effective_source": record.get("effective_source"),
        "gate_adjustments": [a.get("detail") for a in gate.get("adjustments", [])],
        "orders": _orders(record),
        "metrics": record.get("metrics", {}),
    }


def build_series(records: list[dict]) -> list[dict]:
    """The allocation-delta curve over cycles, for a sparkline."""
    series = []
    for r in records:
        this = (r.get("metrics") or {}).get("this_cycle")
        if not this:
            continue
        series.append(
            {
                "cycle_id": this.get("cycle_id") or r.get("cycle_id"),
                "actual_equity": this.get("actual_equity"),
                "equal_weight_equity": this.get("equal_weight_equity"),
                "cumulative_delta": this.get("cumulative_delta"),
            }
        )
    return series


def _cycles_index(records: list[dict]) -> list[dict]:
    return [
        {
            "cycle_id": r.get("cycle_id"),
            "asof": r.get("asof"),
            "status": r.get("status"),
            "verdict": (r.get("challenge") or {}).get("verdict"),
        }
        for r in records
    ]


def build_feed(
    records: list[dict],
    *,
    shocks: dict | None = None,
    correlation: dict | None = None,
    now: dt.datetime | None = None,
) -> dict:
    """Distil cycle records (+ optional shocks, correlation) into the feed.

    `records` is the parsed contents of logs/cycles.jsonl, oldest first. An
    empty list yields a valid "awaiting first cycle" feed the UI can render.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    feed: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(),
        "strategies": strategies_meta(),
    }

    if not records:
        feed["state"] = "awaiting_first_cycle"
        feed["has_data"] = False
        feed["latest"] = None
        feed["series"] = []
        feed["cycles"] = []
    else:
        feed["state"] = "live"
        feed["has_data"] = True
        # The latest cycle that actually produced a decision; fall back to the
        # most recent record (which may be a hold or a market-closed skip).
        decisions = [r for r in records if r.get("proposal")]
        latest = decisions[-1] if decisions else records[-1]
        feed["latest"] = build_latest(latest)
        feed["series"] = build_series(records)
        feed["cycles"] = _cycles_index(records)
        # Headline metrics come from the most recent record that has them.
        for r in reversed(records):
            if r.get("metrics"):
                feed["metrics"] = r["metrics"]
                break

    if correlation is not None:
        feed["correlation"] = {
            "keys": correlation.get("keys"),
            "matrix": correlation.get("matrix"),
        }

    if shocks is not None:
        # Compact, and flagged simulated so it is never mistaken for scored P&L.
        scenarios = []
        for s in shocks.get("scenarios", []):
            scenarios.append(
                {
                    "name": s.get("name"),
                    "regime": s.get("regime"),
                    "expectation": s.get("expectation"),
                    "final_allocation": s.get("final_allocation"),
                    "verdict": (s.get("challenge") or {}).get("verdict"),
                    "error": s.get("error"),
                }
            )
        feed["shock_simulations"] = {"simulated": True, "scenarios": scenarios}

    return feed
