"""Day-one empirical test: one small spread (PRD 2.7, 4.2).

Two questions cannot be answered from documentation, only by placing one real
order and watching (PRD 2.7):

  1. Do multi-leg orders fill ATOMICALLY, or leg by leg? A legged fill means a
     moment of unpaired short-leg risk -- the naked exposure a defined-risk
     spread is supposed to preclude.
  2. What MARGIN does paper actually hold for a defined-risk spread? The answer
     decides whether the 20-25% risk budget is realistic. Theory says margin
     equals max loss; this measures it.

This places ONE minimum-size QQQ bull put spread (PRD's own example), the "open
one strategy end to end before touching the allocator" step of PRD 4.2. It uses
the same client_order_id format as the live system, so when the runner starts it
adopts this position as its first bull-put holding -- the test is not wasted.

SAFETY: previews by default and prints the exact order. It places nothing unless
`--place` is given. `--place` IS the confirmation.

    python scripts/day_one_test.py                 # preview only, places nothing
    python scripts/day_one_test.py --place         # actually place (market must be open)
    python scripts/day_one_test.py --place --slippage 0.05   # more aggressive fill
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alloc_agent import config
from alloc_agent.broker_mcp import McpBrokerGateway
from alloc_agent.execution import build_order, interpret_fill
from alloc_agent.gateway import verify_after_submit
from alloc_agent.selection import build_sizing_quote, select_vertical
from alloc_agent.sizing import SizingPlan, verify_chain
from alloc_agent.strategies import BULL_PUT_SPREAD

OUT_PATH = os.path.join(config.ARTIFACT_DIR, "day_one_test.json")


def _account_snapshot(raw: dict) -> dict:
    """The margin-relevant account fields, as floats."""
    def f(key):
        v = raw.get(key)
        return float(v) if v is not None else None

    return {
        "buying_power": f("buying_power"),
        "options_buying_power": f("options_buying_power"),
        "initial_margin": f("initial_margin"),
        "maintenance_margin": f("maintenance_margin"),
        "cash": f("cash"),
        "equity": f("equity"),
    }


def _one_contract_plan(quote) -> SizingPlan:
    """A sizing plan for exactly one contract (minimum size for the test)."""
    max_loss = quote.max_loss_per_contract
    return SizingPlan(
        strategy_key=BULL_PUT_SPREAD.key,
        target_alloc_frac=max_loss / (config.RISK.total_budget_frac * 100_000.0),
        target_max_loss=max_loss,
        contracts=1,
        contracts_current=0,
        contract_delta=1,
        actual_max_loss=max_loss,
        actual_alloc_frac=max_loss / (config.RISK.total_budget_frac * 100_000.0),
        estimated_margin=max_loss,
        action="open",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", action="store_true",
                        help="actually place the order (default: preview only)")
    parser.add_argument("--force", action="store_true",
                        help="place even if chain verification fails (e.g. a wide "
                             "weekend market); use deliberately")
    parser.add_argument("--slippage", type=float, default=0.03,
                        help="credit concession per share to improve fill (default 0.03)")
    parser.add_argument("--poll-seconds", type=float, default=20.0,
                        help="how long to poll for a fill after placing")
    args = parser.parse_args()

    asof = dt.date.today()
    report: dict = {
        "test": "day_one_spread",
        "strategy": BULL_PUT_SPREAD.key,
        "asof": asof.isoformat(),
        "placed": False,
    }

    with McpBrokerGateway(asof=asof) as gw:
        market_open = gw.is_market_open()
        report["market_open"] = market_open
        print(f"market open: {market_open}")

        # 1. Select the spread from the live chain (fixed 25-delta / 3-below).
        exp_gte = (asof + dt.timedelta(days=BULL_PUT_SPREAD.dte_min)).isoformat()
        exp_lte = (asof + dt.timedelta(days=BULL_PUT_SPREAD.dte_max)).isoformat()
        chain = gw.option_chain(
            config.UNDERLYING,
            type="put",
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
        )
        selection = select_vertical(chain, BULL_PUT_SPREAD, asof=asof)
        quote = build_sizing_quote(selection, BULL_PUT_SPREAD)

        # 2. Verify the chain still supports the trade (delta, spread, DTE).
        try:
            verify_chain(quote, BULL_PUT_SPREAD)
            report["chain_verified"] = True
        except Exception as exc:
            report["chain_verified"] = False
            report["chain_rejection"] = str(exc)
            print(f"chain verification failed: {exc}")
            if not args.place:
                print("(preview) would not place; chain not tradeable right now.")

        # 3. Build the one-contract order.
        plan = _one_contract_plan(quote)
        spec = build_order(plan, selection.legs, intent="open", slippage=args.slippage)
        report["theoretical_max_loss_per_contract"] = quote.max_loss_per_contract
        report["order"] = spec.to_mcp_kwargs()
        report["order_summary"] = spec.human_summary()

        print("\n" + spec.human_summary())
        print(f"\ntheoretical max loss / contract: ${quote.max_loss_per_contract:.2f}")

        # 4. Record the account BEFORE.
        before = _account_snapshot(_raw_account(gw))
        report["account_before"] = before
        print(f"buying power before: ${before['buying_power']:,.0f} | "
              f"options bp before: ${before['options_buying_power']:,.0f}")

        if not args.place:
            print("\nPREVIEW ONLY -- nothing placed. Re-run with --place to execute.")
            _write(report)
            return 0

        # Refuse to place an unverified (e.g. weekend-wide) spread unless forced.
        if not report.get("chain_verified") and not args.force:
            print("\nREFUSING to place: chain verification failed and --force not "
                  "given. On an open market this should pass; if it does not, "
                  "inspect the rejection before overriding with --force.")
            _write(report)
            return 1

        if not market_open:
            print("\nWARNING: market is closed. A limit order will rest and may not "
                  "fill; the margin/fill answers need a live fill.")

        # 5. Place, then verify ACTUAL status (never trust the submit response).
        print("\nplacing order ...")
        submit = gw.place_order(spec)
        report["placed"] = True
        report["client_order_id"] = spec.client_order_id
        fill = verify_after_submit(gw, spec, submit)

        # Poll briefly for a resting order to fill.
        deadline = time.time() + args.poll_seconds
        while fill.verdict == "working" and time.time() < deadline:
            time.sleep(3)
            fill = interpret_fill(gw.order_status(client_order_id=spec.client_order_id))

        report["fill"] = {
            "verdict": fill.verdict,
            "status": fill.status,
            "atomic": fill.atomic,
            "leg_imbalance": fill.has_leg_imbalance,
            "leg_fills": list(fill.leg_fills),
            "detail": fill.detail,
        }
        print(f"\nfill: {fill.verdict} ({fill.detail})")
        print(f"ATOMIC FILL: {fill.atomic}   leg imbalance: {fill.has_leg_imbalance}")

        # 6. Record the account AFTER; margin held = the buying-power delta.
        after = _account_snapshot(_raw_account(gw))
        report["account_after"] = after
        if before["buying_power"] is not None and after["buying_power"] is not None:
            bp_delta = before["buying_power"] - after["buying_power"]
            report["buying_power_consumed"] = bp_delta
            report["margin_vs_max_loss"] = (
                bp_delta / quote.max_loss_per_contract
                if quote.max_loss_per_contract else None
            )
            print(f"\nMARGIN HELD (buying-power delta): ${bp_delta:,.2f}")
            print(f"theoretical max loss:            ${quote.max_loss_per_contract:,.2f}")
            print(f"ratio held/theoretical:          {report['margin_vs_max_loss']:.2f}"
                  if report['margin_vs_max_loss'] else "")

    _write(report)
    return 0


def _raw_account(gw) -> dict:
    """The raw account dict from MCP (McpBrokerGateway.account parses a subset)."""
    return gw._b.call("get_account_info")  # noqa: SLF001 -- intentional, this is a probe


def _write(report: dict) -> None:
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
