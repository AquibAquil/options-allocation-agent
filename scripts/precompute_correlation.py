"""Precompute the 3x3 cross-strategy correlation matrix (PRD 2.4).

    python scripts/precompute_correlation.py              # fetch, cache, compute
    python scripts/precompute_correlation.py --from-cache # recompute offline

Writes artifacts/correlation.json, which the evidence packet reads at every
decision cycle. Run once before going live and leave it alone; recomputing
mid-window would move an input the allocator is being judged against.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alloc_agent.config import ARTIFACT_DIR, CORRELATION_LOOKBACK_DAYS, UNDERLYING
from alloc_agent.data import bars as bars_mod
from alloc_agent.evidence import correlation as corr


def print_matrix(keys, matrix, indent=""):
    width = max(len(k) for k in keys)
    print(indent + " " * (width + 2) + "  ".join(f"{k[:8]:>8}" for k in keys))
    for name, row in zip(keys, matrix):
        cells = "  ".join(f"{v:>+8.3f}" for v in row)
        print(f"{indent}{name:<{width}}  {cells}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=UNDERLYING)
    parser.add_argument("--days", type=int, default=CORRELATION_LOOKBACK_DAYS)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--out", default=os.path.join(ARTIFACT_DIR, "correlation.json"))
    args = parser.parse_args()

    try:
        if args.from_cache:
            bars = bars_mod.read_cache(args.symbol)
            print(f"loaded {len(bars)} cached bars")
        else:
            bars = bars_mod.fetch_daily_bars(args.symbol, days=args.days)
            path = bars_mod.write_cache(args.symbol, bars)
            print(f"fetched {len(bars)} bars -> {path}")
    except bars_mod.BarsUnavailable as exc:
        print(f"cannot load bars: {exc}", file=sys.stderr)
        print(
            "set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY, or pass --from-cache "
            "if artifacts/cache already holds a series.",
            file=sys.stderr,
        )
        return 2

    closes = np.array([float(b["c"]) for b in bars], dtype=float)
    artifact = corr.precompute(
        closes,
        sample_start=bars_mod.bar_date(bars[0]),
        sample_end=bars_mod.bar_date(bars[-1]),
    )
    corr.save(artifact, args.out)

    meta = artifact.revaluation_meta
    pw = artifact.piecewise_inputs
    print()
    print(f"{args.symbol}  {artifact.sample_start} -> {artifact.sample_end}")
    print(f"realised vol over sample: {pw['sigma_annual']:.2%}")
    print(
        f"EWMA vol used for pricing: {meta['sigma_min']:.1%} to {meta['sigma_max']:.1%} "
        f"(mean {meta['sigma_mean']:.1%}, lambda={meta['ewma_lambda']})"
    )
    print(
        f"25-delta breach at {pw['spread_dte']} DTE: "
        f"{pw['breach_threshold_down']:+.2%} / {pw['breach_threshold_up']:+.2%}  "
        f"({pw['n_breach_down']} down, {pw['n_breach_up']} up days in sample)"
    )
    print()

    print(f"CORRELATION ({artifact.construction}, {meta['n_days']} days)")
    print_matrix(artifact.keys, artifact.matrix)
    print()

    print("daily P&L as a fraction of max loss:")
    for key in artifact.keys:
        print(
            f"  {key:<18} mean {meta['mean_daily_pnl_frac'][key]:+.4f}   "
            f"sd {meta['std_daily_pnl_frac'][key]:.4f}"
        )
    print()

    check = artifact.shape_check
    if check["passed"]:
        print("shape check PASSED (PRD 2.4 expected shape)")
    else:
        print("shape check FAILED -- construction is wrong, do not ship this input:")
        for failure in check["failures"]:
            print(f"  - {failure}")
    print()

    print("for comparison, PRD 2.4 piecewise construction:")
    print_matrix(artifact.keys, artifact.matrix_piecewise, indent="  ")
    pw_check = artifact.shape_check_piecewise
    print(f"  shape check {'PASSED' if pw_check['passed'] else 'FAILED'}", end="")
    print(f" (spreads {pw_check['bull_put_vs_bear_call']:+.3f})")

    print(f"\nwrote {args.out}")
    return 0 if check["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
