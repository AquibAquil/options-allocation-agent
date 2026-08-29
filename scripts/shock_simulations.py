"""Run the shock simulations and write the deliverable (PRD 4.4).

    python scripts/shock_simulations.py            # live models
    python scripts/shock_simulations.py --list     # just list the scenarios

Feeds each synthetic regime through the REAL allocator, challenger, and risk
gates, and writes artifacts/shock_simulations.json plus a readable report. This
is SIMULATION, clearly labelled -- paper equity remains the only scored P&L.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alloc_agent import config
from alloc_agent.allocator import Allocator
from alloc_agent.challenger import Challenger
from alloc_agent.llm import make_client
from alloc_agent.shocks import all_scenarios, run_scenario
from alloc_agent.strategies import KEYS

OUT_JSON = os.path.join(config.ARTIFACT_DIR, "shock_simulations.json")
OUT_REPORT = os.path.join(config.ARTIFACT_DIR, "shock_simulations.md")


def _fmt_alloc(alloc: dict) -> str:
    return "  ".join(f"{k.split('_')[0]:>7} {alloc.get(k, 0.0):>4.0%}" for k in KEYS)


def render_report(records: list[dict]) -> str:
    lines = [
        "# Shock Simulations",
        "",
        "**These are SIMULATIONS, not scored P&L.** Each regime below is a "
        "synthetic evidence packet -- crafted market conditions the four-day "
        "live window will not produce -- fed through the real allocator, "
        "challenger, and risk gates. Paper account equity remains the only P&L "
        "record. The purpose (PRD 4.4) is to exercise the allocator's judgement "
        "where the quiet live tape cannot.",
        "",
        "The `expectation` line is what good judgement looks like, in plain "
        "words. It is context for the reader, not a pass/fail assertion: model "
        "judgement is not deterministic, and this shows what the allocator "
        "actually did.",
        "",
    ]
    for r in records:
        lines.append(f"## {r['regime']}")
        lines.append("")
        lines.append(r["description"])
        lines.append("")
        lines.append(f"**Expectation:** {r['expectation']}")
        lines.append("")
        if r.get("error"):
            lines.append(f"**Result:** model unavailable -- {r['error']} (held).")
            lines.append("")
            continue
        lines.append(f"**Held before:** `{_fmt_alloc(r['current_allocation'])}`")
        lines.append("")
        lines.append(f"**Allocator proposed:** `{_fmt_alloc(r['proposal']['allocations'])}`")
        lines.append("")
        for k in KEYS:
            reason = r["proposal"]["reasoning"].get(k, "")
            lines.append(f"- _{k}_: {reason}")
        lines.append("")
        ch = r["challenge"]
        lines.append(f"**Challenger:** {ch['verdict']} -- {ch['critique']}")
        lines.append("")
        lines.append(f"**Final after gates ({r['effective_source']}):** "
                     f"`{_fmt_alloc(r['final_allocation'])}`")
        if r["gate_adjustments"]:
            lines.append("")
            lines.append(f"Gate adjustments: {', '.join(r['gate_adjustments'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = parser.parse_args()

    scenarios = all_scenarios()
    if args.list:
        for s in scenarios:
            print(f"{s.name:38} {s.regime}")
        return 0

    allocator = Allocator(make_client())
    challenger = Challenger(make_client())

    records = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario.regime} ...", flush=True)
        record = run_scenario(scenario, allocator, challenger)
        if record.get("error"):
            print(f"    ! {record['error']}")
        else:
            print(f"    allocator {_fmt_alloc(record['proposal']['allocations'])}"
                  f"  | challenger {record['challenge']['verdict']}"
                  f"  | final {_fmt_alloc(record['final_allocation'])}")
        records.append(record)

    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"simulated": True, "scenarios": records}, fh, indent=2, default=str)
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write(render_report(records))

    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
