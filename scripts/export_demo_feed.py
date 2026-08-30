"""Export the compact demo-screen feed (supports PRD 4.3).

    python scripts/export_demo_feed.py            # logs -> artifacts/demo_feed.json
    python scripts/export_demo_feed.py --watch 30 # re-export every 30 seconds

Reads logs/cycles.jsonl (plus the correlation matrix and the shock simulations
if present) and writes artifacts/demo_feed.json -- one tidy file the demo screen
reads instead of the verbose log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alloc_agent import config
from alloc_agent.demo_feed import build_feed

CYCLES_LOG = os.path.join(config.LOG_DIR, "cycles.jsonl")
CORR_PATH = os.path.join(config.ARTIFACT_DIR, "correlation.json")
SHOCKS_PATH = os.path.join(config.ARTIFACT_DIR, "shock_simulations.json")
OUT_PATH = os.path.join(config.ARTIFACT_DIR, "demo_feed.json")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially written last line during a live run; skip it.
                continue
    return records


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def export_once() -> dict:
    records = _read_jsonl(CYCLES_LOG)
    feed = build_feed(
        records,
        shocks=_read_json(SHOCKS_PATH),
        correlation=_read_json(CORR_PATH),
    )
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    # Write atomically so a reader never sees a half-written file.
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(feed, fh, indent=2, default=str)
    os.replace(tmp, OUT_PATH)
    return feed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=float, default=0.0,
                        help="re-export every N seconds (0 = once)")
    args = parser.parse_args()

    while True:
        feed = export_once()
        n = len(feed.get("cycles", []))
        state = feed.get("state")
        print(f"wrote {OUT_PATH}  (state={state}, {n} cycle(s))")
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
