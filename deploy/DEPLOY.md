# Deploying to a VPS

The agent is designed to run unattended for the trading window on a small Linux
VPS (PRD 2.5). A laptop sleep or wifi loss costs a decision cycle, and there are
only ~10, so the recommended move is: develop and place the day-one test from
the laptop, then **migrate to the VPS on day two once the system is stable** —
not day one, not day three.

One container is the whole agent. It runs the autonomous loop and spawns the
official Alpaca MCP server itself as a stdio subprocess; there is no side-car.

## What you need

- A Linux VPS with Docker + the compose plugin (1 vCPU / 1 GB RAM is plenty).
- The three secrets: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `GROQ_API_KEY`.

## Set up

```bash
git clone https://github.com/AquibAquil/options-allocation-agent
cd options-allocation-agent/deploy
```

Create `deploy/.env` with your credentials (same keys as the app's `.env`):

```
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_PAPER_TRADE=true
MODEL_PROVIDER=groq
GROQ_API_KEY=your_groq_key
```

`deploy/.env` is gitignored and is never copied into the image — compose injects
it at run time.

## Rehearse, then run

Always dry-run first on the VPS to confirm connectivity and credentials before
placing anything:

```bash
docker compose run --rm agent --dry-run --once
```

That runs one full cycle — Alpaca MCP evidence, Groq allocator + challenger,
gates, sizing — and places nothing. Expect a `dry_run` status and a planned
order in the log.

Then start the real run, detached:

```bash
docker compose up -d --build
docker compose logs -f            # watch the cycles
```

On first boot the entrypoint precomputes the 252-day correlation matrix into the
`artifacts` volume (it is left untouched on later restarts, so a mid-window
reboot never moves an input the allocator is judged against). It fails fast if a
credential is missing rather than holding every cycle in the dark.

The default command runs the window and stops at Friday's close
(`--stop-at 2026-09-04T16:00`). Override it in `docker-compose.yml` or on the
command line.

## The demo feed (optional, alongside)

To keep `artifacts/demo_feed.json` fresh for the demo screen while the agent
runs:

```bash
docker compose exec agent python scripts/export_demo_feed.py --watch 30
```

## Monitoring

- `docker compose logs -f` — one line per cycle: `cycle <id> -> <status>`.
- `logs/cycles.jsonl` (on the host via the volume) — the full audit record.
- The runner holds the last valid allocation on any failure and continues; a
  crashing cycle is logged and the loop goes on. You should rarely need to
  intervene.

## Stop / restart

```bash
docker compose down          # graceful: SIGTERM, stops between cycles
docker compose up -d         # resume; picks up the next scheduled slot
```

`restart: unless-stopped` means a VPS reboot brings the agent back automatically.

## Building the image by hand

Compose sets the build context to the repo root. To build directly, do the same
from the repo root:

```bash
docker build -f deploy/Dockerfile -t alloc-agent .
```

## Without Docker (systemd alternative)

If you would rather run it directly on the VPS:

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=... GROQ_API_KEY=... MODEL_PROVIDER=groq
export ALPACA_MCP_EXE=alpaca-mcp-server
python scripts/precompute_correlation.py           # once
python -m alloc_agent.runner --stop-at 2026-09-04T16:00
```

A minimal `systemd` unit (`/etc/systemd/system/alloc-agent.service`):

```ini
[Unit]
Description=Multi-Strategy Allocation Agent
After=network-online.target

[Service]
WorkingDirectory=/opt/options-allocation-agent
EnvironmentFile=/opt/options-allocation-agent/.env
Environment=ALPACA_MCP_EXE=alpaca-mcp-server
ExecStart=/opt/options-allocation-agent/.venv/bin/python -m alloc_agent.runner --stop-at 2026-09-04T16:00
Restart=on-failure
KillSignal=SIGTERM
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
```

`EnvironmentFile` reads the same `KEY=value` lines as `.env`.
