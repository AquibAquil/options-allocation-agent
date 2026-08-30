#!/usr/bin/env bash
# Container entrypoint: ensure the correlation matrix exists, then run the agent.
#
# The correlation precompute (PRD 2.4) is a one-time input the runner reads every
# cycle. It is regenerated here if absent -- fresh 252-day pull from Alpaca -- so
# a clean volume self-heals on first boot. It is NOT recomputed if present, so a
# restart mid-window does not move an input the allocator is judged against.
set -euo pipefail

mkdir -p /app/logs /app/artifacts/cache

if [ ! -f /app/artifacts/correlation.json ]; then
  echo "[entrypoint] correlation.json missing -- precomputing from Alpaca ..."
  if ! python scripts/precompute_correlation.py; then
    echo "[entrypoint] FATAL: correlation precompute failed (check Alpaca creds)" >&2
    exit 1
  fi
else
  echo "[entrypoint] correlation.json present -- leaving it unchanged"
fi

# Fail fast on missing credentials rather than holding every cycle in the dark.
: "${ALPACA_API_KEY:?ALPACA_API_KEY is not set}"
: "${ALPACA_SECRET_KEY:?ALPACA_SECRET_KEY is not set}"
if [ "${MODEL_PROVIDER:-groq}" = "groq" ]; then
  : "${GROQ_API_KEY:?GROQ_API_KEY is not set (MODEL_PROVIDER=groq)}"
fi

echo "[entrypoint] starting runner: python -m alloc_agent.runner $*"
exec python -m alloc_agent.runner "$@"
