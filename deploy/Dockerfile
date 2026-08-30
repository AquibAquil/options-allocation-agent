# Multi-Strategy Allocation Agent -- unattended runner for a small Linux VPS.
#
# The container runs the autonomous loop (alloc_agent.runner). It drives the
# official Alpaca MCP server, which it spawns itself as a stdio subprocess -- so
# one container is the whole agent, no side-car needed.
#
# Secrets are NOT baked in. They arrive as environment variables at run time
# (ALPACA_API_KEY, ALPACA_SECRET_KEY, GROQ_API_KEY) -- see docker-compose.yml and
# DEPLOY.md.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PROVIDER=groq \
    ALPACA_PAPER_TRADE=true \
    # On Linux the Alpaca MCP server is the console script on PATH, not a venv .exe.
    ALPACA_MCP_EXE=alpaca-mcp-server

WORKDIR /app

# Dependencies first, for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code (tests and dev cruft excluded via .dockerignore).
COPY alloc_agent ./alloc_agent
COPY scripts ./scripts
COPY deploy/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Logs and artifacts persist on the host via volumes (see docker-compose.yml).
RUN mkdir -p /app/logs /app/artifacts/cache
VOLUME ["/app/logs", "/app/artifacts"]

ENTRYPOINT ["./entrypoint.sh"]
# Default: run the hackathon window and stop at Friday's close. Override on
# `docker run` / compose `command` (e.g. --dry-run to rehearse).
CMD ["--stop-at", "2026-09-04T16:00"]
