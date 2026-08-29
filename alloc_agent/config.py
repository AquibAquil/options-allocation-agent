"""Central configuration.

Every number the system treats as a policy limit lives here, not scattered
through the modules that enforce it. Values trace back to the PRD sections
named in the comments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    """Minimal .env reader. Real environment variables always win.

    Runs before any os.environ.get below, so values set in .env (provider,
    Groq key, Alpaca keys, Anthropic profile) are visible to every read in this
    module. Empty values are skipped, so a blank ANTHROPIC_API_KEY= line cannot
    shadow an OAuth profile.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


_load_dotenv(os.path.join(REPO_ROOT, ".env"))

UNDERLYING = "QQQ"

# --- Risk budget (PRD 2.2) -------------------------------------------------
# Allocation is a share of a risk budget defined as MAXIMUM LOSS, not capital
# deployed. Buying power is a separate hard gate, never the denominator.


@dataclass(frozen=True)
class RiskBounds:
    # Total max-loss outstanding as a fraction of account equity.
    # Starting point, not computed. Must be verified against live chain
    # premium and observed paper margin before it is fixed (PRD 2.2, 2.7).
    total_budget_frac: float = 0.20
    total_budget_frac_ceiling: float = 0.25

    # No single strategy may hold more than this share of the risk budget.
    per_strategy_max: float = 0.45

    # Below this share, snap to zero rather than hold a token position.
    snap_to_zero_below: float = 0.10

    # Minimum change in allocation (in percentage points, as a fraction)
    # that justifies trading. Hard risk reductions bypass this.
    adjustment_threshold: float = 0.05

    def validate(self) -> None:
        assert 0 < self.total_budget_frac <= self.total_budget_frac_ceiling < 1
        assert 0 < self.snap_to_zero_below < self.per_strategy_max <= 1
        # Three strategies at the cap must still be able to fill the budget.
        assert self.per_strategy_max * 3 >= 1.0, "per-strategy cap cannot span the budget"


RISK = RiskBounds()

# --- Cadence (PRD 2.5) -----------------------------------------------------
# Two cycles per trading day. Times are ET wall-clock intent; the scheduler
# resolves them against Alpaca's market clock and calendar, never a hardcode.
DECISION_TIMES_ET = ("10:00", "14:00")
MARKET_TZ = "America/New_York"

# --- Correlation precompute (PRD 2.4) --------------------------------------
CORRELATION_LOOKBACK_DAYS = 252

# --- Model calls (PRD 2.1) -------------------------------------------------
# The allocator and challenger run on a configurable provider. Default is Groq
# (free tier, Llama 3.3 70B): the Claude subscription does not cover the raw
# API, and this project declines to pay for API credits. The model call is
# isolated behind the ModelClient interface, so the provider is a one-line
# config switch and nothing downstream changes.
#
# PRD 2.1 asks for "temperature zero" for reproducibility. On Groq's Llama
# models temperature IS available and is set to 0, which honours that intent
# more directly than the current Opus models (where the parameter is removed).
# Structured output is still requested; the parsers validate defensively either
# way, so a malformed response holds rather than trades.
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "groq").lower()
MODEL_MAX_TOKENS = 16000

# Anthropic (used only if MODEL_PROVIDER=anthropic and the org has API credits).
ALLOCATOR_MODEL = "claude-opus-5"
CHALLENGER_MODEL = "claude-opus-5"

# Groq (default). Free API key from console.groq.com; no card required.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# --- Paths -----------------------------------------------------------------
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")
CACHE_DIR = os.path.join(ARTIFACT_DIR, "cache")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

# --- Credentials -----------------------------------------------------------
# Read from the environment, seeded from the gitignored .env loaded at the top
# of this module. Never committed, never logged, never printed in a packet.

# Canonical names are the ones the official Alpaca MCP server reads
# (ALPACA_API_KEY / ALPACA_SECRET_KEY), so a single .env drives both this code
# and the MCP server. The longer forms Alpaca's own docs use are accepted as a
# fallback, because two names for one credential is a trap.
ALPACA_KEY_ID = os.environ.get("ALPACA_API_KEY") or os.environ.get(
    "ALPACA_API_KEY_ID", ""
)
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get(
    "ALPACA_API_SECRET_KEY", ""
)
ALPACA_PAPER = True
ALPACA_DATA_BASE = "https://data.alpaca.markets"
ALPACA_TRADING_BASE = "https://paper-api.alpaca.markets"


def have_alpaca_credentials() -> bool:
    return bool(ALPACA_KEY_ID and ALPACA_SECRET_KEY)
