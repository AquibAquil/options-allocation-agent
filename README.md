# Multi-Strategy Allocation Agent

An autonomous agent that runs three QQQ options strategies at once and continuously
decides which of them deserve capital and how much.

The agent does not invent strategies. It is given a library of verified strategies,
each with a written thesis describing the conditions under which it works. Its job is
to judge, at runtime, whether those conditions currently hold and whether recent
performance is consistent with the thesis or evidence against it.

The live experiment: **does adaptive AI allocation beat equal weighting across the
same three strategies?** That gives a hypothesis, a benchmark, and a result that is
interesting whichever way it lands. The allocation delta is reported regardless of
sign.

Built for the Alpaca AI Trading Agents Hackathon, 28 Aug to 4 Sep 2026.

---

## Status

| Component | State |
| --- | --- |
| Strategy library and theses | done |
| Correlation precompute (PRD 2.4) | done, needs a live 252-day pull |
| Daily bars + cache | done, blocked on Alpaca credentials |
| IV reference series (CBOE VXN) | done, live |
| Evidence metrics | done |
| Evidence packet assembly | done, awaiting live position/chain feed |
| Hard risk gates | done |
| Sizing and chain verification | done |
| Alpaca MCP server | connected, live-verified |
| Execution + contract selection | done, verified on a real chain |
| Allocator / Challenger | done, tested offline; live run needs a Claude credential |
| Orchestration / scheduler | not started |
| Equal-weight benchmark and allocation delta | not started |

Order of work follows PRD 4.2: evidence engine first, then one strategy end to end
through MCP, then the models, then the remaining two strategies, then the VPS.

## Quick start

```bash
pip install -r requirements.txt
```

Then put your Alpaca paper credentials in `.env` (gitignored, created for you
by the step above if absent):

```
ALPACA_API_KEY=your_key_id
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER_TRADE=true
```

The same file drives both this code and the MCP server, deliberately: the
official Alpaca MCP server reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`, so
those are canonical here too. Alpaca's own docs also use `ALPACA_API_KEY_ID` /
`ALPACA_API_SECRET_KEY`, which are accepted as a fallback.

```bash
python scripts/precompute_correlation.py
```

```bash
python -m pytest tests/ -q
```

The test suite runs entirely offline against a synthetic fat-tailed price series, so
the maths can be checked without credentials.

## MCP

The interface to Alpaca is the **official Alpaca MCP server**
(`alpacahq/alpaca-mcp-server`, MIT, v2.3.0 from PyPI), installed into `.venv`
on Python 3.13 and registered as a stdio server. Evidence gathering and
execution go through it; scheduling, sizing maths and risk gates stay in own
code.

It covers option chains, latest quotes and trades, snapshots (which carry
Greeks), option bars, and order placement including multi-leg. MCP option
orders are market and limit only, which is all this system places.

Two open questions are deliberately left for a single small live order on day
one (PRD 2.7), because neither can be answered from documentation: whether
multi-leg orders fill atomically or leg by leg, and what margin paper actually
holds for a defined-risk spread. The margin answer determines whether the
20-25% risk budget is realistic, and `estimate_margin` currently returns the
theoretical value pending that measurement.

## Risk gates and sizing

Gates are enforced on the allocator's OUTPUT, in a fixed order, and every
change is recorded with the rule that caused it. The order is not arbitrary:
the adjustment threshold is applied last, because applying it earlier would let
a proposal violating a hard limit survive on the grounds that it was a small
change.

Sizing is own code. The allocator never sees a contract count and is never
asked to produce one. It proposes a share of the risk budget; sizing converts
that to integer contracts against max loss, rounding down so a position can
never exceed its allocation, then checks buying power as a separate gate.
Buying power caps increases and never blocks a reduction.

### Two spec conflicts, resolved and flagged

**Snap-to-zero vs scale-down-only.** PRD 2.2 says snap to zero below 10% of the
budget; PRD 2.3 says the strangle is scale-down-only because closing it fully
converts a maybe into a certain loss at the worst moment. Resolved in favour of
2.3, the more specific rule and the one carrying a stated reason: a
scale-down-only strategy holding a live position floors at the threshold rather
than snapping to zero. It still reaches zero on an explicit target, so a
genuine invalidation can still close it -- it just cannot be closed by drift.
`SNAP_TO_ZERO_OVERRIDES_SCALE_DOWN` in `risk.py` restores the literal 2.2
behaviour.

**"Enough contracts that partial reduction is meaningful."** PRD 2.3 requires
this of the strangle without naming a number. Rather than invent one, the floor
is derived: at n contracts holding a fraction f of the budget, one contract
moves the allocation by f/n, and that step should be no larger than the 5
percentage point threshold below which the system does not trade at all.

## Allocator and Challenger

Two model calls, and the only two in the system. Everything around them --
building the prompt, validating and parsing the response, resolving a
challenge, feeding the result to the gates -- is pure and tested offline with a
fake client. The single seam where a network call happens is `llm.py`.

The **allocator** receives the evidence packet plus each strategy's thesis and
returns target shares of the risk budget with written reasoning, as structured
JSON. It is asked to reconcile evidence against each thesis and judge
risk-adjusted opportunity across the portfolio. It is never asked to forecast
prices, size positions, or emit a confidence score. Its system prompt repeats
the most expensive error in bold terms: a strategy losing money is not the same
as its thesis being invalidated -- read the not_invalidation field.

The **challenger** attacks the proposal on the same evidence and returns
APPROVE, MODIFY, or REJECT with the specific evidence responsible. It is one
model reviewing another with no ground truth and a known bias toward agreement,
so it is a filter that improves the average decision, not certification of any
one. Its rejection rate is reported regardless. It is tuned against performance
chasing, and its own blind spot -- endorsing a premature cut of a bleeding
strategy -- is named in its prompt.

APPROVE uses the proposal, MODIFY uses the challenger's corrected numbers,
REJECT holds the current allocation. Whatever results then goes through the hard
risk gates, which correct anything the models let by.

### Determinism and "temperature zero"

PRD 2.1 specifies temperature zero for reproducibility. That parameter no longer
exists on the current Opus models (it returns a 400). Determinism is pursued the
modern way instead: the response is constrained to a JSON schema, and the
reasoning lands in structured fields rather than free-form prose. Exact
bit-for-bit reproducibility is not achievable with these models; the audit trail
-- logged evidence, reasoning, and decision -- is what provides accountability.

### No confidence scores, enforced twice

PRD forbids model confidence numbers: they look calibrated and are not, and if
one ever multiplied into sizing it would drive real capital off a meaningless
figure. The schemas have no confidence field, and the parsers reject any payload
that smuggles one in under `confidence`, `certainty`, `probability`, or
`conviction`.

### Credentials

`llm.py` never reads or stores a credential; it defers to the SDK's resolution
chain (an `ANTHROPIC_API_KEY`, an `ant auth login` subscription profile, or
workload identity). A missing credential surfaces as `ModelUnavailable`, which
upstream means hold the last valid allocation and do not trade -- the same as
any other model failure (PRD 2.5). The unattended VPS run should use an API key;
a subscription profile works for local development.

## Allocation semantics

Allocation is a share of a **risk budget defined as maximum loss**, not capital
deployed. All three strategies are defined-risk, which gives a common denominator
that capital deployed would not:

| Strategy | Max loss per contract |
| --- | --- |
| Bull put spread | (strike width x 100) - premium collected |
| Bear call spread | (strike width x 100) - premium collected |
| Long strangle | premium paid |

A 40% allocation means the strategy holds 40% of the permitted max-loss exposure, and
means the same thing whichever strategy it lands on. Buying power is a separate hard
gate, never the denominator: margin requirement is a broker constraint independent of
how risk is defined, and the system checks both.

Bounds live in `alloc_agent/config.py`. Total risk budget 20-25% of equity, 45% per
strategy maximum, snap to zero below 10%, and a 5 percentage point adjustment
threshold that hard risk reductions bypass.

## Correlation precompute

Live correlation is impossible with four trading days of data, so the 3x3 matrix is
precomputed from 252 days of QQQ history.

**This is a proxy, not a backtest.** No chain data, no real strikes, no quotes, no
fills, no expiration handling, no transaction costs. It answers one question -- do
these three respond to different conditions -- which is all the allocator needs
correlation for.

### The PRD 2.4 construction does not work, and the reason is structural

PRD 2.4 specifies a piecewise proxy: a small constant on quiet days, the return itself
when a threshold is breached. It also specifies the shape that construction must
produce -- spreads strongly negatively correlated, strangle low or negative against
both -- and states that a construction failing it is wrong.

The piecewise construction fails its own check, returning about **-0.02** between the
two spreads instead of a strong negative. It fails identically at 10%, 17% and 30%
sample volatility, so this is not a sampling artifact.

The cause: outside a breach both spread proxies are *constant*. Across 252 days the
bear call series takes **three distinct values**. Roughly 240 days carry no
information, the two series almost never move on the same day, and their covariance
collapses to about `-2 * credit_bull * credit_bear` -- negligible against standard
deviations set by breach-day magnitudes. Real spread P&L varies continuously with the
underlying through delta; the piecewise version only registers a breach.

### What replaced it

Daily Black-Scholes revaluation. Each structure is set up on the day's real close at
its target deltas and DTE, then revalued one day later on the real next close with an
updated volatility. Daily P&L is normalised by the structure's own max loss -- the
same denominator the allocator allocates in.

This has no hand-set constants. Strikes come from target delta in closed form, rounded
to the $1 grid QQQ actually lists. Volatility is EWMA with lambda 0.94, the RiskMetrics
daily convention, chosen because it is a named standard rather than something fitted
to this sample, and constructed as a forecast so no day is priced with hindsight.
Spread losses are floored at max loss, which is what defined risk means.

Pricing volatility is realised volatility, so the proxy contains **no volatility risk
premium**. That is deliberate: the correlation then reflects the shape of the three
payoffs rather than an assumed edge doing the work.

Both matrices are written to `artifacts/correlation.json` and printed side by side, so
the difference is visible rather than asserted.

### A result worth knowing before reading the matrix

On a negatively-skewed sample, the strangle comes out mildly **positively** correlated
with the bear call spread (about +0.15) while strongly negative against the bull put
(about -0.64). Both a short call spread and a long strangle profit in a selloff, so
that pair diversifies less than the design assumes. QQQ has the same skew, so expect
this to persist on live data.

## Evidence engine

Emits evidence, not verdicts. Nothing in the layer decides whether a thesis
holds; it assembles the facts and hands the thesis text over unchanged
alongside them. There is no `is_cheap`, no `trend_intact`, no `should_cut`. A
test walks the serialised packet and fails on any boolean or verdict-shaped key,
because the failure mode is quiet: one `thesis_holds` field and the allocator is
reduced to writing prose around it.

Per strategy: allocation as a share of the risk budget, max loss outstanding,
P&L expressed against risk taken rather than against capital, drawdown measured
against max loss, days held, and per-leg strike, expiry, DTE, delta, mid,
bid-ask spread and strike distance. Portfolio: equity, buying power, and risk
budget consumed against available. Plus the precomputed correlation matrix and
the constraints, restated as input so the allocator proposes inside the right
box while enforcement still happens on its output.

Strike distance is reported in **sigma** as well as percent. How far spot sits
from a strike in standard deviations of the move expected over the remaining
life is what makes a 2% buffer at 8% volatility comparable to a 2% buffer at
30%, and it is arithmetic rather than judgement.

A flat strategy still appears in the packet, at zero, with its thesis attached.
A strategy cut to nothing that vanished from the evidence could never be funded
again.

### IV percentile

Implied volatility percentile is load-bearing: the strangle's invalidation is
anchored to whether protection is still cheap relative to its own history, and
the spreads' theses turn on implied sitting above realised. None of that is
evaluable from a live chain snapshot, and four days of accumulated IV is not a
distribution.

The reference series is CBOE's published **VXN** history, back to 2009. VXN is
the Nasdaq-100 volatility index, which is the correct reference for QQQ; VIX is
the S&P and currently understates it by around five vol points.

Both a trailing-252-day and a full-history percentile are reported, because they
can disagree sharply and the disagreement is itself evidence. At the time of
writing VXN sits at 20.24 -- the **20th** percentile of the last year but the
**53rd** of the full history. Protection is cheap by recent standards and
ordinary by long ones.

This is a market-wide reference for where implied volatility sits in its range,
not the implied volatility of the specific contracts held. Those come from the
chain through MCP and are reported alongside it, never replaced by it.

## Layout

```
alloc_agent/
  config.py               policy limits, cadence, credentials
  strategies.py           the three strategies and their theses
  data/bars.py            Alpaca daily bars, cached to CSV
  data/vol_index.py       CBOE VXN history, cached to CSV
  evidence/bs.py          Black-Scholes, precompute only
  evidence/correlation.py the 3x3 matrix and both constructions
  evidence/metrics.py     returns, volatility, percentile, strike distance
  evidence/packet.py      the evidence packet the allocator receives
scripts/
  precompute_correlation.py
tests/
```

Nothing in the live path prices anything. `evidence/bs.py` exists only so the
correlation input has no hand-set constants in it; live Greeks, quotes and premiums
come from the chain through MCP.

## Environment notes

Alpaca paper trading, $100K starting equity. Paper is not a delayed tape -- fills use
live quotes, and chains and quotes through the Market Data API are real-time.

On the free Basic plan the options feed is **Indicative, not full OPRA**, so Greeks are
less reliable than they would be on OPRA. Accepted, and noted here rather than
buried. The only 15-minute restriction on Basic is historical bars and trades, where
the most recent 15 minutes cannot be pulled; realised-volatility calculations are
therefore anchored to completed bars, and daily bars are requested through the last
completed session so a partial bar never enters a volatility calculation.

## Third-party components

Named per hackathon rules:

- **CBOE** -- published VXN daily history, used as the implied-volatility
  reference series. External data provider, no credentials required.
- **scipy** -- normal distribution functions in the Black-Scholes precompute
- **numpy** -- array maths throughout
- **httpx** -- HTTP client for the Alpaca bars and CBOE endpoints
- **pytest** -- test suite

No public GitHub components vendored.

## Known limits

The judgement is only valuable if it beats a simple rule, and that is unproven.

Over four trading days the situations that would demonstrate adaptive allocation may
never occur, particularly in a quiet tape where the correct decision is obvious and
unchanging. Shock simulations in this repo are the mitigation, with paper equity
remaining the scored P&L record.

The Challenger is one model reviewing another with no ground truth, and models have a
documented tendency toward agreement. It is a filter that improves the average
decision, not certification of any individual one. Its rejection rate is reported; if
it approves nearly everything, that will be stated.
