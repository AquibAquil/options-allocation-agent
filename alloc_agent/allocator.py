"""The Allocator (PRD 2.1).

Receives the evidence packet plus each strategy's thesis and outputs target
allocations as shares of the risk budget, with written reasoning, in structured
JSON so the decision is replayable and testable.

What it is asked to do: reconcile the evidence against each stated thesis, and
judge risk-adjusted opportunity relative to the rest of the portfolio, given
correlation and concentration.

What it is never asked to do:
  - forecast prices
  - size positions (that is own code, in sizing.py)
  - emit a confidence score (PRD is explicit: model confidence looks calibrated
    and is not; if one ever multiplied into sizing, a meaningless number would
    drive real capital)

The allocations it returns are proposals. The hard risk gates run on this
output afterwards, regardless of what it proposed -- you cannot bound a
proposal that does not exist yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .evidence.packet import EvidencePacket
from .llm import ModelClient, ModelResponse, ModelUnavailable
from .strategies import KEYS

# --- structured output schema ---------------------------------------------
# Strict: exactly the three strategy keys, numbers in [0, 1], plus reasoning.
# No confidence field anywhere, by design.

_ALLOC_PROPS = {key: {"type": "number", "minimum": 0.0, "maximum": 1.0} for key in KEYS}
_REASON_PROPS = {key: {"type": "string", "minLength": 1} for key in KEYS}

ALLOCATION_SCHEMA = {
    "type": "object",
    "properties": {
        "allocations": {
            "type": "object",
            "properties": _ALLOC_PROPS,
            "required": list(KEYS),
            "additionalProperties": False,
        },
        "reasoning": {
            "type": "object",
            "properties": _REASON_PROPS,
            "required": list(KEYS),
            "additionalProperties": False,
        },
        "portfolio_rationale": {"type": "string", "minLength": 1},
    },
    "required": ["allocations", "reasoning", "portfolio_rationale"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are the allocator in a multi-strategy options trading system. You decide \
what share of a fixed risk budget each of three verified strategies should \
hold, right now, given the evidence.

The risk budget is denominated in MAXIMUM LOSS. An allocation of 0.40 to a \
strategy means it holds 40% of the permitted max-loss exposure. Allocations are \
fractions in [0, 1] and should sum to at most 1.0; holding budget back (summing \
to less than 1.0) is a legitimate, sometimes correct, decision.

CRITICAL -- each number you output is the TOTAL target share you want that \
strategy to hold AFTER this decision: the desired end state, not a change and \
not an addition. Each strategy's current allocation is given in the evidence as \
its starting point. Your number REPLACES it. If a strategy currently holds 35% \
and you still want it at 35%, output 0.35 -- not 0. Never treat the current \
allocation as already spent, and never allocate only the "leftover" budget on \
top of the held positions: propose the full target book from scratch every time, \
as if deciding all three positions afresh. The sum of your three targets is the \
whole book's max-loss exposure, capped at 1.0 -- it is not the amount of new \
budget to add.

Your one job is to reconcile the evidence against each strategy's written \
thesis and its stated invalidation conditions, and to judge risk-adjusted \
opportunity relative to the rest of the portfolio -- accounting for correlation \
and concentration, not "which strategy looks best in isolation".

Hard rules:
- Do NOT forecast prices or predict direction. Judge whether the thesis \
conditions currently hold, not where the market will go.
- Do NOT size positions or mention contract counts. You allocate a share of a \
budget; separate code converts that to contracts.
- Do NOT output confidence scores, probabilities, or certainty levels. They \
look calibrated and are not.
- Use ONLY the evidence provided. Do not invent facts, levels, or history that \
the evidence does not contain.

Critical judgement, stated in the theses and repeated here because it is the \
most common expensive error: a strategy LOSING MONEY is not the same as its \
thesis being invalidated. Read each strategy's not_invalidation field. The long \
strangle in particular bleeds small losses while it waits; cutting it for \
losing money is buying volatility and selling it right before it pays. Anchor \
every judgement to the thesis and invalidation conditions, never to the P&L \
line alone.

The hard risk limits in the constraints are enforced by separate code on your \
output. Propose within them, but know that a proposal breaching them will be \
corrected, not executed as written.

Return your decision in the required JSON structure: a target allocation for \
each strategy, a short evidence-grounded reason for each, and a portfolio-level \
rationale covering correlation and concentration."""


@dataclass(frozen=True)
class AllocationProposal:
    allocations: dict[str, float]
    reasoning: dict[str, str]
    portfolio_rationale: str
    raw: ModelResponse | None = None

    def to_dict(self) -> dict:
        return {
            "allocations": self.allocations,
            "reasoning": self.reasoning,
            "portfolio_rationale": self.portfolio_rationale,
            "model": self.raw.model if self.raw else None,
            "usage": self.raw.usage if self.raw else None,
        }


def build_user_prompt(packet: EvidencePacket) -> str:
    """Serialise the evidence packet into the allocator's input.

    The packet already carries each strategy's thesis, invalidation, and
    not_invalidation, so the model sees the theses and the facts together and
    has to do the reconciling itself.
    """
    payload = packet.to_dict()
    return (
        "Here is the current evidence packet. Every number in it was computed "
        "deterministically upstream; treat it as ground truth and use nothing "
        "else.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Decide the TOTAL target allocation for each strategy now -- the full "
        "share of the risk budget you want it to hold after this decision, "
        "replacing whatever it currently holds (its current share is in the "
        "evidence). For each, state in one or two sentences the specific "
        "evidence that supports your target and how it bears on that strategy's "
        "thesis and invalidation conditions. Then give a portfolio-level "
        "rationale that addresses correlation between the strategies and "
        "concentration of the budget."
    )


def parse_allocation(data: dict, raw: ModelResponse | None = None) -> AllocationProposal:
    """Validate and structure the model's JSON. Defensive despite the schema.

    Structured output guarantees the shape, but this runs the same validation
    against a fake client in tests and against any future schema drift, so it
    does not trust the shape it is handed.
    """
    if not isinstance(data, dict):
        raise ModelUnavailable(f"allocation is not an object: {type(data).__name__}")

    allocations = data.get("allocations")
    reasoning = data.get("reasoning")
    rationale = data.get("portfolio_rationale")

    if not isinstance(allocations, dict):
        raise ModelUnavailable("allocation is missing the allocations object")
    missing = set(KEYS) - set(allocations)
    unknown = set(allocations) - set(KEYS)
    if missing:
        raise ModelUnavailable(f"allocations missing keys: {sorted(missing)}")
    if unknown:
        raise ModelUnavailable(f"allocations has unknown keys: {sorted(unknown)}")

    clean: dict[str, float] = {}
    for key in KEYS:
        value = allocations[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelUnavailable(f"allocation for {key} is not a number: {value!r}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ModelUnavailable(f"allocation for {key} out of [0,1]: {value}")
        clean[key] = value

    if not isinstance(reasoning, dict) or set(KEYS) - set(reasoning):
        raise ModelUnavailable("allocation is missing per-strategy reasoning")
    reasons = {key: str(reasoning[key]) for key in KEYS}

    if not isinstance(rationale, str) or not rationale.strip():
        raise ModelUnavailable("allocation is missing a portfolio rationale")

    # Guard against a confidence field smuggled in under any name.
    banned = {"confidence", "certainty", "probability", "conviction"}
    if banned & {k.lower() for k in data}:
        raise ModelUnavailable("allocation contains a forbidden confidence field")

    return AllocationProposal(
        allocations=clean,
        reasoning=reasons,
        portfolio_rationale=rationale.strip(),
        raw=raw,
    )


class Allocator:
    def __init__(self, client: ModelClient):
        self._client = client

    def propose(self, packet: EvidencePacket) -> AllocationProposal:
        """Produce a target allocation. Raises ModelUnavailable on any failure.

        The caller treats ModelUnavailable as "hold last valid allocation, do
        not trade" (PRD 2.5).
        """
        response = self._client.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(packet),
            schema=ALLOCATION_SCHEMA,
        )
        return parse_allocation(response.data, response)
