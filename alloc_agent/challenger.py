"""The Challenger (PRD 2.1).

Attacks the allocator's proposal using the same evidence, and returns APPROVE,
MODIFY, or REJECT with the specific evidence responsible.

What it is, honestly (PRD 2.1): one model reviewing another, with no ground
truth, and a documented tendency toward agreement. It is a filter that improves
the average decision, not certification of any individual one. Its rejection
rate is reported regardless of what it shows; if it approves nearly everything,
that is stated rather than hidden.

It is tuned against performance chasing -- over-allocating to whatever just went
up. It will therefore NOT reliably catch the opposite error of cutting a
bleeding strategy too early; that error is guarded in the thesis and the risk
gates, not here.

It cannot invent facts the evidence engine did not compute. That is enforced by
prompt and, for its modified numbers, by the same validation the allocator's
output gets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .allocator import AllocationProposal
from .evidence.packet import EvidencePacket
from .llm import ModelClient, ModelResponse, ModelUnavailable
from .strategies import KEYS

VERDICTS = ("APPROVE", "MODIFY", "REJECT")

_ALLOC_PROPS = {key: {"type": "number", "minimum": 0.0, "maximum": 1.0} for key in KEYS}

CHALLENGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "critique": {"type": "string", "minLength": 1},
        "evidence_cited": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        # Always present for a strict schema; authoritative only when the
        # verdict is MODIFY. On APPROVE it should echo the proposal; on REJECT
        # it is ignored (the system holds its current allocation instead).
        "modified_allocations": {
            "type": "object",
            "properties": _ALLOC_PROPS,
            "required": list(KEYS),
            "additionalProperties": False,
        },
    },
    "required": ["verdict", "critique", "evidence_cited", "modified_allocations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are the challenger in a multi-strategy options trading system. Another model \
(the allocator) has proposed target allocations across three strategies, given \
the evidence. Your job is to attack that proposal using the SAME evidence and \
return one of three verdicts: APPROVE, MODIFY, or REJECT.

You are specifically tuned to catch PERFORMANCE CHASING: over-allocating to a \
strategy mainly because it recently made money, or cutting one mainly because \
it recently lost money, when the evidence about the thesis does not justify the \
move. A change in allocation must be justified by evidence bearing on whether \
the thesis conditions hold, not by recent P&L alone.

Hard rules:
- Use ONLY the evidence provided and the allocator's stated reasoning. You may \
NOT invent facts, levels, or history the evidence does not contain. If you \
claim something, it must be traceable to a field in the evidence packet.
- Cite the specific evidence responsible for your verdict. Vague objections are \
not useful; name the number or condition.
- Do NOT output confidence scores or probabilities.
- Do NOT forecast prices.

Be aware of your own bias: you are one model reviewing another, and models tend \
to agree. Do not manufacture disagreement to seem useful, but do not rubber- \
stamp either. APPROVE when the proposal is well-grounded in the evidence.

Also be aware of your blind spot: you are tuned against over-allocating to \
winners, so you are LESS likely to catch the opposite error -- a strategy being \
cut too early while it is merely bleeding as its thesis expects (the long \
strangle especially). Do not reflexively endorse a cut just because a strategy \
lost money; check the not_invalidation field before agreeing to reduce it.

Verdicts:
- APPROVE: the proposal is consistent with the evidence and the theses. \
Echo the allocations unchanged in modified_allocations.
- MODIFY: the proposal is mostly sound but one or more shares are not \
supported by the evidence. Put your corrected allocations in \
modified_allocations and explain each change with the evidence for it.
- REJECT: the proposal is substantially unsupported by the evidence. Explain \
why. modified_allocations will be ignored on a REJECT; the system holds its \
current allocation instead.

Return your verdict in the required JSON structure."""


@dataclass(frozen=True)
class ChallengeResult:
    verdict: str
    critique: str
    evidence_cited: tuple[str, ...]
    modified_allocations: dict[str, float]
    raw: ModelResponse | None = None

    @property
    def approved(self) -> bool:
        return self.verdict == "APPROVE"

    @property
    def rejected(self) -> bool:
        return self.verdict == "REJECT"

    @property
    def modified(self) -> bool:
        return self.verdict == "MODIFY"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "critique": self.critique,
            "evidence_cited": list(self.evidence_cited),
            "modified_allocations": self.modified_allocations,
            "model": self.raw.model if self.raw else None,
            "usage": self.raw.usage if self.raw else None,
        }


def build_user_prompt(packet: EvidencePacket, proposal: AllocationProposal) -> str:
    return (
        "Evidence packet (deterministic ground truth -- use nothing else):\n\n"
        f"{json.dumps(packet.to_dict(), indent=2, default=str)}\n\n"
        "The allocator proposed:\n\n"
        f"{json.dumps(proposal.to_dict(), indent=2, default=str)}\n\n"
        "Attack this proposal. Is each share supported by evidence bearing on "
        "that strategy's thesis and invalidation conditions, or is any of it "
        "performance chasing? Return APPROVE, MODIFY, or REJECT, cite the "
        "specific evidence responsible, and if you MODIFY, give corrected "
        "allocations."
    )


def parse_challenge(data: dict, raw: ModelResponse | None = None) -> ChallengeResult:
    if not isinstance(data, dict):
        raise ModelUnavailable(f"challenge is not an object: {type(data).__name__}")

    verdict = data.get("verdict")
    if verdict not in VERDICTS:
        raise ModelUnavailable(f"challenge verdict is not one of {VERDICTS}: {verdict!r}")

    critique = data.get("critique")
    if not isinstance(critique, str) or not critique.strip():
        raise ModelUnavailable("challenge is missing a critique")

    cited = data.get("evidence_cited")
    if not isinstance(cited, list) or not cited or not all(isinstance(c, str) for c in cited):
        raise ModelUnavailable("challenge is missing cited evidence")

    modified_raw = data.get("modified_allocations")
    if not isinstance(modified_raw, dict) or set(KEYS) - set(modified_raw):
        raise ModelUnavailable("challenge is missing modified_allocations")

    modified: dict[str, float] = {}
    for key in KEYS:
        value = modified_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelUnavailable(f"modified allocation for {key} is not a number: {value!r}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ModelUnavailable(f"modified allocation for {key} out of [0,1]: {value}")
        modified[key] = value

    banned = {"confidence", "certainty", "probability", "conviction"}
    if banned & {k.lower() for k in data}:
        raise ModelUnavailable("challenge contains a forbidden confidence field")

    return ChallengeResult(
        verdict=verdict,
        critique=critique.strip(),
        evidence_cited=tuple(cited),
        modified_allocations=modified,
        raw=raw,
    )


class Challenger:
    def __init__(self, client: ModelClient):
        self._client = client

    def review(self, packet: EvidencePacket, proposal: AllocationProposal) -> ChallengeResult:
        response = self._client.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(packet, proposal),
            schema=CHALLENGE_SCHEMA,
        )
        return parse_challenge(response.data, response)


def resolve_effective_allocation(
    proposal: AllocationProposal,
    challenge: ChallengeResult,
    current_allocation: dict[str, float],
) -> tuple[dict[str, float], str]:
    """Turn a proposal + verdict into the allocation that goes to the gates.

    APPROVE -> the proposal.
    MODIFY  -> the challenger's corrected allocations.
    REJECT  -> hold the current allocation; do not trade on an unsupported call.

    Returns (allocation, source) where source names which of the three applied,
    for the cycle log.
    """
    if challenge.approved:
        return dict(proposal.allocations), "allocator_proposal"
    if challenge.modified:
        return dict(challenge.modified_allocations), "challenger_modified"
    return dict(current_allocation), "held_on_reject"
