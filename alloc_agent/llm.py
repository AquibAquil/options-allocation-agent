"""Thin model-client boundary (PRD 2.1).

The allocator and challenger are the only two model calls in the system. Every
piece of logic around them -- building the prompt, validating and parsing the
response, resolving a challenge, feeding the result to the risk gates -- is
pure and tested offline. This module is the single seam where an actual network
call happens, expressed as a small interface so the rest is testable with a
fake.

Authentication is deferred to the SDK's own resolution chain: an
ANTHROPIC_API_KEY, or an `ant auth login` subscription profile, or workload
identity. Nothing here reads or stores a credential. That means a developer
with a Claude subscription can run it with no key, while the unattended VPS run
uses a key -- and this code does not care which.

Determinism (PRD 2.1 "temperature zero"): temperature is gone on the current
Opus models. Instead the response is constrained to a JSON schema via
output_config.format, so the shape is guaranteed and the reasoning lands in
structured fields. See config.py for the full note.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from . import config


class ModelUnavailable(RuntimeError):
    """A model call could not be made or returned unusably.

    PRD 2.5: on a failed or malformed model call, hold the last valid
    allocation and do not trade. Callers catch this and decline to act rather
    than letting a transient failure move the portfolio.
    """


@dataclass(frozen=True)
class ModelResponse:
    data: dict
    raw_text: str
    model: str
    usage: dict = field(default_factory=dict)


class ModelClient(Protocol):
    """The seam. Real and fake implementations both satisfy this."""

    def complete_json(
        self, *, system: str, user: str, schema: dict, max_tokens: int = ...
    ) -> ModelResponse: ...


class AnthropicClient:
    """Real client. The `anthropic` import is lazy so tests never need it."""

    def __init__(self, model: str = config.ALLOCATOR_MODEL):
        self.model = model
        self._client = None

    def _ensure(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailable(
                    "the `anthropic` package is not installed (pip install anthropic)"
                ) from exc
            # Zero-arg constructor: resolves ANTHROPIC_API_KEY, then an
            # `ant auth login` profile, then workload identity. No key is read
            # or stored here.
            self._client = anthropic.Anthropic()
        return self._client

    def complete_json(
        self, *, system: str, user: str, schema: dict, max_tokens: int = config.MODEL_MAX_TOKENS
    ) -> ModelResponse:
        client = self._ensure()
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # Structured output: the response is a single text block of
                # schema-valid JSON. No temperature (removed on Opus 5/4.8).
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except ModelUnavailable:
            raise
        except Exception as exc:  # SDK raises many typed errors; treat all as a hold
            raise ModelUnavailable(f"model call failed: {type(exc).__name__}: {exc}") from exc

        if getattr(resp, "stop_reason", None) == "refusal":
            raise ModelUnavailable("model refused the request")

        try:
            text = next(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except StopIteration as exc:
            raise ModelUnavailable("model returned no text block") from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelUnavailable(f"model returned non-JSON despite schema: {exc}") from exc

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
            }
        return ModelResponse(
            data=data, raw_text=text, model=getattr(resp, "model", self.model), usage=usage
        )
