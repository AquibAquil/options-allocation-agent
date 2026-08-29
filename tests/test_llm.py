"""Model-client boundary tests (PRD 2.1, 2.5).

The GroqClient request-building and response-parsing are checked with a stubbed
httpx transport -- no network, no key. The contract that matters: every failure
mode (missing key, HTTP error, truncation, non-JSON, empty) becomes
ModelUnavailable, which upstream means hold the last valid allocation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from alloc_agent import config, llm
from alloc_agent.llm import GroqClient, ModelUnavailable, make_client

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "string"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _stub_transport(handler):
    """Wrap a request handler as an httpx MockTransport and patch httpx.Client."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    return factory


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")


def groq_response(content: str, *, finish="stop", status=200, model="llama-3.3-70b-versatile"):
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text="upstream error")
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    return handler


# --- request building ------------------------------------------------------


def test_request_targets_groq_with_json_mode_and_temperature_zero(groq_key, monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "llama-3.3-70b-versatile",
            "choices": [{"message": {"content": '{"ok": "yes"}'}, "finish_reason": "stop"}],
            "usage": {},
        })

    monkeypatch.setattr(httpx, "Client", _stub_transport(handler))
    GroqClient().complete_json(system="SYS", user="USR", schema=SCHEMA)

    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer test-key"
    body = captured["body"]
    assert body["temperature"] == 0                       # PRD 2.1 intent, honoured
    assert body["response_format"] == {"type": "json_object"}
    # The schema is pasted into the system message for robustness.
    assert "JSON Schema" in body["messages"][0]["content"]
    assert "SYS" in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "USR"


def test_valid_json_response_parses(groq_key, monkeypatch):
    monkeypatch.setattr(httpx, "Client", _stub_transport(groq_response('{"ok": "yes"}')))
    r = GroqClient().complete_json(system="s", user="u", schema=SCHEMA)
    assert r.data == {"ok": "yes"}
    assert r.model == "llama-3.3-70b-versatile"
    assert r.usage["prompt_tokens"] == 10


# --- failure modes all become ModelUnavailable (PRD 2.5) -------------------


def test_missing_key_holds(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(ModelUnavailable, match="GROQ_API_KEY is not set"):
        GroqClient().complete_json(system="s", user="u", schema=SCHEMA)


def test_http_error_holds(groq_key, monkeypatch):
    monkeypatch.setattr(httpx, "Client", _stub_transport(groq_response("", status=500)))
    with pytest.raises(ModelUnavailable, match="500"):
        GroqClient().complete_json(system="s", user="u", schema=SCHEMA)


def test_truncated_response_holds(groq_key, monkeypatch):
    monkeypatch.setattr(httpx, "Client", _stub_transport(groq_response('{"ok":', finish="length")))
    with pytest.raises(ModelUnavailable, match="truncated"):
        GroqClient().complete_json(system="s", user="u", schema=SCHEMA)


def test_non_json_response_holds(groq_key, monkeypatch):
    monkeypatch.setattr(httpx, "Client", _stub_transport(groq_response("here you go: {ok}")))
    with pytest.raises(ModelUnavailable, match="non-JSON"):
        GroqClient().complete_json(system="s", user="u", schema=SCHEMA)


def test_empty_message_holds(groq_key, monkeypatch):
    monkeypatch.setattr(httpx, "Client", _stub_transport(groq_response("")))
    with pytest.raises(ModelUnavailable, match="empty message"):
        GroqClient().complete_json(system="s", user="u", schema=SCHEMA)


# --- factory ---------------------------------------------------------------


def test_factory_returns_groq_by_default(monkeypatch):
    monkeypatch.setattr(config, "MODEL_PROVIDER", "groq")
    assert isinstance(make_client(), GroqClient)


def test_factory_returns_anthropic_when_selected(monkeypatch):
    monkeypatch.setattr(config, "MODEL_PROVIDER", "anthropic")
    assert isinstance(make_client(), llm.AnthropicClient)


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "MODEL_PROVIDER", "gpt5")
    with pytest.raises(ModelUnavailable, match="unknown MODEL_PROVIDER"):
        make_client()


# --- the client satisfies the allocator/challenger contract ----------------


def test_groq_client_drives_the_allocator(groq_key, monkeypatch):
    """End to end with the real allocator parsing a Groq-shaped response."""
    from alloc_agent.allocator import Allocator
    from alloc_agent.strategies import KEYS
    import datetime as dt
    from alloc_agent.evidence import packet as pk
    from conftest import synthetic_bars, synthetic_vxn

    payload = json.dumps({
        "allocations": {k: v for k, v in zip(KEYS, [0.4, 0.1, 0.15])},
        "reasoning": {k: "grounded in the evidence" for k in KEYS},
        "portfolio_rationale": "diversified; spreads negatively correlated",
    })
    monkeypatch.setattr(httpx, "Client", _stub_transport(groq_response(payload)))

    packet = pk.build_packet(
        cycle_id="c1", symbol="QQQ", bars=synthetic_bars(), vol_index_rows=synthetic_vxn(),
        positions={}, account=pk.AccountSnapshot(equity=100_000.0, buying_power=200_000.0),
        correlation={"keys": list(KEYS), "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        asof=dt.date(2026, 8, 31),
    )
    proposal = Allocator(GroqClient()).propose(packet)
    assert proposal.allocations[KEYS[0]] == 0.4
