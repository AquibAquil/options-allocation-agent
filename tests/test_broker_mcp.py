"""MCP gateway parsing and mapping tests (PRD 2.6).

The live connection cannot be unit-tested (it needs the running server, which is
validated separately). What IS tested here is everything pure: unwrapping the
security envelope, surfacing tool errors as holds, and -- the load-bearing part
-- mapping live leg positions back to the strategy that opened them, using our
own order history as the source of truth.
"""

from __future__ import annotations

import datetime as dt

import pytest

from alloc_agent import broker_mcp as bm
from alloc_agent.gateway import BrokerError
from alloc_agent.strategies import BULL_PUT_SPREAD, BEAR_CALL_SPREAD, KEYS, LONG_STRANGLE

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key
ASOF = dt.date(2026, 9, 2)


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, text):
        self.content = [_Block(text)]


# --- envelope handling -----------------------------------------------------


def test_extract_and_unwrap_strip_the_envelope():
    result = _Result('{"_alpaca_mcp_security": {"trust": "x"}, "data": {"is_open": true}}')
    data = bm.unwrap(bm.extract_json(result), "get_clock")
    assert data == {"is_open": True}


def test_string_payload_is_surfaced_as_a_hold():
    """A string 'data' is an error the server passed through -> BrokerError."""
    result = _Result('"2 validation errors for call"')
    with pytest.raises(BrokerError, match="validation errors"):
        bm.unwrap(bm.extract_json(result), "get_stock_bars")


def test_non_json_content_raises():
    with pytest.raises(BrokerError, match="non-JSON"):
        bm.extract_json(_Result("not json at all {"))


def test_payload_without_data_is_tolerated():
    assert bm.unwrap({"is_open": False}, "get_clock") == {"is_open": False}


# --- account / bars --------------------------------------------------------


def test_parse_account():
    acct = bm.parse_account({"equity": "100000", "buying_power": "400000", "options_buying_power": "100000"})
    assert acct.equity == 100_000.0
    assert acct.buying_power == 400_000.0
    assert acct.options_buying_power == 100_000.0


def test_parse_account_missing_field_holds():
    with pytest.raises(BrokerError, match="equity"):
        bm.parse_account({"buying_power": "1"})


def test_parse_bars_takes_the_symbol_series_and_tail():
    data = {"bars": {"QQQ": [{"c": i} for i in range(300)]}}
    bars = data["bars"]["QQQ"]
    assert bm.parse_bars(data, "QQQ") == bars


def test_parse_bars_missing_symbol_holds():
    with pytest.raises(BrokerError, match="no bars"):
        bm.parse_bars({"bars": {}}, "QQQ")


# --- client_order_id -> strategy ------------------------------------------


def test_strategy_recovered_from_client_order_id():
    assert bm.strategy_from_client_order_id("alloc-bull_put_spread-open-abc123") == BULL
    assert bm.strategy_from_client_order_id("alloc-long_strangle-reduce-def456") == STRANGLE


def test_foreign_client_order_id_is_ignored():
    assert bm.strategy_from_client_order_id("some-other-order") is None
    assert bm.strategy_from_client_order_id(None) is None
    assert bm.strategy_from_client_order_id("alloc-not_a_strategy-open-x") is None


# --- position mapping (the load-bearing case) ------------------------------


def bull_put_order():
    return {
        "client_order_id": "alloc-bull_put_spread-open-abc",
        "submitted_at": "2026-09-01T14:00:00Z",
        "filled_at": "2026-09-01T14:00:01Z",
        "legs": [
            {"symbol": "QQQ260911P00702000"},
            {"symbol": "QQQ260911P00699000"},
        ],
    }


def bull_put_positions():
    return [
        {"symbol": "QQQ260911P00702000", "qty": "-10", "avg_entry_price": "3.55", "unrealized_pl": "120"},
        {"symbol": "QQQ260911P00699000", "qty": "10", "avg_entry_price": "3.03", "unrealized_pl": "-40"},
    ]


def test_legs_map_back_to_their_strategy():
    mapping = bm.map_symbols_to_strategies([bull_put_order()])
    assert mapping["QQQ260911P00702000"] == BULL
    assert mapping["QQQ260911P00699000"] == BULL


def test_position_snapshot_is_assembled_from_legs():
    result = bm.map_positions_to_strategies(bull_put_positions(), [bull_put_order()], asof=ASOF)
    snap = result[BULL]
    assert snap is not None
    assert snap.contracts == 10
    assert len(snap.legs) == 2
    # 3-wide spread, 0.52 credit: max loss = 300 - 52 = 248
    assert snap.max_loss_per_contract == pytest.approx(248.0, abs=1.0)
    assert snap.unrealized_pnl == pytest.approx(80.0)   # 120 + (-40)
    assert snap.opened_at == "2026-09-01"
    # Short/long sides recovered from signed qty.
    short = next(l for l in snap.legs if l.action == "sell")
    assert short.strike == 702.0


def test_flat_strategies_map_to_none():
    result = bm.map_positions_to_strategies([], [], asof=ASOF)
    assert set(result) == set(KEYS)
    assert all(v is None for v in result.values())


def test_only_our_orders_participate_in_mapping():
    """A foreign position (not from our orders) is not attributed to a strategy."""
    positions = bull_put_positions() + [
        {"symbol": "AAPL260101C00200000", "qty": "1", "avg_entry_price": "5", "unrealized_pl": "0"}
    ]
    result = bm.map_positions_to_strategies(positions, [bull_put_order()], asof=ASOF)
    # Only the two QQQ legs are attributed; AAPL is ignored.
    assert result[BULL].contracts == 10
    assert result[BEAR] is None


def test_strangle_max_loss_is_the_debit():
    order = {
        "client_order_id": "alloc-long_strangle-open-xyz",
        "submitted_at": "2026-09-01T10:00:00Z",
        "legs": [{"symbol": "QQQ261009C00760000"}, {"symbol": "QQQ261009P00675000"}],
    }
    positions = [
        {"symbol": "QQQ261009C00760000", "qty": "3", "avg_entry_price": "4.10", "unrealized_pl": "10"},
        {"symbol": "QQQ261009P00675000", "qty": "3", "avg_entry_price": "3.90", "unrealized_pl": "-5"},
    ]
    snap = bm.map_positions_to_strategies(positions, [order], asof=ASOF)[STRANGLE]
    assert snap.contracts == 3
    # debit = (4.10 + 3.90) * 100 = 800
    assert snap.max_loss_per_contract == pytest.approx(800.0)
