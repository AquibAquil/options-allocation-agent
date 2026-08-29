"""Live broker gateway over the Alpaca MCP server (PRD 2.6).

Implements the BrokerGateway interface by driving the official Alpaca MCP server
through the `mcp` Python client. MCP is a protocol, so a program can be an MCP
client -- this runs unattended on a VPS while still using MCP as the interface,
which is what reconciles PRD 2.6 (MCP) with PRD 2.5 (autonomous).

The server takes a few seconds to boot and its tools are fast once up, so the
session is persistent: a background thread owns an asyncio loop and keeps one
stdio session open for the gateway's lifetime, and the synchronous BrokerGateway
methods marshal calls onto it. (The background-thread-plus-subprocess pattern is
a Windows asyncio hazard; it is verified to work here.)

Pure parsing and the position->strategy mapping are module-level functions,
tested offline against captured shapes. The live connection is validated against
the running server; order placement is validated when the market is open.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import threading

from . import config
from .evidence.packet import AccountSnapshot, LegSnapshot, PositionSnapshot
from .execution import OrderSpec, parse_occ_symbol
from .gateway import BrokerError
from .strategies import BY_KEY, KEYS, max_loss_per_contract

DEFAULT_EXE = os.path.join(config.REPO_ROOT, ".venv", "Scripts", "alpaca-mcp-server.exe")
DEFAULT_ENV_FILE = os.path.join(config.REPO_ROOT, ".env")


# ---------------------------------------------------------------------------
# Pure parsing (tested offline)
# ---------------------------------------------------------------------------


def extract_json(call_result) -> object:
    """Pull the JSON payload out of an MCP CallToolResult's text content."""
    for block in getattr(call_result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except (json.JSONDecodeError, AttributeError) as exc:
                raise BrokerError(f"MCP returned non-JSON: {exc}") from exc
    raise BrokerError("MCP result had no text content")


def unwrap(payload: object, tool: str) -> object:
    """Strip the Alpaca MCP security envelope and surface tool errors.

    Every tool result is {_alpaca_mcp_security: {...}, data: <payload>}. A
    string payload is an error message the server passed through; raise it so a
    failed call becomes a hold, never a silent bad value.
    """
    if isinstance(payload, str):
        raise BrokerError(f"{tool}: {payload[:200]}")
    if not isinstance(payload, dict):
        raise BrokerError(f"{tool}: unexpected result type {type(payload).__name__}")
    if "data" not in payload:
        # Some tools may return the payload directly; tolerate that.
        return payload
    return payload["data"]


def parse_account(data: dict) -> AccountSnapshot:
    try:
        equity = float(data["equity"])
        buying_power = float(data["buying_power"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BrokerError(f"account missing equity/buying_power: {exc}") from exc
    obp = data.get("options_buying_power")
    return AccountSnapshot(
        equity=equity,
        buying_power=buying_power,
        options_buying_power=float(obp) if obp is not None else None,
    )


def parse_bars(data: dict, symbol: str) -> list[dict]:
    bars = (data.get("bars") or {}).get(symbol)
    if not bars:
        raise BrokerError(f"no bars for {symbol}")
    return bars


def strategy_from_client_order_id(client_order_id: str | None) -> str | None:
    """Recover the strategy key from an order's client_order_id.

    Our ids are 'alloc-<strategy_key>-<intent>-<digest>'. Strategy keys use
    underscores, so the second hyphen-field is the whole key.
    """
    if not client_order_id or not client_order_id.startswith("alloc-"):
        return None
    parts = client_order_id.split("-")
    if len(parts) >= 2 and parts[1] in KEYS:
        return parts[1]
    return None


def map_symbols_to_strategies(orders: list[dict]) -> dict[str, str]:
    """Build {occ_symbol: strategy_key} from our own order history.

    Alpaca is the source of truth: every leg of every order we placed carries
    the strategy in its client_order_id, so the mapping survives restarts
    without a local state file. Later orders win, which is correct when a symbol
    is reused across cycles.
    """
    mapping: dict[str, str] = {}
    for order in sorted(orders, key=lambda o: str(o.get("submitted_at") or o.get("created_at") or "")):
        strategy = strategy_from_client_order_id(order.get("client_order_id"))
        if not strategy:
            continue
        legs = order.get("legs") or []
        if legs:
            for leg in legs:
                sym = leg.get("symbol")
                if sym:
                    mapping[sym] = strategy
        elif order.get("symbol"):
            mapping[order["symbol"]] = strategy
    return mapping


def build_position_snapshot(
    strategy_key: str, leg_positions: list[dict], *, asof: dt.date, opened_at: str | None
) -> PositionSnapshot | None:
    """Assemble a PositionSnapshot from a strategy's Alpaca leg positions.

    NOTE: validated live only once a real multi-leg position exists (after the
    first fill). The flat case -- no legs -> None -- is the current reality and
    is exercised now.
    """
    if not leg_positions:
        return None
    strategy = BY_KEY[strategy_key]

    legs: list[LegSnapshot] = []
    contracts = 0
    unrealized = 0.0
    short_entry = long_entry = None
    strikes: list[float] = []
    for pos in leg_positions:
        symbol = pos["symbol"]
        parsed = parse_occ_symbol(symbol)
        qty = int(float(pos.get("qty", 0)))
        action = "sell" if qty < 0 else "buy"
        entry = float(pos.get("avg_entry_price", 0.0))
        contracts = max(contracts, abs(qty))
        unrealized += float(pos.get("unrealized_pl", 0.0) or 0.0)
        strikes.append(parsed["strike"])
        if action == "sell":
            short_entry = entry
        else:
            long_entry = entry
        legs.append(
            LegSnapshot(
                occ_symbol=symbol,
                right=parsed["right"],
                action=action,
                strike=parsed["strike"],
                expiry=parsed["expiry"],
                contracts=abs(qty),
                delta=None,
                mid=None,
                implied_vol=None,
            )
        )

    # Per-contract economics from entry prices.
    if strategy.collects_premium and short_entry is not None and long_entry is not None:
        width = abs(strikes[0] - strikes[1]) if len(strikes) >= 2 else 0.0
        premium = (short_entry - long_entry) * 100.0
        try:
            max_loss = max_loss_per_contract(strategy_key, premium=premium, strike_width=width)
        except ValueError:
            max_loss = max(width * 100.0 - premium, 1.0)
    else:
        premium = sum(float(p.get("avg_entry_price", 0.0)) for p in leg_positions) * 100.0
        max_loss = premium if premium > 0 else 1.0

    return PositionSnapshot(
        strategy_key=strategy_key,
        contracts=contracts,
        legs=tuple(legs),
        max_loss_per_contract=max_loss,
        entry_premium=premium,
        opened_at=opened_at or asof.isoformat(),
        unrealized_pnl=unrealized,
    )


def map_positions_to_strategies(
    positions: list[dict], orders: list[dict], *, asof: dt.date
) -> dict[str, PositionSnapshot | None]:
    """Group live leg positions into per-strategy PositionSnapshots."""
    symbol_map = map_symbols_to_strategies(orders)
    opened: dict[str, str] = {}
    for order in orders:
        s = strategy_from_client_order_id(order.get("client_order_id"))
        when = order.get("filled_at") or order.get("submitted_at") or order.get("created_at")
        if s and when and s not in opened:
            opened[s] = str(when)[:10]

    by_strategy: dict[str, list[dict]] = {k: [] for k in KEYS}
    for pos in positions:
        sym = pos.get("symbol")
        strat = symbol_map.get(sym)
        if strat:
            by_strategy[strat].append(pos)

    return {
        key: build_position_snapshot(key, legs, asof=asof, opened_at=opened.get(key))
        for key, legs in by_strategy.items()
    }


# ---------------------------------------------------------------------------
# Async session bridge
# ---------------------------------------------------------------------------


class _Bridge:
    """A persistent MCP session on an asyncio loop in a background thread."""

    def __init__(self, exe: str, env_file: str, *, connect_timeout: float = 60.0):
        self._exe = exe
        self._env_file = env_file
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(connect_timeout):
            raise BrokerError("MCP session did not become ready in time")
        if self._error:
            raise BrokerError(f"MCP session failed to start: {self._error}")

    def _run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._serve())
        except Exception as exc:  # surfaced to the constructor via _ready/_error
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        env = dict(os.environ)
        env["ALPACA_PAPER_TRADE"] = "true"
        params = StdioServerParameters(
            command=self._exe,
            args=["--transport", "stdio", "--env-file", self._env_file],
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop.wait()

    def call(self, tool: str, args: dict | None = None, *, timeout: float = 90.0) -> object:
        if self._session is None or self._loop is None:
            raise BrokerError("MCP session is not connected")
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool, args or {}), self._loop
        )
        try:
            result = fut.result(timeout=timeout)
        except Exception as exc:
            raise BrokerError(f"MCP call {tool} failed: {type(exc).__name__}: {exc}") from exc
        return unwrap(extract_json(result), tool)

    def close(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread.is_alive():
            self._thread.join(timeout=15)


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class McpBrokerGateway:
    """Live BrokerGateway. Use as a context manager, one session per lifetime."""

    def __init__(
        self,
        *,
        exe: str = DEFAULT_EXE,
        env_file: str = DEFAULT_ENV_FILE,
        symbol: str = config.UNDERLYING,
        asof: dt.date | None = None,
    ):
        self._exe = exe
        self._env_file = env_file
        self._symbol = symbol
        self._asof = asof or dt.date.today()
        self._bridge: _Bridge | None = None

    def __enter__(self) -> "McpBrokerGateway":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        if self._bridge is None:
            self._bridge = _Bridge(self._exe, self._env_file)

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    @property
    def _b(self) -> _Bridge:
        if self._bridge is None:
            raise BrokerError("gateway is not connected (use `with McpBrokerGateway() as gw:`)")
        return self._bridge

    # -- BrokerGateway ------------------------------------------------------

    def is_market_open(self) -> bool:
        data = self._b.call("get_clock")
        return bool(data.get("is_open"))

    def account(self) -> AccountSnapshot:
        return parse_account(self._b.call("get_account_info"))

    def daily_bars(self, symbol: str, *, days: int) -> list[dict]:
        data = self._b.call(
            "get_stock_bars",
            {
                "symbols": symbol,
                "timeframe": "1Day",
                "days": int(days * 1.6) + 20,  # overshoot weekends/holidays
                "adjustment": "all",
                "sort": "asc",
                "limit": 10000,
            },
        )
        return parse_bars(data, symbol)[-days:]

    def positions(self) -> dict:
        positions = (self._b.call("get_all_positions") or {}).get("result", [])
        orders = (self._b.call("get_orders", {"status": "all", "limit": 500, "nested": True}) or {}).get("result", [])
        return map_positions_to_strategies(positions, orders, asof=self._asof)

    def option_chain(self, symbol: str, **filters) -> dict:
        args = {"underlying_symbol": symbol, "feed": "indicative", "limit": 1000}
        args.update(filters)
        return self._b.call("get_option_chain", args)

    def place_order(self, spec: OrderSpec) -> dict:
        return self._b.call("place_option_order", spec.to_mcp_kwargs())

    def order_status(self, *, client_order_id: str) -> dict:
        return self._b.call("get_order_by_client_id", {"client_order_id": client_order_id})
