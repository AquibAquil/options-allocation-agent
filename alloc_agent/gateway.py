"""The broker seam (PRD 2.6).

Every interaction with Alpaca -- clock, calendar, account, positions, chains,
order placement, order status -- goes through this interface. The real
implementation drives the official Alpaca MCP server (the `mcp` Python SDK lets
a program be an MCP client, so this runs unattended on a VPS while still using
MCP as the interface). Tests use a fake that returns canned data, so the whole
orchestration cycle is exercised offline.

Nothing above this interface knows or cares whether it is talking to MCP, a
REST client, or a fixture. That is the point: the cycle logic is pure, and this
is the one place the outside world enters.
"""

from __future__ import annotations

from typing import Protocol

from .execution import FillReport, OrderSpec


class BrokerError(RuntimeError):
    """A broker interaction failed. PRD 2.5: on failure, do not assume -- check
    actual state before acting, and when in doubt hold."""


class MarketClosed(RuntimeError):
    """The market is not open. Not an error; a reason to hold this cycle."""


class BrokerGateway(Protocol):
    def is_market_open(self) -> bool: ...

    def account(self):
        """An AccountSnapshot (equity, buying power, options buying power)."""
        ...

    def daily_bars(self, symbol: str, *, days: int) -> list[dict]:
        """Completed daily bars, oldest first (PRD 2.7: no partial last bar)."""
        ...

    def positions(self) -> dict:
        """Live positions as {strategy_key: PositionSnapshot | None}.

        Mapping option legs back to the strategy that opened them is the real
        gateway's job (it tags every order's client_order_id with the strategy
        key). A flat strategy maps to None.
        """
        ...

    def option_chain(self, symbol: str, **filters) -> dict:
        """Chain snapshot map for selection and marking."""
        ...

    def place_order(self, spec: OrderSpec) -> dict:
        """Submit an order. Returns the raw order object."""
        ...

    def order_status(self, *, client_order_id: str) -> dict:
        """Fetch an order's current state, for post-submit verification."""
        ...


def verify_after_submit(
    gateway: BrokerGateway, spec: OrderSpec, submit_response: dict
) -> FillReport:
    """Check an order's ACTUAL status after submitting (PRD 2.5).

    An order failing to place, or a submit call timing out, must never be
    assumed either way. The idempotency key lets a status re-query stand in for
    a possibly-lost submit response: the truth is the order record, not the
    return value of the call that created it.
    """
    from .execution import interpret_fill

    report = interpret_fill(submit_response or {})
    if report.status in ("unknown", "") or not report.order_id:
        # The submit response was unusable; go find the order by its id.
        try:
            actual = gateway.order_status(client_order_id=spec.client_order_id)
        except BrokerError:
            # Cannot confirm; report the uncertainty rather than guess a fill.
            return interpret_fill({"status": "unknown", "client_order_id": spec.client_order_id})
        return interpret_fill(actual)
    return report
