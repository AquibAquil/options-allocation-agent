"""Shadow book tests (PRD 2.8; fixes the PRD 3 bias).

The point of the shadow book is that every strategy has a return every cycle,
computed the same way whether or not the real portfolio holds it. So the tests
check: positions open on first sight, the return signs match each strategy's
exposure under a move, defined risk floors the loss, and the book rolls when a
structure decays out of its DTE band.
"""

from __future__ import annotations

import datetime as dt

import pytest

from alloc_agent import shadow
from alloc_agent.strategies import BEAR_CALL_SPREAD, BULL_PUT_SPREAD, KEYS, LONG_STRANGLE

BULL, BEAR, STRANGLE = BULL_PUT_SPREAD.key, BEAR_CALL_SPREAD.key, LONG_STRANGLE.key
D0 = dt.date(2026, 8, 31)
SPOT = 580.0
VOL = 0.18


def test_first_mark_opens_all_and_returns_zero():
    book = shadow.ShadowBook()
    r = book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    assert set(r) == set(KEYS)
    assert all(v == 0.0 for v in r.values())      # just opened, no return yet
    assert set(book.positions) == set(KEYS)


def test_every_strategy_has_a_return_even_when_unheld():
    """The whole fix: r_i exists for all three, independent of any real book."""
    book = shadow.ShadowBook()
    book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    r = book.mark(spot=SPOT * 0.98, annual_vol=VOL, asof=D0 + dt.timedelta(days=1))
    assert set(r) == set(KEYS)
    assert all(isinstance(v, float) for v in r.values())


def test_selloff_signs_are_correct():
    """A down move: bull put loses, bear call gains, strangle gains on the move."""
    book = shadow.ShadowBook()
    book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    r = book.mark(spot=SPOT * 0.94, annual_vol=VOL * 1.4, asof=D0 + dt.timedelta(days=1))
    assert r[BULL] < 0        # short put side, spot fell toward/through it
    assert r[BEAR] > 0        # short call side, now safer
    assert r[STRANGLE] > 0    # long vol, a big move pays


def test_rally_signs_are_mirror():
    book = shadow.ShadowBook()
    book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    r = book.mark(spot=SPOT * 1.06, annual_vol=VOL * 1.4, asof=D0 + dt.timedelta(days=1))
    assert r[BULL] > 0
    assert r[BEAR] < 0
    assert r[STRANGLE] > 0


def test_quiet_day_bleeds_the_strangle_and_pays_the_spreads():
    """No move, one day of decay: spreads earn, the long strangle bleeds."""
    book = shadow.ShadowBook()
    book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    r = book.mark(spot=SPOT, annual_vol=VOL, asof=D0 + dt.timedelta(days=1))
    assert r[BULL] > 0
    assert r[BEAR] > 0
    assert r[STRANGLE] < 0


def test_spread_loss_is_floored_at_defined_risk():
    """A catastrophic move cannot lose a spread more than one unit of max loss."""
    book = shadow.ShadowBook()
    book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    # Huge gap down, one day later.
    book.mark(spot=SPOT * 0.80, annual_vol=VOL * 2, asof=D0 + dt.timedelta(days=1))
    pos = book.positions[BULL]
    # The position's marked P&L never exceeds its max loss on the downside.
    pnl = shadow._mark_pnl(pos, SPOT * 0.80, VOL * 2, D0 + dt.timedelta(days=1))
    assert pnl >= -pos.max_loss - 1e-9


def test_book_rolls_when_the_structure_decays_out_of_band():
    """Past the DTE band, the position is replaced by a fresh selection."""
    book = shadow.ShadowBook()
    book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    bull_before = book.positions[BULL]
    # Advance well past the bull put's max DTE so it must roll.
    far = D0 + dt.timedelta(days=BULL_PUT_SPREAD.dte_max + 3)
    r = book.mark(spot=SPOT, annual_vol=VOL, asof=far)
    bull_after = book.positions[BULL]
    assert bull_after.opened_asof == far          # a fresh position
    assert bull_after is not bull_before
    assert isinstance(r[BULL], float)


def test_returns_compound_into_a_coherent_curve():
    """Successive marks give successive period returns the delta can compound."""
    book = shadow.ShadowBook()
    book.mark(spot=SPOT, annual_vol=VOL, asof=D0)
    r1 = book.mark(spot=SPOT, annual_vol=VOL, asof=D0 + dt.timedelta(days=1))
    r2 = book.mark(spot=SPOT, annual_vol=VOL, asof=D0 + dt.timedelta(days=2))
    # Two quiet days: the strangle keeps bleeding, not a one-off.
    assert r1[STRANGLE] < 0 and r2[STRANGLE] < 0


def test_degenerate_volatility_does_not_blow_up():
    book = shadow.ShadowBook()
    r = book.mark(spot=SPOT, annual_vol=0.0, asof=D0)
    assert all(v == 0.0 for v in r.values())      # opened, no crash on vol=0
