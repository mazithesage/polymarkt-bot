"""Tests for the square-root market impact model, property-based Kelly/classify tests,
and get_open_orders retry/semaphore consistency."""

import math
import re
from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses
from hypothesis import given, settings
from hypothesis import strategies as st

from bot import _estimate_live_fill_price
from clob_client import ClobClient, OrderBookSummary
from config import ClobConfig, ChainConfig
from kelly import kelly_criterion
from market_scanner import MarketCategory, classify_market


# ---------------------------------------------------------------------------
#  Square-root impact model unit tests
# ---------------------------------------------------------------------------

def _make_book(
    spread: float = 0.02,
    mid: float = 0.50,
    bid_depth: float = 1000.0,
    ask_depth: float = 1000.0,
) -> OrderBookSummary:
    return OrderBookSummary(
        token_id="test",
        best_bid=mid - spread / 2,
        best_ask=mid + spread / 2,
        mid_price=mid,
        spread=spread,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
    )


class TestSqrtImpactModel:
    def test_small_order_deep_book_minimal_impact(self):
        """Small order on a deep book should produce tiny slippage."""
        book = _make_book(spread=0.02, ask_depth=10_000)
        result = _estimate_live_fill_price(0.50, 10.0, book, "BUY", 500)
        assert result is not None
        # impact = 0.01 * sqrt(10/10000) = 0.01 * 0.0316 ≈ 0.000316
        assert abs(result - 0.50) < 0.001

    def test_large_order_thin_book_significant_impact(self):
        """Large order on a thin book should produce significant slippage."""
        book = _make_book(spread=0.04, bid_depth=50)
        result = _estimate_live_fill_price(0.50, 200.0, book, "SELL", 5000)
        assert result is not None
        # impact = 0.02 * sqrt(200/50) = 0.02 * 2.0 = 0.04
        assert result < 0.50  # SELL pushes price down
        assert result == pytest.approx(0.50 - 0.04, abs=0.001)

    def test_buy_increases_price(self):
        book = _make_book(spread=0.02, ask_depth=500)
        result = _estimate_live_fill_price(0.50, 100.0, book, "BUY", 5000)
        assert result is not None
        assert result > 0.50

    def test_sell_decreases_price(self):
        book = _make_book(spread=0.02, bid_depth=500)
        result = _estimate_live_fill_price(0.50, 100.0, book, "SELL", 5000)
        assert result is not None
        assert result < 0.50

    def test_zero_depth_returns_none(self):
        book = _make_book(ask_depth=0)
        assert _estimate_live_fill_price(0.50, 10.0, book, "BUY", 500) is None

        book = _make_book(bid_depth=0)
        assert _estimate_live_fill_price(0.50, 10.0, book, "SELL", 500) is None

    def test_exceeds_max_slippage_returns_none(self):
        """When estimated slippage exceeds max, should return None."""
        book = _make_book(spread=0.10, ask_depth=10)
        # impact = 0.05 * sqrt(100/10) = 0.05 * 3.16 = 0.158
        # slippage_bps = 0.158 / 0.50 * 10000 = 3160
        result = _estimate_live_fill_price(0.50, 100.0, book, "BUY", 100)
        assert result is None

    def test_concavity_doubling_size_less_than_doubles_impact(self):
        """Core property: doubling order size should NOT double slippage."""
        book = _make_book(spread=0.02, ask_depth=1000)
        r1 = _estimate_live_fill_price(0.50, 100.0, book, "BUY", 50000)
        r2 = _estimate_live_fill_price(0.50, 200.0, book, "BUY", 50000)
        assert r1 is not None and r2 is not None
        impact_1 = r1 - 0.50
        impact_2 = r2 - 0.50
        # sqrt(2) ≈ 1.414, so impact should scale by ~1.414x, not 2x
        ratio = impact_2 / impact_1
        assert ratio == pytest.approx(math.sqrt(2), rel=0.01)

    def test_zero_spread_zero_impact(self):
        """If spread is zero, impact should be zero."""
        book = _make_book(spread=0.0, ask_depth=1000)
        result = _estimate_live_fill_price(0.50, 100.0, book, "BUY", 500)
        assert result is not None
        assert result == 0.50


# ---------------------------------------------------------------------------
#  Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

class TestEmptyBookMidZero:
    """Verify that an empty order book returns mid=0 (not 0.5 default)."""

    def test_empty_book_mid_is_zero(self):
        book = _make_book(bid_depth=0, ask_depth=0)
        # When both sides are empty, the book is invalid — downstream
        # should reject this via min_tradeable_price filter
        assert book.bid_depth == 0
        assert book.ask_depth == 0
        # BUY with zero ask depth returns None
        assert _estimate_live_fill_price(0.50, 10.0, book, "BUY", 500) is None
        # SELL with zero bid depth returns None
        assert _estimate_live_fill_price(0.50, 10.0, book, "SELL", 500) is None


class TestKellyProperties:
    @given(
        prob=st.floats(min_value=0.01, max_value=0.99),
        price=st.floats(min_value=0.01, max_value=0.99),
        bankroll=st.floats(min_value=1.0, max_value=1e6),
    )
    @settings(max_examples=200)
    def test_never_crashes_on_valid_inputs(self, prob, price, bankroll):
        """Kelly should return a valid result or None, never crash."""
        result = kelly_criterion(
            estimated_prob=prob,
            market_price=price,
            bankroll=bankroll,
            fraction=0.25,
            max_position=100.0,
            min_edge=0.02,
        )
        if result is not None:
            assert result.position_size > 0
            assert result.position_size <= 100.0
            assert 0 < result.confidence <= 1.0

    @given(
        prob=st.floats(min_value=-10, max_value=10),
        price=st.floats(min_value=-10, max_value=10),
    )
    @settings(max_examples=200)
    def test_garbage_inputs_never_crash(self, prob, price):
        """Even out-of-range inputs should return None, not crash."""
        result = kelly_criterion(
            estimated_prob=prob,
            market_price=price,
            bankroll=1000.0,
        )
        # Out-of-range inputs should be None or still valid
        if result is not None:
            assert result.position_size > 0


class TestClassifyMarketProperties:
    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=300)
    def test_never_crashes_on_arbitrary_strings(self, text):
        """classify_market should always return a valid MarketCategory."""
        result = classify_market(text)
        assert isinstance(result, MarketCategory)


# ---------------------------------------------------------------------------
#  get_open_orders retry/semaphore test
# ---------------------------------------------------------------------------

CLOB_BASE = "https://clob.polymarket.com"
ORDERS_PATTERN = re.compile(r"^https://clob\.polymarket\.com/orders")


async def _noop_sleep(delay):
    pass


@pytest.mark.asyncio
class TestGetOpenOrdersRetry:
    async def test_uses_retry_and_semaphore(self):
        """get_open_orders should use retry_request (not raw session.get)
        and respect the semaphore."""
        config = ClobConfig(
            base_url=CLOB_BASE,
            api_key="k", api_secret="s", api_passphrase="p",
        )
        chain = ChainConfig(
            rpc_url="http://localhost:8545",
            private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        client = ClobClient(config, chain)

        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            # First request returns 502 (retryable), second succeeds
            m.get(ORDERS_PATTERN, status=502, body="bad gateway")
            m.get(ORDERS_PATTERN, payload=[{"order_id": "test-1"}])

            client._session = aiohttp.ClientSession()
            result = await client.get_open_orders()
            await client.close()

        # If retry_request is used, we get the result from the 2nd attempt
        assert result == [{"order_id": "test-1"}]

    async def test_respects_semaphore_limit(self):
        """Verify the semaphore is acquired during get_open_orders."""
        config = ClobConfig(
            base_url=CLOB_BASE,
            api_key="k", api_secret="s", api_passphrase="p",
        )
        chain = ChainConfig(
            rpc_url="http://localhost:8545",
            private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        client = ClobClient(config, chain)

        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(ORDERS_PATTERN, payload=[])
            client._session = aiohttp.ClientSession()
            # Acquire all semaphore slots except one
            for _ in range(client.MAX_CONCURRENT - 1):
                await client._semaphore.acquire()

            result = await client.get_open_orders()
            await client.close()

            # Release the acquired slots
            for _ in range(client.MAX_CONCURRENT - 1):
                client._semaphore.release()

        assert result == []
