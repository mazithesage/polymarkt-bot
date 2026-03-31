"""Tests for market scanning and classification logic."""

import re
from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

from config import ClobConfig
from market_scanner import (
    Market,
    MarketCategory,
    MarketScanner,
    _parse_market,
    classify_market,
)


class TestClassifyMarket:
    def test_crypto_btc(self):
        assert classify_market("Will Bitcoin hit $100k?") == MarketCategory.CRYPTO

    def test_crypto_eth(self):
        assert classify_market("Ethereum price above $5000?") == MarketCategory.CRYPTO

    def test_politics_election(self):
        assert classify_market("Will Trump win the 2024 election?") == MarketCategory.POLITICS

    def test_politics_congress(self):
        assert classify_market("Will Democrats control Congress?") == MarketCategory.POLITICS

    def test_sports_nfl(self):
        assert classify_market("NFL Super Bowl winner?") == MarketCategory.SPORTS

    def test_sports_basketball(self):
        assert classify_market("NBA Championship winner?") == MarketCategory.SPORTS

    def test_entertainment(self):
        assert classify_market("Best Picture Oscar winner?") == MarketCategory.ENTERTAINMENT

    def test_science(self):
        assert classify_market("Will NASA land on Mars?") == MarketCategory.SCIENCE

    def test_finance(self):
        assert classify_market("Will the Fed raise interest rates?") == MarketCategory.FINANCE

    def test_other_unclassifiable(self):
        assert classify_market("Will it rain tomorrow?") == MarketCategory.OTHER

    def test_description_helps_classify(self):
        cat = classify_market(
            "Price above threshold?",
            description="This market is about Bitcoin cryptocurrency price.",
        )
        assert cat == MarketCategory.CRYPTO

    def test_multiple_keywords_strongest_wins(self):
        # "bitcoin btc crypto" has 3 crypto keywords
        cat = classify_market("Bitcoin BTC crypto price vs Fed interest rate")
        assert cat == MarketCategory.CRYPTO

    def test_empty_question(self):
        assert classify_market("") == MarketCategory.OTHER

    def test_case_insensitive(self):
        assert classify_market("BITCOIN price prediction") == MarketCategory.CRYPTO


class TestParseMarket:
    def test_parse_basic_market(self):
        raw = {
            "condition_id": "0xabc123",
            "question": "Will BTC hit 100k?",
            "description": "Bitcoin price market",
            "tokens": [
                {"token_id": "111", "outcome": "Yes"},
                {"token_id": "222", "outcome": "No"},
            ],
            "end_date_iso": "2025-12-31",
            "active": True,
            "closed": False,
            "volume": 50000.0,
            "liquidity": 10000.0,
            "neg_risk": False,
        }
        market = _parse_market(raw)
        assert market.condition_id == "0xabc123"
        assert market.question == "Will BTC hit 100k?"
        assert market.category == MarketCategory.CRYPTO
        assert market.active is True
        assert market.volume == 50000.0
        assert market.yes_token_id == "111"
        assert market.no_token_id == "222"

    def test_parse_inactive_market(self):
        raw = {
            "condition_id": "0xdef",
            "question": "Test",
            "description": "",
            "tokens": [],
            "end_date_iso": "",
            "active": False,
            "closed": True,
            "volume": 0,
            "liquidity": 0,
            "neg_risk": False,
        }
        market = _parse_market(raw)
        assert market.active is False

    def test_parse_neg_risk_market(self):
        raw = {
            "condition_id": "0x789",
            "question": "Election market",
            "description": "Political election",
            "tokens": [{"token_id": "t1", "outcome": "Yes"}],
            "end_date_iso": "2025-11-05",
            "active": True,
            "closed": False,
            "volume": 100000,
            "liquidity": 50000,
            "neg_risk": True,
        }
        market = _parse_market(raw)
        assert market.neg_risk is True
        assert market.category == MarketCategory.POLITICS


class TestMarketProperties:
    def test_token_ids(self):
        market = Market(
            condition_id="c1",
            question="Test",
            description="",
            category=MarketCategory.OTHER,
            tokens=[
                {"token_id": "t1", "outcome": "Yes"},
                {"token_id": "t2", "outcome": "No"},
            ],
            end_date="",
            active=True,
            volume=0,
            liquidity=0,
            neg_risk=False,
        )
        assert market.token_ids == ["t1", "t2"]
        assert market.yes_token_id == "t1"
        assert market.no_token_id == "t2"

    def test_empty_tokens(self):
        market = Market(
            condition_id="c2",
            question="Test",
            description="",
            category=MarketCategory.OTHER,
            tokens=[],
            end_date="",
            active=True,
            volume=0,
            liquidity=0,
            neg_risk=False,
        )
        assert market.yes_token_id == ""
        assert market.no_token_id == ""


def _make_raw_market(i: int) -> dict:
    """Helper to create a raw market dict for pagination tests."""
    return {
        "condition_id": f"cond-{i}",
        "question": f"Market {i}?",
        "description": "",
        "tokens": [
            {"token_id": f"t{i}-yes", "outcome": "Yes"},
            {"token_id": f"t{i}-no", "outcome": "No"},
        ],
        "end_date_iso": "2025-12-31",
        "active": True,
        "closed": False,
        "volume": 1000 - i,
        "liquidity": 500.0,
        "neg_risk": False,
    }


GAMMA_URL = "https://gamma-api.polymarket.com"
# Regex pattern to match the markets endpoint with any query params
MARKETS_PATTERN = re.compile(r"^https://gamma-api\.polymarket\.com/markets")


async def _noop_sleep(delay):
    pass


def _make_scanner():
    """Create a scanner with a simple session for test compatibility."""
    config = ClobConfig(gamma_url=GAMMA_URL)
    scanner = MarketScanner(config)
    return scanner


@pytest.mark.asyncio
class TestMarketScannerPagination:
    async def test_single_page(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(MARKETS_PATTERN,
                  payload=[_make_raw_market(i) for i in range(5)])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=5)
            await scanner.close()
        assert len(markets) == 5

    async def test_multi_page_aggregation(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(MARKETS_PATTERN,
                  payload=[_make_raw_market(i) for i in range(100)])
            m.get(MARKETS_PATTERN,
                  payload=[_make_raw_market(100 + i) for i in range(50)])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=200)
            await scanner.close()
        assert len(markets) == 150

    async def test_empty_first_page(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(MARKETS_PATTERN, payload=[])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=100)
            await scanner.close()
        assert len(markets) == 0

    async def test_limit_respected(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(MARKETS_PATTERN,
                  payload=[_make_raw_market(i) for i in range(10)])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=10)
            await scanner.close()
        assert len(markets) <= 10

    async def test_api_failure_mid_pagination(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            # Page 1 succeeds
            m.get(MARKETS_PATTERN,
                  payload=[_make_raw_market(i) for i in range(100)])
            # Page 2 fails — provide enough mocks for retries (max_retries=3 + 1)
            for _ in range(4):
                m.get(MARKETS_PATTERN,
                      exception=aiohttp.ClientConnectionError())
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=200)
            await scanner.close()
        # Should return what was collected from page 1
        assert len(markets) == 100

    async def test_max_pages_cap(self):
        """Scanner should stop after max_pages even if more data exists."""
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            for _ in range(25):
                m.get(MARKETS_PATTERN,
                      payload=[_make_raw_market(i) for i in range(5)])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=10000)
            await scanner.close()
        # max_pages=20, 5 per page = 100 max
        assert len(markets) <= 100
