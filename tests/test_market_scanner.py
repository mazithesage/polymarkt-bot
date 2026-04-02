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


# --- classification tests (no async, no network) ---

def test_classify_crypto_btc():
    assert classify_market("Will Bitcoin hit $100k?") == MarketCategory.CRYPTO

def test_classify_crypto_eth():
    assert classify_market("Ethereum price above $5000?") == MarketCategory.CRYPTO

def test_classify_politics():
    assert classify_market("Will Trump win the 2024 election?") == MarketCategory.POLITICS

def test_classify_politics_congress():
    assert classify_market("Will Democrats control Congress?") == MarketCategory.POLITICS

def test_classify_sports_nfl():
    assert classify_market("NFL Super Bowl winner?") == MarketCategory.SPORTS

def test_classify_sports_basketball():
    assert classify_market("NBA Championship winner?") == MarketCategory.SPORTS

def test_classify_entertainment():
    assert classify_market("Best Picture Oscar winner?") == MarketCategory.ENTERTAINMENT

def test_classify_science():
    assert classify_market("Will NASA land on Mars?") == MarketCategory.SCIENCE

def test_classify_finance():
    assert classify_market("Will the Fed raise interest rates?") == MarketCategory.FINANCE

def test_classify_other():
    assert classify_market("Will it rain tomorrow?") == MarketCategory.OTHER

def test_classify_uses_description():
    cat = classify_market(
        "Price above threshold?",
        description="This market is about Bitcoin cryptocurrency price.",
    )
    assert cat == MarketCategory.CRYPTO

def test_classify_strongest_category_wins():
    # "bitcoin btc crypto" has 3 crypto keywords
    assert classify_market("Bitcoin BTC crypto price vs Fed interest rate") == MarketCategory.CRYPTO

def test_classify_empty():
    assert classify_market("") == MarketCategory.OTHER

def test_classify_case_insensitive():
    assert classify_market("BITCOIN price prediction") == MarketCategory.CRYPTO


# --- keyword matching edge cases ---

def test_handles_plurals():
    """Prefix matching should handle 'democrats' → 'democrat'."""
    assert classify_market("Democrats control Congress") == MarketCategory.POLITICS

def test_handles_punctuation():
    """Tokenizer strips punctuation so 'bitcoin?' matches 'bitcoin'."""
    assert classify_market("Bitcoin?") == MarketCategory.CRYPTO

def test_multi_word_phrase():
    """Multi-word phrases like 'super bowl' should match as substrings."""
    assert classify_market("Who will win the Super Bowl this year?") == MarketCategory.SPORTS

def test_no_false_positive_on_substring():
    """Single-word keywords should NOT match inside other words (e.g. 'eth' in 'whether')."""
    result = classify_market("Whether or not it happens")
    assert result == MarketCategory.OTHER


# --- market parsing ---

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
            "active": True, "closed": False,
            "volume": 50000.0, "liquidity": 10000.0, "neg_risk": False,
        }
        market = _parse_market(raw)
        assert market.condition_id == "0xabc123"
        assert market.category == MarketCategory.CRYPTO
        assert market.yes_token_id == "111"
        assert market.no_token_id == "222"

    def test_parse_inactive_market(self):
        raw = {
            "condition_id": "0xdef", "question": "Test", "description": "",
            "tokens": [], "end_date_iso": "",
            "active": False, "closed": True,
            "volume": 0, "liquidity": 0, "neg_risk": False,
        }
        assert _parse_market(raw).active is False

    def test_parse_neg_risk_market(self):
        raw = {
            "condition_id": "0x789", "question": "Election market",
            "description": "Political election",
            "tokens": [{"token_id": "t1", "outcome": "Yes"}],
            "end_date_iso": "2025-11-05",
            "active": True, "closed": False,
            "volume": 100000, "liquidity": 50000, "neg_risk": True,
        }
        market = _parse_market(raw)
        assert market.neg_risk is True
        assert market.category == MarketCategory.POLITICS

    def test_parse_camelcase_gamma_response(self):
        """Gamma list endpoint returns camelCase fields with JSON-string tokens."""
        raw = {
            "conditionId": "0xcamel",
            "question": "Will BTC hit 100k?",
            "description": "Bitcoin price market",
            "clobTokenIds": '["tok-yes-1", "tok-no-1"]',
            "outcomes": '["Yes", "No"]',
            "endDateIso": "2025-12-31",
            "active": True, "closed": False,
            "volume": 75000.0, "liquidity": 20000.0, "negRisk": True,
        }
        market = _parse_market(raw)
        assert market.condition_id == "0xcamel"
        assert market.end_date == "2025-12-31"
        assert market.neg_risk is True
        assert market.yes_token_id == "tok-yes-1"
        assert market.no_token_id == "tok-no-1"
        assert market.category == MarketCategory.CRYPTO

    def test_parse_camelcase_neg_risk_false(self):
        """negRisk=False must not fall through to snake_case fallback."""
        raw = {
            "conditionId": "0xbool",
            "question": "Test",
            "description": "",
            "clobTokenIds": '["t1"]',
            "outcomes": '["Yes"]',
            "endDateIso": "",
            "active": True, "closed": False,
            "volume": 0, "liquidity": 0, "negRisk": False,
        }
        assert _parse_market(raw).neg_risk is False

    def test_parse_malformed_clob_token_ids(self):
        """Malformed clobTokenIds JSON should result in empty tokens."""
        raw = {
            "conditionId": "0xbad",
            "question": "Test",
            "description": "",
            "clobTokenIds": "not-valid-json",
            "outcomes": '["Yes"]',
            "endDateIso": "",
            "active": True, "closed": False,
            "volume": 0, "liquidity": 0, "negRisk": False,
        }
        market = _parse_market(raw)
        assert market.tokens == []
        assert market.yes_token_id == ""


# --- market properties ---

def test_token_ids():
    market = Market(
        condition_id="c1", question="Test", description="",
        category=MarketCategory.OTHER,
        tokens=[{"token_id": "t1", "outcome": "Yes"}, {"token_id": "t2", "outcome": "No"}],
        end_date="", active=True, volume=0, liquidity=0, neg_risk=False,
    )
    assert market.token_ids == ["t1", "t2"]
    assert market.yes_token_id == "t1"
    assert market.no_token_id == "t2"

def test_empty_tokens():
    market = Market(
        condition_id="c2", question="Test", description="",
        category=MarketCategory.OTHER, tokens=[],
        end_date="", active=True, volume=0, liquidity=0, neg_risk=False,
    )
    assert market.yes_token_id == ""
    assert market.no_token_id == ""


# --- pagination (async) ---

def _make_raw_market(i: int) -> dict:
    return {
        "condition_id": f"cond-{i}", "question": f"Market {i}?", "description": "",
        "tokens": [
            {"token_id": f"t{i}-yes", "outcome": "Yes"},
            {"token_id": f"t{i}-no", "outcome": "No"},
        ],
        "end_date_iso": "2025-12-31",
        "active": True, "closed": False,
        "volume": 1000 - i, "liquidity": 500.0, "neg_risk": False,
    }

GAMMA_URL = "https://gamma-api.polymarket.com"
MARKETS_PATTERN = re.compile(r"^https://gamma-api\.polymarket\.com/markets")

async def _noop_sleep(delay):
    pass

def _make_scanner():
    return MarketScanner(ClobConfig(gamma_url=GAMMA_URL))


@pytest.mark.asyncio
class TestMarketScannerPagination:
    async def test_single_page(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(MARKETS_PATTERN, payload=[_make_raw_market(i) for i in range(5)])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=5)
            await scanner.close()
        assert len(markets) == 5

    async def test_multi_page_aggregation(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(MARKETS_PATTERN, payload=[_make_raw_market(i) for i in range(100)])
            m.get(MARKETS_PATTERN, payload=[_make_raw_market(100 + i) for i in range(50)])
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
            m.get(MARKETS_PATTERN, payload=[_make_raw_market(i) for i in range(10)])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=10)
            await scanner.close()
        assert len(markets) <= 10

    async def test_api_failure_mid_pagination(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            m.get(MARKETS_PATTERN, payload=[_make_raw_market(i) for i in range(100)])
            for _ in range(4):
                m.get(MARKETS_PATTERN, exception=aiohttp.ClientConnectionError())
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=200)
            await scanner.close()
        assert len(markets) == 100  # got page 1, page 2 failed

    async def test_max_pages_cap(self):
        scanner = _make_scanner()
        with aioresponses() as m, patch("retry.asyncio.sleep", side_effect=_noop_sleep):
            for _ in range(25):
                m.get(MARKETS_PATTERN, payload=[_make_raw_market(i) for i in range(5)])
            scanner._session = aiohttp.ClientSession()
            markets = await scanner.fetch_active_markets(limit=10000)
            await scanner.close()
        assert len(markets) <= 100  # max_pages=20, 5/page
