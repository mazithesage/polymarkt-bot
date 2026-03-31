"""Tests for market scanning and classification logic."""

import pytest

from market_scanner import (
    Market,
    MarketCategory,
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
