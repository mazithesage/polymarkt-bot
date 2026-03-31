"""
Market scanning and classification module.
Sourced from: realfishsam/prediction-market-arbitrage-bot (market discovery),
              demone456/kalshi-polymarket-bot (dual-platform scanning),
              ThinkEnigmatic/polymarket-bot-arena (market classification),
              Polymarket/poly-market-maker (Gamma API integration).

Handles: async market discovery via Gamma API, classification of markets
by category, filtering for tradeable markets, and edge detection.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from config import ClobConfig


class MarketCategory(str, Enum):
    CRYPTO = "crypto"
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    FINANCE = "finance"
    OTHER = "other"


CATEGORY_KEYWORDS = {
    MarketCategory.CRYPTO: [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "token", "solana",
        "blockchain", "defi", "nft",
    ],
    MarketCategory.POLITICS: [
        "election", "president", "congress", "senate", "vote", "democrat",
        "republican", "trump", "biden", "political", "governor",
    ],
    MarketCategory.SPORTS: [
        "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
        "baseball", "hockey", "championship", "super bowl", "world cup",
    ],
    MarketCategory.ENTERTAINMENT: [
        "oscar", "grammy", "emmy", "movie", "film", "album", "tv show",
        "streaming", "box office",
    ],
    MarketCategory.SCIENCE: [
        "nasa", "space", "climate", "vaccine", "fda", "drug", "trial",
        "scientific", "discovery",
    ],
    MarketCategory.FINANCE: [
        "fed", "interest rate", "inflation", "gdp", "unemployment",
        "stock", "s&p", "dow", "nasdaq", "treasury",
    ],
}


@dataclass
class Market:
    condition_id: str
    question: str
    description: str
    category: MarketCategory
    tokens: List[Dict]  # [{"token_id": ..., "outcome": "Yes"}, ...]
    end_date: str
    active: bool
    volume: float
    liquidity: float
    neg_risk: bool

    @property
    def token_ids(self) -> List[str]:
        return [t["token_id"] for t in self.tokens]

    @property
    def yes_token_id(self) -> str:
        for t in self.tokens:
            if t.get("outcome", "").lower() == "yes":
                return t["token_id"]
        return self.tokens[0]["token_id"] if self.tokens else ""

    @property
    def no_token_id(self) -> str:
        for t in self.tokens:
            if t.get("outcome", "").lower() == "no":
                return t["token_id"]
        return self.tokens[1]["token_id"] if len(self.tokens) > 1 else ""


def classify_market(question: str, description: str = "") -> MarketCategory:
    """Classify a market by its question text using keyword matching."""
    text = (question + " " + description).lower()
    scores: dict[MarketCategory, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[category] = score
    if not scores:
        return MarketCategory.OTHER
    return max(scores, key=scores.get)


def _parse_market(raw: dict) -> Market:
    """Parse a raw Gamma API market response into a Market object."""
    tokens = []
    for t in raw.get("tokens", []):
        tokens.append({
            "token_id": t.get("token_id", ""),
            "outcome": t.get("outcome", ""),
        })
    question = raw.get("question", "")
    description = raw.get("description", "")
    return Market(
        condition_id=raw.get("condition_id", ""),
        question=question,
        description=description,
        category=classify_market(question, description),
        tokens=tokens,
        end_date=raw.get("end_date_iso", ""),
        active=raw.get("active", False) and not raw.get("closed", True),
        volume=float(raw.get("volume", 0)),
        liquidity=float(raw.get("liquidity", 0)),
        neg_risk=raw.get("neg_risk", False),
    )


class MarketScanner:
    """Async scanner that discovers and classifies Polymarket markets."""

    def __init__(self, clob_config: ClobConfig):
        self.gamma_url = clob_config.gamma_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_active_markets(
        self, limit: int = 100, min_liquidity: float = 0.0
    ) -> List[Market]:
        """Fetch active markets from Gamma API."""
        session = await self._get_session()
        params = {
            "active": "true",
            "closed": "false",
            "limit": str(limit),
            "order": "volume",
            "ascending": "false",
        }
        async with session.get(
            f"{self.gamma_url}/markets", params=params
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()

        markets = [_parse_market(m) for m in data]
        if min_liquidity > 0:
            markets = [m for m in markets if m.liquidity >= min_liquidity]
        return markets

    async def fetch_market_by_condition(self, condition_id: str) -> Optional[Market]:
        """Fetch a specific market by condition ID."""
        session = await self._get_session()
        async with session.get(
            f"{self.gamma_url}/markets/{condition_id}"
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return _parse_market(data)

    async def scan_and_classify(
        self, limit: int = 100, min_liquidity: float = 0.0,
        categories: Optional[List[MarketCategory]] = None,
    ) -> List[Market]:
        """Scan markets, classify them, and optionally filter by category."""
        markets = await self.fetch_active_markets(limit, min_liquidity)
        if categories:
            markets = [m for m in markets if m.category in categories]
        return markets
