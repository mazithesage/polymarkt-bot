"""Market discovery and classification via the Gamma API."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from config import ClobConfig
from retry import retry_request

logger = logging.getLogger("polymarket-bot")


class MarketCategory(str, Enum):
    CRYPTO = "crypto"
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    FINANCE = "finance"
    OTHER = "other"


# Keyword corpus per category — multi-word phrases are matched as substrings,
# single words are matched as tokens.
CATEGORY_KEYWORDS: Dict[MarketCategory, List[str]] = {
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
    tokens: List[Dict]
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


_TOKEN_RE = re.compile(r"[a-z0-9&]+(?:'[a-z]+)?")


def classify_market(question: str, description: str = "") -> MarketCategory:
    """Classify market text by keyword hit count.

    Single-word keywords use prefix matching against tokenized text
    (handles plurals/possessives).  Multi-word phrases use substring
    matching.  Category with the most hits wins.
    """
    text = (question + " " + description).lower()
    text_tokens = _TOKEN_RE.findall(text)

    best_category = MarketCategory.OTHER
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if " " in kw:
                if kw in text:
                    score += 1
            else:
                if any(tok.startswith(kw) for tok in text_tokens):
                    score += 1
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def _parse_market(raw: dict) -> Market:
    # Build tokens list — handle both nested array and JSON-string formats.
    tokens = []
    raw_tokens = raw.get("tokens", [])
    if raw_tokens:
        # Individual market endpoint format: tokens is a list of dicts
        for t in raw_tokens:
            tokens.append({
                "token_id": t.get("token_id", ""),
                "outcome": t.get("outcome", ""),
            })
    else:
        # List endpoint format: clobTokenIds + outcomes as JSON strings
        try:
            token_ids = json.loads(raw.get("clobTokenIds", "[]"))
            outcomes = json.loads(raw.get("outcomes", "[]"))
            for tid, outcome in zip(token_ids, outcomes):
                tokens.append({"token_id": tid, "outcome": outcome})
        except (json.JSONDecodeError, TypeError):
            pass

    question = raw.get("question", "")
    description = raw.get("description", "")

    # Support both camelCase (Gamma API) and snake_case field names.
    condition_id = raw.get("conditionId") or raw.get("condition_id", "")
    end_date = raw.get("endDateIso") or raw.get("end_date_iso", "")

    neg_risk_val = raw.get("negRisk")
    if neg_risk_val is None:
        neg_risk_val = raw.get("neg_risk", False)

    return Market(
        condition_id=condition_id,
        question=question,
        description=description,
        category=classify_market(question, description),
        tokens=tokens,
        end_date=end_date,
        active=raw.get("active", False) and not raw.get("closed", True),
        volume=float(raw.get("volume", 0)),
        liquidity=float(raw.get("liquidity", 0)),
        neg_risk=neg_risk_val,
    )


class MarketScanner:

    def __init__(self, clob_config: ClobConfig):
        self.gamma_url = clob_config.gamma_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(5)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_active_markets(
        self, limit: int = 100, min_liquidity: float = 0.0
    ) -> List[Market]:
        session = await self._get_session()
        all_markets: List[Market] = []
        page_size = min(limit, 100)
        max_pages = 20
        offset = 0

        for page in range(max_pages):
            remaining = limit - len(all_markets)
            if remaining <= 0:
                break
            fetch_size = min(page_size, remaining)

            params = {
                "active": "true",
                "closed": "false",
                "limit": str(fetch_size),
                "offset": str(offset),
                "order": "liquidity",
                "ascending": "false",
            }
            try:
                async with self._semaphore:
                    resp = await retry_request(
                        session, "GET", f"{self.gamma_url}/markets", params=params,
                    )
                    if resp.status != 200:
                        logger.warning(f"Market fetch returned {resp.status} on page {page}")
                        break
                    data = await resp.json()
            except Exception as e:
                logger.warning(
                    f"Pagination failed on page {page}: {e}, "
                    f"returning {len(all_markets)} markets collected so far"
                )
                break

            if not data:
                break

            page_markets = [_parse_market(m) for m in data]
            all_markets.extend(page_markets)
            logger.debug(
                f"Page {page}: fetched {len(page_markets)} markets "
                f"(total: {len(all_markets)})"
            )

            if len(data) < fetch_size:
                break
            offset += len(data)

        if min_liquidity > 0:
            all_markets = [m for m in all_markets if m.liquidity >= min_liquidity]
        return all_markets

    async def fetch_market_by_condition(self, condition_id: str) -> Optional[Market]:
        session = await self._get_session()
        async with session.get(f"{self.gamma_url}/markets/{condition_id}") as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return _parse_market(data)

    async def scan_and_classify(
        self, limit: int = 100, min_liquidity: float = 0.0,
        categories: Optional[List[MarketCategory]] = None,
    ) -> List[Market]:
        markets = await self.fetch_active_markets(limit, min_liquidity)
        if categories:
            markets = [m for m in markets if m.category in categories]
        return markets
