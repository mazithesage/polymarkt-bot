"""
Backtest: validate the OB-imbalance → Kelly strategy against resolved markets.

Fetches ~500 resolved markets from the Gamma API and simulates trades at
multiple entry prices and synthetic OB imbalance levels.  The known resolution
(Yes=1 or No=1) provides ground-truth P&L.

Usage:
    python3 backtest.py                  # full backtest (~500 markets)
    python3 backtest.py --markets 50     # quick test with fewer markets
    python3 backtest.py --dry-run        # fetch markets only, no simulation
"""

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

from kelly import kelly_criterion

logger = logging.getLogger("backtest")

GAMMA_URL = "https://gamma-api.polymarket.com"

# Simulation grid
ENTRY_PRICES = [0.30, 0.40, 0.50, 0.60, 0.70]
IMBALANCE_LEVELS = [-0.30, -0.10, 0.00, 0.10, 0.30]

# Strategy defaults (mirror BotConfig)
DEFAULT_BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MIN_EDGE = 0.02
MAX_POSITION = 100.0
OB_WEIGHT = 0.05
SPREAD = 0.02  # 2 cents — realistic tight spread
SLIPPAGE_SPREAD_BPS = 50
SLIPPAGE_IMPACT_BPS = 10

# Category keywords (simplified version of market_scanner.py)
CATEGORY_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "blockchain"],
    "politics": ["election", "president", "congress", "trump", "biden", "vote", "democrat", "republican"],
    "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "super bowl"],
    "entertainment": ["oscar", "grammy", "emmy", "movie", "film", "tv show"],
    "science": ["nasa", "space", "climate", "vaccine", "fda"],
    "finance": ["fed", "interest rate", "inflation", "gdp", "stock", "s&p", "nasdaq"],
}


@dataclass
class BacktestTrade:
    condition_id: str
    question: str
    entry_price: float
    fill_price: float
    side: str  # "YES" or "NO"
    size: float  # USDC notional
    resolved_yes: bool
    pnl: float
    imbalance: float
    category: str


@dataclass
class BacktestReport:
    markets: int = 0
    trades: List[BacktestTrade] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def sharpe(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        pnls = [t.pnl for t in self.trades]
        mean = sum(pnls) / len(pnls)
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return 0.0
        return mean / std

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        worst_dd = 0.0
        for t in self.trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > worst_dd:
                worst_dd = dd
        return worst_dd


def classify_market(question: str) -> str:
    q_lower = question.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return category
    return "other"


def ob_imbalance_prob(
    mid: float,
    bid_depth: float,
    ask_depth: float,
    weight: float = OB_WEIGHT,
) -> float:
    """Estimate probability from synthetic OB imbalance — mirrors bot._ob_imbalance_probability."""
    total = bid_depth + ask_depth
    if total == 0:
        return mid
    imbalance = (bid_depth - ask_depth) / total
    return max(0.01, min(0.99, mid + imbalance * weight))


def simulate_trade(
    condition_id: str,
    question: str,
    entry_price: float,
    imbalance: float,
    resolved_yes: bool,
    category: str,
    liquidity: float,
    bankroll: float = DEFAULT_BANKROLL,
    ob_weight: float = OB_WEIGHT,
    min_edge: float = MIN_EDGE,
) -> Optional[BacktestTrade]:
    """Simulate a single trade at a given entry price and synthetic imbalance.

    Returns None if Kelly finds no edge (the strategy correctly passes).
    """
    # Build synthetic order book
    half_spread = SPREAD / 2
    best_bid = entry_price - half_spread
    best_ask = entry_price + half_spread
    mid = entry_price

    # Derive bid/ask depth from liquidity and imbalance
    # Total depth is a fraction of the market's reported liquidity
    total_depth = max(liquidity * 0.1, 100.0)  # at least $100 each side
    # imbalance = (bid - ask) / (bid + ask)  →  bid = total*(1+imb)/2
    bid_depth = total_depth * (1 + imbalance) / 2
    ask_depth = total_depth * (1 - imbalance) / 2

    # OB imbalance probability estimate
    estimated_prob = ob_imbalance_prob(mid, bid_depth, ask_depth, weight=ob_weight)

    # Kelly sizing
    result = kelly_criterion(
        estimated_prob=estimated_prob,
        market_price=mid,
        bankroll=bankroll,
        fraction=KELLY_FRACTION,
        max_position=MAX_POSITION,
        min_edge=min_edge,
    )
    if result is None:
        return None  # no edge — strategy correctly passes

    # Apply paper slippage model (same as bot.py)
    nominal_price = mid
    size_tokens = result.position_size / nominal_price if nominal_price > 0 else 0
    spread_cost = SLIPPAGE_SPREAD_BPS / 10_000
    impact = SLIPPAGE_IMPACT_BPS / 10_000 * size_tokens

    if result.side == "BUY":
        fill_price = min(nominal_price + spread_cost + impact, 0.99)
    else:
        fill_price = max(nominal_price - spread_cost - impact, 0.01)

    # P&L from known resolution
    token = result.token_choice  # "YES" or "NO"
    if token == "YES":
        # Bought YES tokens at fill_price; pays $1 if YES wins, $0 otherwise
        if resolved_yes:
            pnl = (1.0 - fill_price) * result.position_size / fill_price
        else:
            pnl = -result.position_size
    else:
        # Bought NO tokens at (1 - fill_price equivalent)
        no_fill = 1.0 - fill_price
        if not resolved_yes:
            pnl = (1.0 - no_fill) * result.position_size / no_fill
        else:
            pnl = -result.position_size

    return BacktestTrade(
        condition_id=condition_id,
        question=question,
        entry_price=entry_price,
        fill_price=round(fill_price, 6),
        side=token,
        size=result.position_size,
        resolved_yes=resolved_yes,
        pnl=round(pnl, 4),
        imbalance=imbalance,
        category=category,
    )


async def fetch_resolved_markets(n: int = 500, timeout: float = 30.0) -> List[dict]:
    """Paginate Gamma API for high-volume resolved (closed) markets."""
    markets = []
    page_size = 100
    offset = 0

    async with aiohttp.ClientSession() as session:
        while len(markets) < n:
            params = {
                "closed": "true",
                "limit": str(page_size),
                "order": "volume",
                "ascending": "false",
                "offset": str(offset),
            }
            url = f"{GAMMA_URL}/markets"
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status != 200:
                        logger.warning("Gamma API returned %d at offset %d", resp.status, offset)
                        break
                    data = await resp.json()
                    if not data:
                        break
                    markets.extend(data)
                    offset += page_size
                    logger.info("Fetched %d markets so far (offset=%d)", len(markets), offset)
            except Exception as e:
                logger.error("Gamma API error at offset %d: %s", offset, e)
                break

    return markets[:n]


def parse_resolution(market: dict) -> Optional[bool]:
    """Determine if YES won from outcomePrices.

    outcomePrices is a JSON-encoded list like '["1","0"]' where index 0 = YES.
    Returns True if YES resolved to 1, False if NO resolved to 1, None if ambiguous.
    """
    raw = market.get("outcomePrices")
    if not raw:
        return None
    try:
        if isinstance(raw, str):
            prices = json.loads(raw)
        else:
            prices = raw
        if len(prices) < 2:
            return None
        yes_price = float(prices[0])
        no_price = float(prices[1])
        if yes_price == 1.0 and no_price == 0.0:
            return True
        if yes_price == 0.0 and no_price == 1.0:
            return False
        return None  # partially resolved or ambiguous
    except (json.JSONDecodeError, ValueError, IndexError):
        return None


def imbalance_aligns_with_outcome(imbalance: float, resolved_yes: bool) -> Optional[str]:
    """Classify whether the imbalance direction aligned with the actual outcome."""
    if abs(imbalance) < 0.05:
        return "neutral"
    # Positive imbalance (bid > ask) → market expects YES
    if imbalance > 0 and resolved_yes:
        return "aligns"
    if imbalance < 0 and not resolved_yes:
        return "aligns"
    return "opposes"


def print_report(report: BacktestReport) -> None:
    trades = report.trades
    n = len(trades)

    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Markets:       {report.markets}")
    print(f"  Trades:        {n} (out of {report.markets * len(ENTRY_PRICES) * len(IMBALANCE_LEVELS)} simulations)")

    if n == 0:
        print("  No trades generated — Kelly found no edge in any simulation.")
        max_imb = max(abs(i) for i in IMBALANCE_LEVELS)
        max_adj = max_imb * OB_WEIGHT
        print(f"\n  Diagnostic: max |imbalance|={max_imb:.2f} × ob_weight={OB_WEIGHT} "
              f"= {max_adj:.4f} edge")
        print(f"  But min_edge={MIN_EDGE} — OB imbalance alone cannot clear this threshold.")
        print(f"\n  Try:  python3 backtest.py --ob-weight 0.10")
        print(f"    or: python3 backtest.py --min-edge 0.01")
        print("=" * 55)
        return

    print(f"  Win Rate:      {report.win_rate:.1%}")
    print(f"  Total P&L:     ${report.total_pnl:,.2f}")
    print(f"  Avg P&L:       ${report.avg_pnl:.2f} per trade")
    print(f"  Sharpe:        {report.sharpe:.2f}")
    print(f"  Max Drawdown:  ${report.max_drawdown:,.2f}")

    # --- By Imbalance Direction ---
    print(f"\n{'--- By Imbalance Direction ---':^55}")
    buckets = {"aligns": [], "opposes": [], "neutral": []}
    for t in trades:
        label = imbalance_aligns_with_outcome(t.imbalance, t.resolved_yes)
        if label:
            buckets[label].append(t)

    for label in ("aligns", "opposes", "neutral"):
        b = buckets[label]
        if b:
            wr = sum(1 for t in b if t.pnl > 0) / len(b)
            avg = sum(t.pnl for t in b) / len(b)
            print(f"  {label:>20s}:  {len(b):>4d} trades, {wr:.1%} win rate, avg P&L ${avg:.2f}")
        else:
            print(f"  {label:>20s}:  0 trades")

    # --- By Entry Price ---
    print(f"\n{'--- By Entry Price ---':^55}")
    for ep in ENTRY_PRICES:
        subset = [t for t in trades if abs(t.entry_price - ep) < 0.01]
        if subset:
            total = sum(t.pnl for t in subset)
            wr = sum(1 for t in subset if t.pnl > 0) / len(subset)
            print(f"  {ep:.2f}:  {len(subset):>4d} trades, ${total:>9,.2f} P&L, {wr:.1%} win rate")

    # --- By Category ---
    print(f"\n{'--- By Category ---':^55}")
    cats = sorted(set(t.category for t in trades))
    for cat in cats:
        subset = [t for t in trades if t.category == cat]
        total = sum(t.pnl for t in subset)
        wr = sum(1 for t in subset if t.pnl > 0) / len(subset)
        print(f"  {cat:>15s}:  {len(subset):>4d} trades, ${total:>9,.2f} P&L, {wr:.1%} win rate")

    print("=" * 55)


async def run_backtest(
    n_markets: int = 500,
    dry_run: bool = False,
    ob_weight: float = OB_WEIGHT,
    min_edge: float = MIN_EDGE,
) -> BacktestReport:
    report = BacktestReport()

    logger.info("Fetching up to %d resolved markets from Gamma API...", n_markets)
    raw_markets = await fetch_resolved_markets(n_markets)
    logger.info("Fetched %d raw markets", len(raw_markets))

    # Filter to cleanly resolved binary markets
    # Resolved markets have liquidity=0 (trading ended), so use volume as
    # the proxy for historical depth in the synthetic order book.
    valid_markets = []
    for m in raw_markets:
        resolved_yes = parse_resolution(m)
        if resolved_yes is None:
            continue
        volume = float(m.get("volume", 0) or 0)
        if volume <= 0:
            continue
        valid_markets.append((m, resolved_yes, volume))

    report.markets = len(valid_markets)
    logger.info("Valid resolved markets: %d", report.markets)

    if dry_run:
        print(f"\n[DRY RUN] {report.markets} valid resolved markets found.")
        for m, res, vol in valid_markets[:10]:
            q = m.get("question", "?")[:70]
            print(f"  {'YES' if res else 'NO ':>3s} resolved | vol=${vol:>10,.0f} | {q}")
        return report

    # Run simulation grid
    t0 = time.time()
    for m, resolved_yes, volume in valid_markets:
        cid = m.get("conditionId", "unknown")
        question = m.get("question", "?")
        category = classify_market(question)

        for entry_price in ENTRY_PRICES:
            for imbalance in IMBALANCE_LEVELS:
                trade = simulate_trade(
                    condition_id=cid,
                    question=question,
                    entry_price=entry_price,
                    imbalance=imbalance,
                    resolved_yes=resolved_yes,
                    category=category,
                    liquidity=volume,  # use historical volume as depth proxy
                    ob_weight=ob_weight,
                    min_edge=min_edge,
                )
                if trade is not None:
                    report.trades.append(trade)

    elapsed = time.time() - t0
    logger.info("Simulation complete: %d trades in %.1fs", len(report.trades), elapsed)
    return report


def print_params(ob_weight: float, min_edge: float) -> None:
    print(f"\n  Parameters: ob_weight={ob_weight}, min_edge={min_edge}, "
          f"kelly_fraction={KELLY_FRACTION}, bankroll=${DEFAULT_BANKROLL:.0f}")
    max_imb = max(abs(i) for i in IMBALANCE_LEVELS)
    max_adj = max_imb * ob_weight
    can_trade = "YES" if max_adj >= min_edge else "NO"
    print(f"  Max edge from OB imbalance: {max_imb:.2f} × {ob_weight} = {max_adj:.4f} "
          f"(>= min_edge {min_edge}? {can_trade})")


def main():
    parser = argparse.ArgumentParser(description="Backtest OB-imbalance + Kelly strategy on resolved markets")
    parser.add_argument("--markets", type=int, default=500, help="Number of resolved markets to fetch (default: 500)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch markets only, skip simulation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--ob-weight", type=float, default=OB_WEIGHT,
                        help=f"OB imbalance weight (default: {OB_WEIGHT}, bot default)")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE,
                        help=f"Minimum edge for Kelly (default: {MIN_EDGE}, bot default)")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print_params(args.ob_weight, args.min_edge)

    report = asyncio.run(run_backtest(
        n_markets=args.markets,
        dry_run=args.dry_run,
        ob_weight=args.ob_weight,
        min_edge=args.min_edge,
    ))
    if not args.dry_run:
        print_report(report)


if __name__ == "__main__":
    main()
