"""
Backtest: calibration sweep — "how accurate must the OB signal be for the strategy to profit?"

Fetches resolved markets from the Gamma API and, for each signal-accuracy level,
simulates trades where the signal points toward the winning side with
probability = accuracy.  The known resolution provides ground-truth P&L.

This replaces the previous Monte Carlo random-imbalance approach which was
tautological (symmetric draws produced 50/50 wins by construction).

Usage:
    python3 backtest.py                       # full calibration sweep
    python3 backtest.py --markets 100         # fewer markets for speed
    python3 backtest.py --accuracy 0.60       # single accuracy level
    python3 backtest.py --ob-weight 0.10      # higher OB sensitivity
    python3 backtest.py --dry-run             # fetch markets only, no sim
"""

import argparse
import asyncio
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

from kelly import kelly_criterion
from market_scanner import classify_market, MarketCategory

logger = logging.getLogger("backtest")

GAMMA_URL = "https://gamma-api.polymarket.com"

# Simulation grid
ENTRY_PRICES = [0.30, 0.40, 0.50, 0.60, 0.70]
ACCURACY_LEVELS = [0.50, 0.52, 0.55, 0.60, 0.65, 0.70]

# Strategy defaults (mirror BotConfig)
DEFAULT_BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MIN_EDGE = 0.02
MAX_POSITION = 100.0
OB_WEIGHT = 0.05
SPREAD = 0.02  # 2 cents — realistic tight spread
SLIPPAGE_SPREAD_BPS = 50
SLIPPAGE_IMPACT_BPS = 10
DEFAULT_TRIALS = 20  # draws per (market, entry_price) at each accuracy level

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
    total_simulations: int = 0
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
    mid = entry_price

    # Derive bid/ask depth from volume (liquidity proxy) and imbalance
    total_depth = max(liquidity * 0.1, 100.0)
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

    # Apply paper slippage model (same sqrt model as bot.py)
    nominal_price = mid
    size_tokens = result.position_size / nominal_price if nominal_price > 0 else 0
    spread_cost = SLIPPAGE_SPREAD_BPS / 10_000
    depth = ask_depth if result.side == "BUY" else bid_depth
    if depth > 0:
        impact = spread_cost * math.sqrt(size_tokens / depth)
    else:
        impact = spread_cost

    if result.side == "BUY":
        fill_price = min(nominal_price + impact, 0.99)
    else:
        fill_price = max(nominal_price - impact, 0.01)

    # P&L from known resolution
    token = result.token_choice  # "YES" or "NO"
    if token == "YES":
        # Bought YES tokens at fill_price; pays $1 if YES wins, $0 otherwise
        if resolved_yes:
            pnl = (1.0 - fill_price) * result.position_size / fill_price
        else:
            pnl = -result.position_size
    else:
        # Bought NO tokens; effective cost is (1 - fill_price)
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
        imbalance=round(imbalance, 4),
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


def run_calibration_backtest(
    valid_markets: list,
    accuracy: float,
    n_trials: int,
    ob_weight: float,
    min_edge: float,
    rng: random.Random,
) -> BacktestReport:
    """Run backtest at a single accuracy level.

    For each (market, entry_price, trial):
    - With probability=accuracy: imbalance points toward the winning side
    - With probability=1-accuracy: imbalance points the wrong way
    - Imbalance magnitude drawn from Uniform(0.2, 0.3) to clear min_edge
    """
    report = BacktestReport()
    report.markets = len(valid_markets)
    report.total_simulations = report.markets * len(ENTRY_PRICES) * n_trials

    for m, resolved_yes, volume in valid_markets:
        cid = m.get("conditionId", "unknown")
        question = m.get("question", "?")
        category = classify_market(question)

        for entry_price in ENTRY_PRICES:
            for _ in range(n_trials):
                # Draw imbalance magnitude — must be large enough that
                # magnitude * ob_weight >= min_edge, otherwise Kelly never
                # finds an edge.  With default ob_weight=0.05 and min_edge=0.02,
                # the minimum magnitude is 0.40.
                min_magnitude = min_edge / ob_weight if ob_weight > 0 else 0.40
                magnitude = rng.uniform(min_magnitude, min_magnitude + 0.20)

                # With probability=accuracy, signal is correct
                correct = rng.random() < accuracy

                if correct:
                    # Signal points toward the winner
                    imbalance = magnitude if resolved_yes else -magnitude
                else:
                    # Signal points the wrong way
                    imbalance = -magnitude if resolved_yes else magnitude

                trade = simulate_trade(
                    condition_id=cid,
                    question=question,
                    entry_price=entry_price,
                    imbalance=imbalance,
                    resolved_yes=resolved_yes,
                    category=category,
                    liquidity=volume,
                    ob_weight=ob_weight,
                    min_edge=min_edge,
                )
                if trade is not None:
                    report.trades.append(trade)

    return report


def print_calibration_report(
    results: list,
    n_markets: int,
    n_trials: int,
    ob_weight: float,
    min_edge: float,
) -> None:
    """Print the accuracy-sweep calibration table."""
    print("\n" + "=" * 60)
    print("  CALIBRATION BACKTEST  (required signal accuracy)")
    print("=" * 60)
    print(f"  Markets: {n_markets} | Entry prices: {len(ENTRY_PRICES)} | "
          f"Trials per combo: {n_trials}")
    print(f"  ob_weight={ob_weight}, min_edge={min_edge}, "
          f"kelly_fraction={KELLY_FRACTION}, bankroll=${DEFAULT_BANKROLL:.0f}")
    print()
    print(f"  {'Accuracy':>8s}  {'Trades':>6s}  {'Win Rate':>8s}  "
          f"{'Total P&L':>10s}  {'Avg P&L':>8s}  {'Sharpe':>6s}")
    print(f"  {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 6}")

    breakeven_accuracy = None
    min_viable_accuracy = None

    for accuracy, report in results:
        n = len(report.trades)
        wr = report.win_rate
        total = report.total_pnl
        avg = report.avg_pnl
        sharpe = report.sharpe

        print(f"  {accuracy:>7.0%}  {n:>6d}  {wr:>7.1%}  "
              f"${total:>9,.2f}  ${avg:>7.2f}  {sharpe:>6.2f}")

        if breakeven_accuracy is None and sharpe > 0.5:
            breakeven_accuracy = accuracy
        if min_viable_accuracy is None and total > 0:
            min_viable_accuracy = accuracy

    print()
    if breakeven_accuracy is not None:
        print(f"  -> Breakeven accuracy: ~{breakeven_accuracy:.0%} (Sharpe > 0.5)")
    else:
        print(f"  -> Breakeven accuracy: not reached in sweep")
    if min_viable_accuracy is not None:
        print(f"  -> Minimum viable accuracy: ~{min_viable_accuracy:.0%} "
              f"(covers slippage + fees)")
    else:
        print(f"  -> Minimum viable accuracy: not reached in sweep")
    print("=" * 60)


async def run_backtest(
    n_markets: int = 500,
    dry_run: bool = False,
    ob_weight: float = OB_WEIGHT,
    min_edge: float = MIN_EDGE,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = 42,
    single_accuracy: Optional[float] = None,
) -> Optional[list]:
    """Fetch markets and run calibration sweep.

    Returns list of (accuracy, BacktestReport) tuples, or None for dry_run.
    """
    logger.info("Fetching up to %d resolved markets from Gamma API...", n_markets)
    raw_markets = await fetch_resolved_markets(n_markets)
    logger.info("Fetched %d raw markets", len(raw_markets))

    # Filter to cleanly resolved binary markets
    valid_markets = []
    for m in raw_markets:
        resolved_yes = parse_resolution(m)
        if resolved_yes is None:
            continue
        volume = float(m.get("volume", 0) or 0)
        if volume <= 0:
            continue
        valid_markets.append((m, resolved_yes, volume))

    n_valid = len(valid_markets)
    logger.info("Valid resolved markets: %d", n_valid)

    if dry_run:
        yes_count = sum(1 for _, r, _ in valid_markets if r)
        no_count = n_valid - yes_count
        print(f"\n[DRY RUN] {n_valid} valid resolved markets "
              f"(YES: {yes_count}, NO: {no_count})")
        for m, res, vol in valid_markets[:10]:
            q = m.get("question", "?")[:65]
            print(f"  {'YES' if res else 'NO ':>3s} resolved | vol=${vol:>10,.0f} | {q}")
        return None

    accuracies = [single_accuracy] if single_accuracy else ACCURACY_LEVELS
    results = []

    t0 = time.time()
    for accuracy in accuracies:
        rng = random.Random(seed)
        logger.info("Running accuracy=%.0f%% ...", accuracy * 100)
        report = run_calibration_backtest(
            valid_markets, accuracy, n_trials, ob_weight, min_edge, rng,
        )
        results.append((accuracy, report))
        logger.info("  accuracy=%.0f%%: %d trades, P&L=$%.2f, Sharpe=%.2f",
                     accuracy * 100, len(report.trades),
                     report.total_pnl, report.sharpe)

    elapsed = time.time() - t0
    total_trades = sum(len(r.trades) for _, r in results)
    logger.info("Calibration sweep complete: %d total trades in %.1fs",
                total_trades, elapsed)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calibration backtest: required OB signal accuracy for profitability")
    parser.add_argument("--markets", type=int, default=500,
                        help="Number of resolved markets to fetch (default: 500)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Draws per (market, price) at each accuracy (default: {DEFAULT_TRIALS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--accuracy", type=float, default=None,
                        help="Single accuracy level to test (default: sweep all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch markets only, skip simulation")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
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

    results = asyncio.run(run_backtest(
        n_markets=args.markets,
        dry_run=args.dry_run,
        ob_weight=args.ob_weight,
        min_edge=args.min_edge,
        n_trials=args.trials,
        seed=args.seed,
        single_accuracy=args.accuracy,
    ))
    if results is not None:
        # Use market count from first report
        n_markets = results[0][1].markets if results else 0
        print_calibration_report(
            results, n_markets, args.trials,
            args.ob_weight, args.min_edge,
        )


if __name__ == "__main__":
    main()
