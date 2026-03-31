"""
Kelly criterion and position sizing module.
Sourced from: Polymarket/poly-market-maker (position sizing logic),
              realfishsam/prediction-market-arbitrage-bot (edge calculation),
              demone456/kalshi-polymarket-bot (risk management).

Implements fractional Kelly sizing with position limits and edge detection.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KellyResult:
    edge: float
    kelly_fraction: float
    position_size: float
    confidence: float
    side: str  # "BUY" or "SELL"
    token_choice: str  # "YES" or "NO"


def kelly_criterion(
    estimated_prob: float,
    market_price: float,
    bankroll: float,
    fraction: float = 0.25,
    max_position: float = 100.0,
    min_edge: float = 0.02,
) -> Optional[KellyResult]:
    """
    Calculate fractional Kelly bet size for a binary market.

    The Kelly criterion for a binary bet is:
        f* = (p * b - q) / b
    where:
        p = estimated probability of winning
        b = odds received (net odds: payout/stake - 1)
        q = 1 - p

    We apply a fraction (e.g., 0.25 = quarter Kelly) for safety.

    Args:
        estimated_prob: Our estimated true probability (0-1)
        market_price: Current market price / implied probability (0-1)
        bankroll: Available capital in USDC
        fraction: Kelly fraction (0.25 = quarter Kelly)
        max_position: Maximum position size in USDC
        min_edge: Minimum edge required to trade

    Returns:
        KellyResult or None if no edge detected.
    """
    if not (0 < estimated_prob < 1) or not (0 < market_price < 1):
        return None

    # Determine if we should buy YES or NO token
    # Buy YES if our estimated prob > market price
    # Buy NO if our estimated prob < market price
    if estimated_prob > market_price:
        # Buy YES token at market_price
        p = estimated_prob
        price = market_price
        side = "BUY"
        token_choice = "YES"
    else:
        # Buy NO token at (1 - market_price)
        p = 1 - estimated_prob
        price = 1 - market_price
        side = "BUY"
        token_choice = "NO"

    if price <= 0:
        return None

    edge = p - price
    if edge < min_edge:
        return None

    # Kelly formula: f* = (p * (1/price - 1) - (1-p)) / (1/price - 1)
    # Simplified: f* = (p - price) / (1 - price)
    if price >= 1.0:
        return None

    kelly_full = (p - price) / (1.0 - price)
    kelly_full = max(0.0, min(1.0, kelly_full))

    kelly_sized = kelly_full * fraction
    position = min(kelly_sized * bankroll, max_position)

    if position <= 0:
        return None

    return KellyResult(
        edge=edge,
        kelly_fraction=kelly_sized,
        position_size=round(position, 2),
        confidence=p,
        side=side,
        token_choice=token_choice,
    )


def multi_outcome_kelly(
    probabilities: List[float],
    market_prices: List[float],
    bankroll: float,
    fraction: float = 0.25,
    max_position: float = 100.0,
    min_edge: float = 0.02,
) -> List[KellyResult]:
    """
    Kelly sizing across multiple outcomes in a single market.
    Used for markets with >2 outcomes (e.g., "Who will win?").
    Finds the best edge among all outcomes.
    """
    results = []
    for i, (prob, price) in enumerate(zip(probabilities, market_prices)):
        result = kelly_criterion(prob, price, bankroll, fraction, max_position, min_edge)
        if result is not None:
            results.append(result)
    # Sort by edge, highest first
    results.sort(key=lambda r: r.edge, reverse=True)
    return results


def check_position_limits(
    current_exposure: float,
    new_position: float,
    max_total_exposure: float,
) -> float:
    """
    Enforce total exposure limits. Returns the adjusted position size.
    """
    remaining = max(0.0, max_total_exposure - current_exposure)
    return min(new_position, remaining)
