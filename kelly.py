"""
Kelly criterion position sizing for binary prediction markets.

Kelly (1956), "A New Interpretation of Information Rate"
f* = edge / odds.  Quarter Kelly because we're not degenerate.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KellyResult:
    edge: float
    kelly_fraction: float
    position_size: float
    confidence: float
    side: str       # "BUY" or "SELL"
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
    Fractional Kelly for a binary market.
    Returns None if no tradeable edge.
    """
    if not (0 < estimated_prob < 1) or not (0 < market_price < 1):
        return None

    # Buy YES if our prob > market, otherwise buy NO
    if estimated_prob > market_price:
        p = estimated_prob
        price = market_price
        side = "BUY"
        token_choice = "YES"
    else:
        p = 1 - estimated_prob
        price = 1 - market_price
        side = "BUY"
        token_choice = "NO"

    if price <= 0:
        return None

    edge = p - price
    if edge < min_edge:
        return None

    # f* = (p - price) / (1 - price)  — simplified binary Kelly
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
    """Kelly across multiple outcomes — find all edges, rank by size."""
    # TODO: parametrize this if we add more exchanges
    results = []
    for prob, price in zip(probabilities, market_prices):
        result = kelly_criterion(prob, price, bankroll, fraction, max_position, min_edge)
        if result is not None:
            results.append(result)
    results.sort(key=lambda r: r.edge, reverse=True)
    return results


def check_position_limits(
    current_exposure: float,
    new_position: float,
    max_total_exposure: float,
) -> float:
    remaining = max(0.0, max_total_exposure - current_exposure)
    return min(new_position, remaining)
