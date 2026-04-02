"""
Main bot — async scan loop, SSVI probability estimation, Kelly sizing, paper/live execution.
"""

import argparse
import asyncio
import dataclasses
import logging
import signal
import sys
import time
import uuid
from typing import Optional

import numpy as np

from clob_client import ClobClient, Order, OrderResult, Side, OrderType
from config import BotConfig, ChainConfig, ClobConfig, load_config, validate_live_config
from kelly import KellyResult, check_position_limits, kelly_criterion
from market_scanner import Market, MarketCategory, MarketScanner
from persistence import PersistenceStore
from ssvi import (
    SSVIParams,
    extract_probability,
    fit_ssvi,
    generate_synthetic_surface,
)

logger = logging.getLogger("polymarket-bot")

# Thresholds for market filtering
MIN_TRADEABLE_PRICE = 0.01
MAX_TRADEABLE_PRICE = 0.99
MAX_SPREAD = 0.10
BANKROLL_MULTIPLIER = 10  # bankroll = 10x max position is a rough heuristic, not portfolio theory


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class PolymarketBot:

    def __init__(
        self,
        clob_config: ClobConfig,
        chain_config: ChainConfig,
        bot_config: BotConfig,
    ):
        self.clob_config = clob_config
        self.chain_config = chain_config
        self.config = bot_config
        self.scanner = MarketScanner(clob_config)
        self.clob = ClobClient(clob_config, chain_config)
        self.store = PersistenceStore(bot_config.db_path)
        self._running = False
        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_breaker_until: float = 0.0

    async def close(self):
        await self.scanner.close()
        await self.clob.close()

    def _trip_circuit_breaker(self) -> None:
        self._circuit_breaker_until = time.time() + self.config.circuit_breaker_cooldown
        logger.warning(
            "Circuit breaker tripped after %d consecutive failures. "
            "Pausing for %ds.",
            self._consecutive_failures,
            self.config.circuit_breaker_cooldown,
        )

    def _reset_circuit_breaker(self) -> None:
        if self._consecutive_failures > 0:
            logger.info("Circuit breaker reset — scan succeeded.")
        self._consecutive_failures = 0
        self._circuit_breaker_until = 0.0

    def _circuit_breaker_active(self) -> bool:
        if self._circuit_breaker_until <= 0:
            return False
        if time.time() >= self._circuit_breaker_until:
            logger.info("Circuit breaker cooldown expired, resuming.")
            self._consecutive_failures = 0
            self._circuit_breaker_until = 0.0
            return False
        return True

    async def scan_once(self) -> dict:
        """Single scan cycle: discover -> evaluate -> size -> execute."""
        if self._circuit_breaker_active():
            remaining = self._circuit_breaker_until - time.time()
            logger.info("Circuit breaker active, %.0fs remaining. Skipping scan.", remaining)
            return {"markets_found": 0, "markets_with_edge": 0,
                    "orders_placed": 0, "errors": ["circuit_breaker_active"]}

        logger.info("Starting scan cycle...")
        summary = {
            "markets_found": 0,
            "markets_with_edge": 0,
            "orders_placed": 0,
            "errors": [],
        }

        try:
            markets = await self.scanner.scan_and_classify(
                limit=self.config.max_markets * 5,
                min_liquidity=100.0,
            )
            summary["markets_found"] = len(markets)
            logger.info(f"Found {len(markets)} active markets")

            current_exposure = self.store.get_total_exposure(self.config.paper_mode)
            bankroll = self.config.max_position_usdc * BANKROLL_MULTIPLIER

            for market in markets[: self.config.max_markets]:
                try:
                    result = await self._evaluate_market(
                        market, bankroll, current_exposure
                    )
                    if result is not None:
                        summary["markets_with_edge"] += 1
                        order_result = await self._execute_trade(market, result)
                        if order_result is not None:
                            summary["orders_placed"] += 1
                            current_exposure += result.position_size
                except Exception as e:
                    msg = f"Error evaluating {market.question[:50]}: {e}"
                    logger.warning(msg, exc_info=True)
                    summary["errors"].append(msg)

            # Scan completed — reset circuit breaker
            self._reset_circuit_breaker()

        except Exception as e:
            msg = f"Scan cycle error: {e}"
            logger.error(msg, exc_info=True)
            summary["errors"].append(msg)
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                self._trip_circuit_breaker()

        self.store.log_scan(
            markets_found=summary["markets_found"],
            markets_with_edge=summary["markets_with_edge"],
            orders_placed=summary["orders_placed"],
            paper_mode=self.config.paper_mode,
        )

        logger.info(
            f"Scan complete: {summary['markets_found']} markets, "
            f"{summary['markets_with_edge']} with edge, "
            f"{summary['orders_placed']} orders placed"
        )
        return summary

    async def _evaluate_market(
        self, market: Market, bankroll: float, current_exposure: float
    ) -> Optional[KellyResult]:
        if not market.yes_token_id:
            return None

        book = await self.clob.get_order_book(market.yes_token_id)
        market_price = book.mid_price

        if market_price <= MIN_TRADEABLE_PRICE or market_price >= MAX_TRADEABLE_PRICE:
            return None
        if book.spread > MAX_SPREAD:
            return None

        estimated_prob = await self._estimate_probability(market, book.mid_price)

        result = kelly_criterion(
            estimated_prob=estimated_prob,
            market_price=market_price,
            bankroll=bankroll,
            fraction=self.config.kelly_fraction,
            max_position=self.config.max_position_usdc,
            min_edge=self.config.min_edge,
        )

        if result is None:
            return None

        adjusted_size = check_position_limits(
            current_exposure, result.position_size,
            self.config.max_position_usdc * 5,
        )
        if adjusted_size <= 0:
            return None

        result.position_size = adjusted_size
        logger.info(
            f"Edge found: {market.question[:60]} | "
            f"edge={result.edge:.4f} kelly={result.kelly_fraction:.4f} "
            f"size=${result.position_size:.2f} side={result.token_choice}"
        )
        return result

    async def _estimate_probability(
        self, market: Market, mid_price: float
    ) -> float:
        # For crypto markets with multiple strikes, try SSVI on real data
        if market.category == MarketCategory.CRYPTO and len(market.tokens) >= 3:
            try:
                return await self._ssvi_probability(market, mid_price)
            except Exception as e:
                logger.debug(f"SSVI failed, using fallback: {e}")

        # Fallback: synthetic surface (paper mode / demo)
        try:
            strikes, ivs = generate_synthetic_surface(
                spot=mid_price, base_vol=0.5, time_to_expiry=1.0 / 365
            )
            params = fit_ssvi(strikes, ivs, forward=mid_price, time_to_expiry=1.0 / 365)
            if params.r_squared < self.config.ssvi_r2_threshold:
                logger.info(
                    f"SSVI R²={params.r_squared:.4f} below threshold "
                    f"{self.config.ssvi_r2_threshold}, using mid-price"
                )
                return mid_price
            prob = extract_probability(
                params, spot=mid_price, forward=mid_price, time_to_expiry=1.0 / 365
            )
            return prob.probability_above
        except Exception:
            return mid_price

    async def _ssvi_probability(self, market: Market, mid_price: float) -> float:
        prices = []
        for token in market.tokens:
            try:
                book = await self.clob.get_order_book(token["token_id"])
                prices.append(book.mid_price)
            except Exception:
                prices.append(0.5)

        if len(prices) < 3:
            return mid_price

        strikes = np.linspace(0.2, 0.8, len(prices))
        ivs = np.array([max(0.1, abs(p - 0.5) * 2 + 0.3) for p in prices])
        params = fit_ssvi(strikes, ivs, forward=0.5, time_to_expiry=1.0 / 365)
        if params.r_squared < self.config.ssvi_r2_threshold:
            logger.info(
                f"Multi-token SSVI R²={params.r_squared:.4f} below threshold, "
                f"falling back to mid-price"
            )
            return mid_price
        prob = extract_probability(params, spot=mid_price, forward=0.5, time_to_expiry=1.0 / 365)
        return prob.probability_above

    async def _execute_trade(
        self, market: Market, kelly_result: KellyResult
    ) -> Optional[OrderResult]:
        token_id = (
            market.yes_token_id
            if kelly_result.token_choice == "YES"
            else market.no_token_id
        )
        if not token_id:
            return None

        book = await self.clob.get_order_book(token_id)
        if kelly_result.side == "BUY":
            price = round(min(book.best_ask, book.mid_price + 0.005), 2)
        else:
            price = round(max(book.best_bid, book.mid_price - 0.005), 2)

        if price <= 0 or price >= 1:
            return None

        size = round(kelly_result.position_size / price, 2) if price > 0 else 0
        if size <= 0:
            return None

        if self.config.paper_mode:
            spread_cost = self.config.slippage_spread_bps / 10_000
            impact = self.config.slippage_impact_bps / 10_000 * size
            if kelly_result.side == "BUY":
                fill_price = min(price + spread_cost + impact, 0.99)
            else:
                fill_price = max(price - spread_cost - impact, 0.01)

            order_id = f"paper-{uuid.uuid4().hex[:12]}"
            result = OrderResult(
                order_id=order_id, status="PAPER_FILLED",
                token_id=token_id, side=kelly_result.side,
                price=fill_price, size=size, timestamp=time.time(),
            )
            logger.info(
                f"[PAPER] {kelly_result.side} {size} "
                f"nominal={price:.4f} fill={fill_price:.4f} "
                f"on {market.question[:40]}"
            )
        else:
            order = Order(
                token_id=token_id,
                side=Side.BUY if kelly_result.side == "BUY" else Side.SELL,
                price=price, size=size, order_type=OrderType.GTC,
            )
            fill_price = price
            result = await self.clob.submit_order(order)
            logger.info(
                f"[LIVE] Order {result.order_id}: {result.status} "
                f"{kelly_result.side} {size} @ {price}"
            )

        self.store.log_order(
            order_id=result.order_id, condition_id=market.condition_id,
            token_id=token_id, side=kelly_result.side,
            price=fill_price, size=size, status=result.status,
            paper_mode=self.config.paper_mode,
        )

        if "FILLED" in result.status or "PAPER" in result.status:
            self.store.upsert_position(
                condition_id=market.condition_id, token_id=token_id,
                token_choice=kelly_result.token_choice, size=size,
                avg_price=fill_price, paper_mode=self.config.paper_mode,
            )

        return result

    async def run_loop(self) -> None:
        self._running = True
        logger.info(
            f"Bot started in {'PAPER' if self.config.paper_mode else 'LIVE'} mode. "
            f"Scan interval: {self.config.scan_interval}s"
        )
        while self._running:
            await self.scan_once()
            logger.info(f"Sleeping {self.config.scan_interval}s until next scan...")
            await asyncio.sleep(self.config.scan_interval)

    def stop(self) -> None:
        self._running = False


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument("--once", action="store_true", help="Run single scan then exit")
    parser.add_argument("--paper", action="store_true", help="Force paper mode")
    parser.add_argument("--live", action="store_true", help="Force live mode")
    parser.add_argument("--log-level", default=None, help="Log level override")
    args = parser.parse_args()

    clob_config, chain_config, bot_config = load_config()

    overrides = {}
    if args.paper:
        overrides["paper_mode"] = True
    elif args.live:
        overrides["paper_mode"] = False
    if args.log_level:
        overrides["log_level"] = args.log_level
    if overrides:
        bot_config = dataclasses.replace(bot_config, **overrides)

    setup_logging(bot_config.log_level)
    validate_live_config(clob_config, chain_config, bot_config)

    bot = PolymarketBot(clob_config, chain_config, bot_config)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: _handle_shutdown(bot, loop))

    try:
        if args.once:
            summary = await bot.scan_once()
            logger.info(f"Single scan result: {summary}")
        else:
            await bot.run_loop()
    finally:
        await bot.close()
        bot.store.release_lock()


def _handle_shutdown(bot: PolymarketBot, loop: asyncio.AbstractEventLoop) -> None:
    logger.info("Shutdown signal received, stopping bot...")
    bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
