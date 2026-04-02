"""
Main bot — async scan loop, OB probability estimation, Kelly sizing, paper/live execution.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import signal
import sys
import time
import uuid
from typing import Optional, Tuple

from clob_client import ClobClient, Order, OrderBookSummary, OrderResult, Side, OrderType
from config import BotConfig, ChainConfig, ClobConfig, load_config, validate_live_config
from kelly import KellyResult, check_position_limits, kelly_criterion
from market_scanner import Market, MarketCategory, MarketScanner
from persistence import PersistenceStore

logger = logging.getLogger("polymarket-bot")


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Propagate any extra structured fields attached to the record
        for key in ("event", "order_id", "condition_id", "side", "price",
                     "size", "edge", "slippage", "market"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
    )


def _estimate_live_fill_price(
    nominal_price: float,
    size: float,
    book: OrderBookSummary,
    side: str,
    max_slippage_bps: int,
) -> Optional[float]:
    """Estimate expected fill price from order book depth.

    Uses the square-root market impact model (Bouchaud et al.):
    impact = (spread / 2) * sqrt(size / depth). This correctly captures
    the concave relationship between order size and slippage — doubling
    order size does *not* double slippage. Returns None if estimated
    slippage exceeds the configured maximum.
    """
    if side == "BUY":
        depth = book.ask_depth
        if depth <= 0:
            return None
        impact = (book.spread / 2) * math.sqrt(size / depth)
        estimated = nominal_price + impact
    else:
        depth = book.bid_depth
        if depth <= 0:
            return None
        impact = (book.spread / 2) * math.sqrt(size / depth)
        estimated = nominal_price - impact

    # Check if estimated slippage exceeds our tolerance
    slippage_bps = abs(estimated - nominal_price) / max(nominal_price, 1e-9) * 10_000
    if slippage_bps > max_slippage_bps:
        return None
    return estimated


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
                limit=self.config.max_markets * 20,
                min_liquidity=self.config.min_liquidity,
            )
            summary["markets_found"] = len(markets)
            logger.info("Found %d active markets", len(markets))

            current_exposure = self.store.get_total_exposure(self.config.paper_mode)
            bankroll = self.config.max_position_usdc * self.config.bankroll_multiplier

            for market in markets[: self.config.max_markets]:
                try:
                    eval_result = await self._evaluate_market(
                        market, bankroll, current_exposure
                    )
                    if eval_result is not None:
                        result, book = eval_result
                        summary["markets_with_edge"] += 1
                        order_result = await self._execute_trade(market, result, book)
                        if order_result is not None:
                            summary["orders_placed"] += 1
                            current_exposure += result.position_size
                except Exception as e:
                    logger.warning(
                        "Error evaluating %s: %s", market.question[:50], e,
                        exc_info=True,
                    )
                    summary["errors"].append(
                        "Error evaluating %s: %s" % (market.question[:50], e)
                    )

            # Scan completed — reset circuit breaker
            self._reset_circuit_breaker()

        except Exception as e:
            logger.error("Scan cycle error: %s", e, exc_info=True)
            summary["errors"].append("Scan cycle error: %s" % e)
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                self._trip_circuit_breaker()

        self.store.log_scan(
            markets_found=summary["markets_found"],
            markets_with_edge=summary["markets_with_edge"],
            orders_placed=summary["orders_placed"],
            paper_mode=self.config.paper_mode,
        )

        if summary["markets_found"] > 0 and summary["markets_with_edge"] == 0:
            logger.warning(
                "Evaluated %d markets but found 0 with edge — all filtered "
                "by spread/price bounds. Check market conditions or loosen "
                "max_spread (currently %s).",
                summary["markets_found"], self.config.max_spread,
            )
        elif summary["markets_found"] == 0:
            logger.warning(
                "No active markets returned from Gamma API — "
                "check min_liquidity filter or API availability."
            )

        logger.info(
            "Scan complete: %d markets, %d with edge, %d orders placed",
            summary["markets_found"], summary["markets_with_edge"],
            summary["orders_placed"],
        )
        return summary

    async def _evaluate_market(
        self, market: Market, bankroll: float, current_exposure: float
    ) -> Optional[Tuple[KellyResult, OrderBookSummary]]:
        if not market.yes_token_id:
            logger.debug("Skip %s: no yes_token_id", market.question[:40])
            return None

        book = await self.clob.get_order_book(market.yes_token_id)
        market_price = book.mid_price

        if market_price <= self.config.min_tradeable_price or market_price >= self.config.max_tradeable_price:
            logger.debug("Skip %s: price=%.4f out of bounds", market.question[:40], market_price)
            return None
        if book.spread > self.config.max_spread:
            logger.debug("Skip %s: spread=%.4f > %s", market.question[:40], book.spread, self.config.max_spread)
            return None

        estimated_prob = await self._estimate_probability(market, book)

        result = kelly_criterion(
            estimated_prob=estimated_prob,
            market_price=market_price,
            bankroll=bankroll,
            fraction=self.config.kelly_fraction,
            max_position=self.config.max_position_usdc,
            min_edge=self.config.min_edge,
        )

        # Log OB snapshot for forward-test validation
        total_depth = book.bid_depth + book.ask_depth
        imbalance = (book.bid_depth - book.ask_depth) / total_depth if total_depth > 0 else 0.0
        try:
            self.store.log_ob_snapshot(
                condition_id=market.condition_id,
                token_id=market.yes_token_id or "",
                best_bid=book.best_bid,
                best_ask=book.best_ask,
                mid_price=book.mid_price,
                spread=book.spread,
                bid_depth=book.bid_depth,
                ask_depth=book.ask_depth,
                imbalance=imbalance,
                estimated_prob=estimated_prob,
                market_price=market_price,
                had_edge=result is not None,
                paper_mode=self.config.paper_mode,
            )
        except Exception as e:
            logger.debug("Failed to log OB snapshot: %s", e)

        if result is None:
            logger.debug(
                "Skip %s: no edge (est=%.4f mkt=%.4f)",
                market.question[:40], estimated_prob, market_price,
            )
            return None

        adjusted_size = check_position_limits(
            current_exposure, result.position_size,
            self.config.max_position_usdc * 5,
        )
        if adjusted_size <= 0:
            return None

        result = dataclasses.replace(result, position_size=adjusted_size)
        logger.info(
            "Edge found: %s | edge=%.4f kelly=%.4f size=$%.2f side=%s",
            market.question[:60], result.edge, result.kelly_fraction,
            result.position_size, result.token_choice,
        )
        return result, book

    async def _estimate_probability(
        self, market: Market, book: OrderBookSummary
    ) -> float:
        return self._ob_imbalance_probability(book)

    def _ob_imbalance_probability(self, book: OrderBookSummary) -> float:
        """Estimate probability from order book depth imbalance.

        When bid_depth >> ask_depth, informed traders expect the price to
        rise, so the true probability is likely above mid-price.
        """
        total_depth = book.bid_depth + book.ask_depth
        if total_depth <= 0:
            logger.debug("OB imbalance: zero total depth, using mid-price")
            return book.mid_price

        imbalance = (book.bid_depth - book.ask_depth) / total_depth
        adjusted = book.mid_price + imbalance * self.config.ob_imbalance_weight
        clamped = max(0.01, min(0.99, adjusted))

        logger.debug(
            "OB imbalance: bid_depth=%.1f ask_depth=%.1f imbalance=%.3f "
            "mid=%.4f adjusted=%.4f",
            book.bid_depth, book.ask_depth, imbalance,
            book.mid_price, clamped,
        )
        return clamped

    async def _execute_trade(
        self, market: Market, kelly_result: KellyResult,
        book: OrderBookSummary,
    ) -> Optional[OrderResult]:
        token_id = (
            market.yes_token_id
            if kelly_result.token_choice == "YES"
            else market.no_token_id
        )
        if not token_id:
            return None

        if kelly_result.side == "BUY":
            price = round(min(book.best_ask, book.mid_price + 0.005), 2)
        else:
            price = round(max(book.best_bid, book.mid_price - 0.005), 2)

        if price <= 0 or price >= 1:
            return None

        size = round(kelly_result.position_size / price, 2)
        if size <= 0:
            return None

        if self.config.paper_mode:
            spread_cost = self.config.slippage_spread_bps / 10_000
            depth = book.ask_depth if kelly_result.side == "BUY" else book.bid_depth
            if depth > 0:
                impact = spread_cost * math.sqrt(size / depth)
            else:
                impact = spread_cost
            if kelly_result.side == "BUY":
                fill_price = min(price + impact, self.config.max_tradeable_price)
            else:
                fill_price = max(price - impact, self.config.min_tradeable_price)

            order_id = f"paper-{uuid.uuid4().hex[:12]}"
            result = OrderResult(
                order_id=order_id, status="PAPER_FILLED",
                token_id=token_id, side=kelly_result.side,
                price=fill_price, size=size, timestamp=time.time(),
            )
            logger.info(
                "[PAPER] %s %s nominal=%.4f fill=%.4f on %s",
                kelly_result.side, size, price, fill_price,
                market.question[:40],
            )
        else:
            # Estimate live slippage from order book depth
            estimated_fill = _estimate_live_fill_price(
                price, size, book, kelly_result.side,
                self.config.slippage_spread_bps,
            )
            if estimated_fill is None:
                logger.warning(
                    "Skipping live order: estimated slippage too high "
                    "(depth too thin for size=%s)", size,
                )
                return None
            order = Order(
                token_id=token_id,
                side=Side.BUY if kelly_result.side == "BUY" else Side.SELL,
                price=price, size=size, order_type=OrderType.GTC,
            )
            fill_price = price
            result = await self.clob.submit_order(order)
            logger.info(
                "[LIVE] Order %s: %s %s %s @ %s (est. fill=%.4f)",
                result.order_id, result.status, kelly_result.side,
                size, price, estimated_fill,
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
            "Bot started in %s mode. Scan interval: %ds",
            "PAPER" if self.config.paper_mode else "LIVE",
            self.config.scan_interval,
        )
        while self._running:
            await self.scan_once()
            logger.info("Sleeping %ds until next scan...", self.config.scan_interval)
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

    setup_logging(bot_config.log_level, bot_config.log_format)
    validate_live_config(clob_config, chain_config, bot_config)

    bot = PolymarketBot(clob_config, chain_config, bot_config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: _handle_shutdown(bot, loop))

    try:
        if args.once:
            summary = await bot.scan_once()
            logger.info("Single scan result: %s", summary)
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
