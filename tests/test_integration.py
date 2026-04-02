"""
Integration tests — full bot pipeline in paper mode.
These are more of smoke tests than unit tests: they validate that
the modules wire together without blowing up.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pytest_asyncio

from bot import PolymarketBot, setup_logging
from clob_client import ClobClient, Order, OrderBookSummary, Side, OrderType
from config import BotConfig, ChainConfig, ClobConfig
from kelly import KellyResult, kelly_criterion
from market_scanner import Market, MarketCategory, MarketScanner
from persistence import PersistenceStore
from ssvi import SSVIParams, extract_probability, fit_ssvi, generate_synthetic_surface


@pytest.mark.slow
class TestSSVICalibrationIntegration:

    def test_ssvi_calibration_r_squared(self):
        strikes, ivs = generate_synthetic_surface(
            spot=100.0, base_vol=0.3, time_to_expiry=30 / 365,
            n_strikes=15, noise_std=0.005,
        )
        params = fit_ssvi(strikes, ivs, forward=100.0, time_to_expiry=30 / 365)
        assert params.r_squared > 0.80, f"R² = {params.r_squared:.4f}, expected > 0.80"

    def test_probability_above_spot_approx_half(self):
        strikes, ivs = generate_synthetic_surface(
            spot=100.0, base_vol=0.3, time_to_expiry=30 / 365,
            n_strikes=15, noise_std=0.005,
        )
        params = fit_ssvi(strikes, ivs, forward=100.0, time_to_expiry=30 / 365)
        prob = extract_probability(params, spot=100.0, forward=100.0, time_to_expiry=30 / 365)
        assert abs(prob.probability_above - 0.50) < 0.15

    def test_ssvi_across_time_horizons(self):
        for days in [7, 30, 90]:
            t = days / 365
            strikes, ivs = generate_synthetic_surface(
                spot=50.0, base_vol=0.5, time_to_expiry=t,
                n_strikes=15, noise_std=0.005,
            )
            params = fit_ssvi(strikes, ivs, forward=50.0, time_to_expiry=t)
            assert params.r_squared > 0.50, f"R² at {days}d = {params.r_squared:.4f}"


class TestFullPipelineIntegration:

    @pytest.fixture
    def paper_bot(self, tmp_path):
        clob = ClobConfig(api_key="test", api_secret="test", api_passphrase="test")
        chain = ChainConfig(
            private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        bot_cfg = BotConfig(
            paper_mode=True,
            db_path=str(tmp_path / "integration.db"),
            max_markets=3, kelly_fraction=0.25, min_edge=0.02,
        )
        setup_logging("DEBUG")
        return PolymarketBot(clob, chain, bot_cfg)

    @pytest.mark.asyncio
    async def test_scan_once_paper_mode_no_exceptions(self, paper_bot):
        try:
            summary = await paper_bot.scan_once()
        finally:
            await paper_bot.close()

        assert isinstance(summary, dict)
        assert "markets_found" in summary
        assert "orders_placed" in summary
        assert isinstance(summary.get("errors", []), list)

    @pytest.mark.asyncio
    async def test_paper_mode_logging(self, paper_bot):
        try:
            await paper_bot.scan_once()
        finally:
            await paper_bot.close()
        assert len(paper_bot.store.get_scan_history(limit=5)) >= 1

    @pytest.mark.asyncio
    async def test_bot_close_is_safe(self, paper_bot):
        await paper_bot.close()
        await paper_bot.close()  # double close should be fine


class TestCircuitBreaker:
    """Verify the circuit breaker trips and resets correctly."""

    @pytest.fixture
    def paper_bot(self, tmp_path):
        clob = ClobConfig(api_key="test", api_secret="test", api_passphrase="test")
        chain = ChainConfig(
            private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        bot_cfg = BotConfig(
            paper_mode=True,
            db_path=str(tmp_path / "cb_test.db"),
            max_consecutive_failures=3,
            circuit_breaker_cooldown=2,  # short for testing
        )
        return PolymarketBot(clob, chain, bot_cfg)

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_after_consecutive_failures(self, paper_bot):
        # Force scan_and_classify to raise so scan_once increments failure counter
        with patch.object(paper_bot.scanner, "scan_and_classify", side_effect=RuntimeError("boom")):
            for _ in range(3):
                summary = await paper_bot.scan_once()
                assert "boom" in summary["errors"][0]

            # After 3 failures, circuit breaker should be active
            assert paper_bot._circuit_breaker_active()

            # Next scan should be skipped
            summary = await paper_bot.scan_once()
            assert "circuit_breaker_active" in summary["errors"]

        await paper_bot.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_after_cooldown(self, paper_bot):
        # Trip the breaker
        with patch.object(paper_bot.scanner, "scan_and_classify", side_effect=RuntimeError("fail")):
            for _ in range(3):
                await paper_bot.scan_once()

        assert paper_bot._circuit_breaker_active()

        # Wait for cooldown to expire
        paper_bot._circuit_breaker_until = time.time() - 1

        # Should no longer be active
        assert not paper_bot._circuit_breaker_active()
        await paper_bot.close()

    @pytest.mark.asyncio
    async def test_successful_scan_resets_failure_count(self, paper_bot):
        # Accumulate some failures (but don't trip)
        with patch.object(paper_bot.scanner, "scan_and_classify", side_effect=RuntimeError("err")):
            await paper_bot.scan_once()
            await paper_bot.scan_once()

        assert paper_bot._consecutive_failures == 2

        # Successful scan resets
        with patch.object(paper_bot.scanner, "scan_and_classify", return_value=[]):
            await paper_bot.scan_once()

        assert paper_bot._consecutive_failures == 0
        await paper_bot.close()


class TestKellyIntegration:

    def test_kelly_with_ssvi_probability(self):
        strikes, ivs = generate_synthetic_surface(
            spot=0.55, base_vol=0.5, time_to_expiry=1 / 365,
        )
        params = fit_ssvi(strikes, ivs, forward=0.55, time_to_expiry=1 / 365)
        prob_est = extract_probability(
            params, spot=0.55, forward=0.55, time_to_expiry=1 / 365,
        )
        result = kelly_criterion(
            estimated_prob=prob_est.probability_above,
            market_price=0.50, bankroll=1000.0,
            fraction=0.25, max_position=100.0, min_edge=0.01,
        )
        # May or may not find edge, but must not crash
        if result is not None:
            assert 0 < result.position_size <= 100.0


class TestPersistenceIntegration:

    def test_full_order_lifecycle(self, tmp_path):
        db_path = str(tmp_path / "lifecycle.db")
        store = PersistenceStore(db_path)

        store.log_order("o1", "c1", "t1", "BUY", 0.55, 10.0, status="PLACED", paper_mode=True)
        store.update_order_status("o1", "FILLED")
        store.upsert_position("c1", "t1", "YES", 10.0, 0.55, paper_mode=True)
        store.log_scan(50, 3, 1, paper_mode=True)

        orders = store.get_recent_orders(paper_mode=True)
        assert orders[0]["status"] == "FILLED"

        positions = store.get_open_positions(paper_mode=True)
        assert positions[0]["size"] == 10.0

        assert store.get_total_exposure(paper_mode=True) == pytest.approx(5.50)
        assert len(store.get_scan_history()) == 1


class TestMarketClassificationIntegration:

    def test_classify_various_markets(self):
        from market_scanner import classify_market
        questions = [
            ("Will Bitcoin hit $100k by end of 2025?", MarketCategory.CRYPTO),
            ("Will Trump win the 2024 election?", MarketCategory.POLITICS),
            ("Super Bowl LVIII winner?", MarketCategory.SPORTS),
            ("Will the Fed cut rates in March?", MarketCategory.FINANCE),
            ("Best Picture Oscar 2025?", MarketCategory.ENTERTAINMENT),
        ]
        classified_count = sum(
            1 for question, expected in questions
            if classify_market(question) == expected
        )
        assert classified_count >= 4


class TestSSVIR2Rejection:

    @pytest.fixture
    def paper_bot(self, tmp_path):
        clob = ClobConfig(api_key="test", api_secret="test", api_passphrase="test")
        chain = ChainConfig(
            private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        bot_cfg = BotConfig(
            paper_mode=True,
            db_path=str(tmp_path / "r2test.db"),
            ssvi_r2_threshold=0.70,
        )
        setup_logging("DEBUG")
        return PolymarketBot(clob, chain, bot_cfg)

    @pytest.mark.asyncio
    async def test_low_r2_falls_back_to_mid_price(self, paper_bot):
        market = Market(
            condition_id="c1", question="Test crypto", description="",
            category=MarketCategory.OTHER,
            tokens=[{"token_id": "t1", "outcome": "Yes"}],
            end_date="", active=True, volume=1000, liquidity=500, neg_risk=False,
        )
        low_r2_params = SSVIParams(theta=0.1, rho=0.0, phi=1.0, r_squared=0.30)
        with patch("bot.fit_ssvi", return_value=low_r2_params):
            prob = await paper_bot._estimate_probability(market, 0.55)
            await paper_bot.close()
        assert prob == 0.55

    @pytest.mark.asyncio
    async def test_high_r2_uses_ssvi_probability(self, paper_bot):
        market = Market(
            condition_id="c1", question="Test", description="",
            category=MarketCategory.OTHER,
            tokens=[{"token_id": "t1", "outcome": "Yes"}],
            end_date="", active=True, volume=1000, liquidity=500, neg_risk=False,
        )
        prob = await paper_bot._estimate_probability(market, 0.55)
        await paper_bot.close()
        assert isinstance(prob, float)
        assert 0 < prob < 1


class TestPaperSlippage:

    @pytest.fixture
    def paper_bot(self, tmp_path):
        clob = ClobConfig(api_key="test", api_secret="test", api_passphrase="test")
        chain = ChainConfig(
            private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        bot_cfg = BotConfig(
            paper_mode=True,
            db_path=str(tmp_path / "slippage.db"),
            slippage_spread_bps=50,
            slippage_impact_bps=10,
        )
        return PolymarketBot(clob, chain, bot_cfg)

    def _make_market(self):
        return Market(
            condition_id="c1", question="Test", description="",
            category=MarketCategory.OTHER,
            tokens=[
                {"token_id": "t1", "outcome": "Yes"},
                {"token_id": "t2", "outcome": "No"},
            ],
            end_date="", active=True, volume=1000, liquidity=500, neg_risk=False,
        )

    @pytest.mark.asyncio
    async def test_paper_buy_fills_above_mid(self, paper_bot):
        market = self._make_market()
        kelly = KellyResult(
            edge=0.05, kelly_fraction=0.10, position_size=50.0,
            confidence=0.60, side="BUY", token_choice="YES",
        )
        book = OrderBookSummary(
            token_id="t1", best_bid=0.50, best_ask=0.55,
            mid_price=0.525, spread=0.05, bid_depth=100, ask_depth=100,
        )
        with patch.object(paper_bot.clob, "get_order_book", return_value=book):
            result = await paper_bot._execute_trade(market, kelly)
            await paper_bot.close()
        assert result is not None
        nominal_price = round(min(book.best_ask, book.mid_price + 0.005), 2)
        assert result.price > nominal_price

    @pytest.mark.asyncio
    async def test_paper_sell_fills_below_mid(self, paper_bot):
        market = self._make_market()
        kelly = KellyResult(
            edge=0.05, kelly_fraction=0.10, position_size=50.0,
            confidence=0.60, side="SELL", token_choice="YES",
        )
        book = OrderBookSummary(
            token_id="t1", best_bid=0.50, best_ask=0.55,
            mid_price=0.525, spread=0.05, bid_depth=100, ask_depth=100,
        )
        with patch.object(paper_bot.clob, "get_order_book", return_value=book):
            result = await paper_bot._execute_trade(market, kelly)
            await paper_bot.close()
        assert result is not None
        nominal_price = round(max(book.best_bid, book.mid_price - 0.005), 2)
        assert result.price < nominal_price

    @pytest.mark.asyncio
    async def test_slippage_capped_at_bounds(self, paper_bot):
        market = self._make_market()
        kelly = KellyResult(
            edge=0.05, kelly_fraction=0.10, position_size=50.0,
            confidence=0.60, side="BUY", token_choice="YES",
        )
        book = OrderBookSummary(
            token_id="t1", best_bid=0.97, best_ask=0.98,
            mid_price=0.975, spread=0.01, bid_depth=100, ask_depth=100,
        )
        with patch.object(paper_bot.clob, "get_order_book", return_value=book):
            result = await paper_bot._execute_trade(market, kelly)
            await paper_bot.close()
        assert result is not None
        assert result.price <= 0.99
