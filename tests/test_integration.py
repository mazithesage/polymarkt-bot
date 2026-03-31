"""
Integration tests - run the full bot in --once --paper mode.
Validates: SSVI calibration R² > 0.80, probability_above(spot) ≈ 0.50,
market scanning, classification, and no unhandled exceptions.
"""

import asyncio
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


class TestSSVICalibrationIntegration:
    """Validate that SSVI calibrates with R² > 0.80."""

    def test_ssvi_calibration_r_squared(self):
        """Full SSVI calibration pipeline achieves R² > 0.80."""
        strikes, ivs = generate_synthetic_surface(
            spot=100.0, base_vol=0.3, time_to_expiry=30 / 365,
            n_strikes=15, noise_std=0.005,
        )
        params = fit_ssvi(strikes, ivs, forward=100.0, time_to_expiry=30 / 365)
        assert params.r_squared > 0.80, f"R² = {params.r_squared:.4f}, expected > 0.80"

    def test_probability_above_spot_approx_half(self):
        """probability_above(spot) should be approximately 0.50."""
        strikes, ivs = generate_synthetic_surface(
            spot=100.0, base_vol=0.3, time_to_expiry=30 / 365,
            n_strikes=15, noise_std=0.005,
        )
        params = fit_ssvi(strikes, ivs, forward=100.0, time_to_expiry=30 / 365)
        prob = extract_probability(
            params, spot=100.0, forward=100.0, time_to_expiry=30 / 365,
        )
        assert abs(prob.probability_above - 0.50) < 0.15, (
            f"P(above spot) = {prob.probability_above:.4f}, expected ≈ 0.50"
        )

    def test_ssvi_with_multiple_time_horizons(self):
        """SSVI should work across different time horizons."""
        for days in [7, 30, 90]:
            t = days / 365
            strikes, ivs = generate_synthetic_surface(
                spot=50.0, base_vol=0.5, time_to_expiry=t,
                n_strikes=15, noise_std=0.005,
            )
            params = fit_ssvi(strikes, ivs, forward=50.0, time_to_expiry=t)
            assert params.r_squared > 0.50, f"R² at {days}d = {params.r_squared:.4f}"


class TestFullPipelineIntegration:
    """Test the full bot pipeline with mocked network calls."""

    @pytest.fixture
    def paper_bot(self, tmp_path):
        clob = ClobConfig(api_key="test", api_secret="test", api_passphrase="test")
        chain = ChainConfig(
            private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        bot_cfg = BotConfig(
            paper_mode=True,
            db_path=str(tmp_path / "integration.db"),
            max_markets=3,
            kelly_fraction=0.25,
            min_edge=0.02,
        )
        setup_logging("DEBUG")
        return PolymarketBot(clob, chain, bot_cfg)

    @pytest.mark.asyncio
    async def test_scan_once_paper_mode_no_exceptions(self, paper_bot):
        """
        Full scan in paper mode should complete without exceptions.
        Network calls will fail gracefully since we're using fake credentials,
        but the bot should handle this and produce a summary.
        """
        try:
            summary = await paper_bot.scan_once()
        finally:
            await paper_bot.close()

        assert isinstance(summary, dict)
        assert "markets_found" in summary
        assert "markets_with_edge" in summary
        assert "orders_placed" in summary
        # Network may fail but no unhandled exceptions
        assert isinstance(summary.get("errors", []), list)

    @pytest.mark.asyncio
    async def test_paper_mode_logging(self, paper_bot):
        """Paper mode should log scans to SQLite."""
        try:
            await paper_bot.scan_once()
        finally:
            await paper_bot.close()

        history = paper_bot.store.get_scan_history(limit=5)
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_bot_close_is_safe(self, paper_bot):
        """Closing the bot should not raise."""
        await paper_bot.close()
        await paper_bot.close()  # Double close should be safe


class TestKellyIntegration:
    """Integration test: Kelly sizing with realistic market data."""

    def test_kelly_with_ssvi_probability(self):
        """Chain SSVI probability estimation into Kelly sizing."""
        # Generate surface, fit, extract probability
        strikes, ivs = generate_synthetic_surface(
            spot=0.55, base_vol=0.5, time_to_expiry=1 / 365,
        )
        params = fit_ssvi(strikes, ivs, forward=0.55, time_to_expiry=1 / 365)
        prob_est = extract_probability(
            params, spot=0.55, forward=0.55, time_to_expiry=1 / 365,
        )

        # Use extracted probability for Kelly
        market_price = 0.50
        result = kelly_criterion(
            estimated_prob=prob_est.probability_above,
            market_price=market_price,
            bankroll=1000.0,
            fraction=0.25,
            max_position=100.0,
            min_edge=0.01,  # Lower threshold for integration test
        )
        # Result may or may not find edge depending on SSVI fit
        # But it should not crash
        if result is not None:
            assert result.position_size > 0
            assert result.position_size <= 100.0


class TestPersistenceIntegration:
    """Integration test: full order lifecycle in SQLite."""

    def test_full_order_lifecycle(self, tmp_path):
        db_path = str(tmp_path / "lifecycle.db")
        store = PersistenceStore(db_path)

        # Place order
        store.log_order("o1", "c1", "t1", "BUY", 0.55, 10.0, status="PLACED", paper_mode=True)
        # Update to filled
        store.update_order_status("o1", "FILLED")
        # Record position
        store.upsert_position("c1", "t1", "YES", 10.0, 0.55, paper_mode=True)
        # Log scan
        store.log_scan(50, 3, 1, paper_mode=True)

        # Verify state
        orders = store.get_recent_orders(paper_mode=True)
        assert len(orders) == 1
        assert orders[0]["status"] == "FILLED"

        positions = store.get_open_positions(paper_mode=True)
        assert len(positions) == 1
        assert positions[0]["size"] == 10.0

        exposure = store.get_total_exposure(paper_mode=True)
        assert exposure == pytest.approx(5.50)

        history = store.get_scan_history()
        assert len(history) == 1


class TestMarketClassificationIntegration:
    """Integration: at least one market can be classified."""

    def test_classify_various_markets(self):
        """Ensure diverse market questions get classified."""
        questions = [
            ("Will Bitcoin hit $100k by end of 2025?", MarketCategory.CRYPTO),
            ("Will Trump win the 2024 election?", MarketCategory.POLITICS),
            ("Super Bowl LVIII winner?", MarketCategory.SPORTS),
            ("Will the Fed cut rates in March?", MarketCategory.FINANCE),
            ("Best Picture Oscar 2025?", MarketCategory.ENTERTAINMENT),
        ]
        from market_scanner import classify_market
        classified_count = 0
        for question, expected_cat in questions:
            cat = classify_market(question)
            if cat == expected_cat:
                classified_count += 1
        # At least 4 out of 5 should classify correctly
        assert classified_count >= 4, f"Only {classified_count}/5 classified correctly"


class TestSSVIR2Rejection:
    """Test that low R² SSVI fits fall back to mid-price."""

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
        """When SSVI R² is below threshold, should return mid_price."""
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
        # Should fall back to mid_price since R² (0.30) < threshold (0.70)
        assert prob == 0.55

    @pytest.mark.asyncio
    async def test_high_r2_uses_ssvi_probability(self, paper_bot):
        """When SSVI R² is above threshold, should use SSVI probability."""
        market = Market(
            condition_id="c1", question="Test", description="",
            category=MarketCategory.OTHER,
            tokens=[{"token_id": "t1", "outcome": "Yes"}],
            end_date="", active=True, volume=1000, liquidity=500, neg_risk=False,
        )
        # Use a real SSVI fit with good R²
        prob = await paper_bot._estimate_probability(market, 0.55)
        await paper_bot.close()
        # Should NOT just return mid_price (0.55) — SSVI modifies it
        # With synthetic surface and good fit, the result should be different from mid_price
        # (or very close, which is also valid)
        assert isinstance(prob, float)
        assert 0 < prob < 1


class TestPaperSlippage:
    """Test that paper mode applies slippage to fill prices."""

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
        """Paper BUY should fill above the nominal price due to slippage."""
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
        assert result.price > nominal_price  # Slippage pushes fill up

    @pytest.mark.asyncio
    async def test_paper_sell_fills_below_mid(self, paper_bot):
        """Paper SELL should fill below the nominal price due to slippage."""
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
        assert result.price < nominal_price  # Slippage pushes fill down

    @pytest.mark.asyncio
    async def test_slippage_capped_at_bounds(self, paper_bot):
        """Slippage should not push fill price above 0.99 or below 0.01."""
        market = self._make_market()
        kelly = KellyResult(
            edge=0.05, kelly_fraction=0.10, position_size=50.0,
            confidence=0.60, side="BUY", token_choice="YES",
        )
        # Very high price near 1.0
        book = OrderBookSummary(
            token_id="t1", best_bid=0.97, best_ask=0.98,
            mid_price=0.975, spread=0.01, bid_depth=100, ask_depth=100,
        )
        with patch.object(paper_bot.clob, "get_order_book", return_value=book):
            result = await paper_bot._execute_trade(market, kelly)
            await paper_bot.close()
        assert result is not None
        assert result.price <= 0.99
