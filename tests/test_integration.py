"""
Integration tests - run the full bot in --once --paper mode.
Validates: SSVI calibration R² > 0.80, probability_above(spot) ≈ 0.50,
market scanning, classification, and no unhandled exceptions.
"""

import asyncio

import numpy as np
import pytest
import pytest_asyncio

from bot import PolymarketBot, setup_logging
from clob_client import ClobClient, Order, OrderBookSummary, Side, OrderType
from config import BotConfig, ChainConfig, ClobConfig
from kelly import kelly_criterion
from market_scanner import Market, MarketCategory, MarketScanner
from persistence import PersistenceStore
from ssvi import extract_probability, fit_ssvi, generate_synthetic_surface


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
