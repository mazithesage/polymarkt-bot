import pytest
from config import BotConfig, ChainConfig, ClobConfig, validate_live_config


class TestBotConfigValidation:
    def test_defaults_are_valid(self):
        cfg = BotConfig()
        assert cfg.kelly_fraction == 0.25

    def test_kelly_fraction_zero_rejected(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            BotConfig(kelly_fraction=0.0)

    def test_kelly_fraction_negative_rejected(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            BotConfig(kelly_fraction=-0.1)

    def test_kelly_fraction_above_one_rejected(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            BotConfig(kelly_fraction=1.5)

    def test_kelly_fraction_one_allowed(self):
        cfg = BotConfig(kelly_fraction=1.0)
        assert cfg.kelly_fraction == 1.0

    def test_scan_interval_zero_rejected(self):
        with pytest.raises(ValueError, match="scan_interval"):
            BotConfig(scan_interval=0)

    def test_scan_interval_negative_rejected(self):
        with pytest.raises(ValueError, match="scan_interval"):
            BotConfig(scan_interval=-10)

    def test_max_position_usdc_zero_rejected(self):
        with pytest.raises(ValueError, match="max_position_usdc"):
            BotConfig(max_position_usdc=0.0)

    def test_max_position_usdc_negative_rejected(self):
        with pytest.raises(ValueError, match="max_position_usdc"):
            BotConfig(max_position_usdc=-50.0)

    def test_min_edge_zero_rejected(self):
        with pytest.raises(ValueError, match="min_edge"):
            BotConfig(min_edge=0.0)

    def test_min_edge_one_rejected(self):
        with pytest.raises(ValueError, match="min_edge"):
            BotConfig(min_edge=1.0)

    def test_min_edge_negative_rejected(self):
        with pytest.raises(ValueError, match="min_edge"):
            BotConfig(min_edge=-0.01)

    def test_max_markets_zero_rejected(self):
        with pytest.raises(ValueError, match="max_markets"):
            BotConfig(max_markets=0)

    def test_max_markets_negative_rejected(self):
        with pytest.raises(ValueError, match="max_markets"):
            BotConfig(max_markets=-1)

    def test_new_fields_have_defaults(self):
        cfg = BotConfig()
        assert cfg.ssvi_r2_threshold == 0.70
        assert cfg.slippage_spread_bps == 50
        assert cfg.slippage_impact_bps == 10
        assert cfg.http_timeout == 30.0
        assert cfg.http_max_retries == 3

    def test_circuit_breaker_defaults(self):
        cfg = BotConfig()
        assert cfg.max_consecutive_failures == 5
        assert cfg.circuit_breaker_cooldown == 300

    def test_market_filter_defaults(self):
        cfg = BotConfig()
        assert cfg.min_tradeable_price == 0.01
        assert cfg.max_tradeable_price == 0.99
        assert cfg.max_spread == 0.10
        assert cfg.bankroll_multiplier == 10.0

    def test_log_format_defaults_to_text(self):
        cfg = BotConfig()
        assert cfg.log_format == "text"

    def test_log_format_json_accepted(self):
        cfg = BotConfig(log_format="json")
        assert cfg.log_format == "json"

    def test_log_format_invalid_rejected(self):
        with pytest.raises(ValueError, match="log_format"):
            BotConfig(log_format="xml")

    def test_inverted_price_bounds_rejected(self):
        with pytest.raises(ValueError, match="min_tradeable_price"):
            BotConfig(min_tradeable_price=0.99, max_tradeable_price=0.01)

    def test_max_spread_zero_rejected(self):
        with pytest.raises(ValueError, match="max_spread"):
            BotConfig(max_spread=0.0)

    def test_bankroll_multiplier_zero_rejected(self):
        with pytest.raises(ValueError, match="bankroll_multiplier"):
            BotConfig(bankroll_multiplier=0.0)


class TestValidateLiveConfig:
    def test_paper_mode_always_passes(self):
        validate_live_config(ClobConfig(), ChainConfig(), BotConfig(paper_mode=True))

    def test_live_mode_missing_all_raises(self):
        with pytest.raises(ValueError, match="PRIVATE_KEY"):
            validate_live_config(ClobConfig(), ChainConfig(), BotConfig(paper_mode=False))

    def test_live_mode_with_credentials_passes(self):
        clob = ClobConfig(api_key="k", api_secret="s", api_passphrase="p")
        chain = ChainConfig(private_key="0xdeadbeef")
        validate_live_config(clob, chain, BotConfig(paper_mode=False))

    def test_live_mode_missing_api_key_raises(self):
        clob = ClobConfig(api_secret="s", api_passphrase="p")
        chain = ChainConfig(private_key="0xdeadbeef")
        with pytest.raises(ValueError, match="API_KEY"):
            validate_live_config(clob, chain, BotConfig(paper_mode=False))
