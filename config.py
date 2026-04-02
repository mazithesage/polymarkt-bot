"""Bot configuration. Loads from env vars with sane defaults."""

import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv

POLYGON_CHAIN_ID = 137
USDC_DECIMALS = 6

# Polymarket contract addresses (Polygon mainnet)
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"


@dataclass(frozen=True)
class ClobConfig:
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"


@dataclass(frozen=True)
class ChainConfig:
    rpc_url: str = "https://polygon-rpc.com"
    chain_id: int = POLYGON_CHAIN_ID
    private_key: str = ""
    proxy_address: str = ""
    ctf_exchange: str = CTF_EXCHANGE
    neg_risk_exchange: str = NEG_RISK_EXCHANGE
    conditional_tokens: str = CONDITIONAL_TOKENS
    usdc_address: str = USDC_ADDRESS


@dataclass(frozen=True)
class BotConfig:
    paper_mode: bool = True
    scan_interval: int = 60
    max_position_usdc: float = 100.0
    kelly_fraction: float = 0.25
    min_edge: float = 0.02
    max_markets: int = 10
    order_type: str = "GTC"
    db_path: str = "data/bot.db"
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"
    ssvi_r2_threshold: float = 0.70
    slippage_spread_bps: int = 50
    slippage_impact_bps: int = 10
    http_timeout: float = 30.0
    http_max_retries: int = 3
    # Circuit breaker — pause after repeated scan failures
    max_consecutive_failures: int = 5
    circuit_breaker_cooldown: int = 300  # seconds
    # Market filtering thresholds — previously hardcoded, now tunable
    min_tradeable_price: float = 0.01
    max_tradeable_price: float = 0.99
    max_spread: float = 0.10
    min_liquidity: float = 5000.0
    bankroll_multiplier: float = 10.0
    ob_imbalance_weight: float = 0.05

    def __post_init__(self):
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")
        if self.scan_interval <= 0:
            raise ValueError(f"scan_interval must be > 0, got {self.scan_interval}")
        if self.max_position_usdc <= 0:
            raise ValueError(f"max_position_usdc must be > 0, got {self.max_position_usdc}")
        if not (0 < self.min_edge < 1):
            raise ValueError(f"min_edge must be in (0, 1), got {self.min_edge}")
        if self.max_markets <= 0:
            raise ValueError(f"max_markets must be > 0, got {self.max_markets}")
        if not (0 < self.min_tradeable_price < self.max_tradeable_price < 1):
            raise ValueError(
                f"Need 0 < min_tradeable_price < max_tradeable_price < 1, "
                f"got [{self.min_tradeable_price}, {self.max_tradeable_price}]"
            )
        if self.max_spread <= 0 or self.max_spread >= 1:
            raise ValueError(f"max_spread must be in (0, 1), got {self.max_spread}")
        if self.bankroll_multiplier <= 0:
            raise ValueError(f"bankroll_multiplier must be > 0, got {self.bankroll_multiplier}")
        if self.log_format not in ("text", "json"):
            raise ValueError(f"log_format must be 'text' or 'json', got {self.log_format}")
        if not (0 < self.ob_imbalance_weight <= 0.20):
            raise ValueError(
                f"ob_imbalance_weight must be in (0, 0.20], got {self.ob_imbalance_weight}"
            )


def validate_live_config(clob: ClobConfig, chain: ChainConfig, bot: BotConfig) -> None:
    if bot.paper_mode:
        return
    errors = []
    if not chain.private_key:
        errors.append("PRIVATE_KEY is required for live trading")
    if not clob.api_key:
        errors.append("POLYMARKET_API_KEY is required for live trading")
    if not clob.api_secret:
        errors.append("POLYMARKET_API_SECRET is required for live trading")
    if not clob.api_passphrase:
        errors.append("POLYMARKET_API_PASSPHRASE is required for live trading")
    if errors:
        raise ValueError("Live mode config errors:\n  " + "\n  ".join(errors))


def load_config() -> Tuple[ClobConfig, ChainConfig, BotConfig]:
    load_dotenv()  # side effects belong in explicit callsites, not module scope

    clob = ClobConfig(
        api_key=os.getenv("POLYMARKET_API_KEY", ""),
        api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
    )
    chain = ChainConfig(
        rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
        chain_id=int(os.getenv("CHAIN_ID", str(POLYGON_CHAIN_ID))),
        private_key=os.getenv("PRIVATE_KEY", ""),
        proxy_address=os.getenv("PROXY_ADDRESS", ""),
    )
    bot = BotConfig(
        paper_mode=os.getenv("PAPER_MODE", "true").lower() in ("true", "1", "yes", "on"),
        scan_interval=int(os.getenv("SCAN_INTERVAL_SECONDS", "60")),
        max_position_usdc=float(os.getenv("MAX_POSITION_USDC", "100.0")),
        kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
        min_edge=float(os.getenv("MIN_EDGE", "0.02")),
        max_markets=int(os.getenv("MAX_MARKETS", "10")),
        db_path=os.getenv("DB_PATH", "data/bot.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "text"),
        ssvi_r2_threshold=float(os.getenv("SSVI_R2_THRESHOLD", "0.70")),
        slippage_spread_bps=int(os.getenv("SLIPPAGE_SPREAD_BPS", "50")),
        slippage_impact_bps=int(os.getenv("SLIPPAGE_IMPACT_BPS", "10")),
        http_timeout=float(os.getenv("HTTP_TIMEOUT", "30.0")),
        http_max_retries=int(os.getenv("HTTP_MAX_RETRIES", "3")),
        max_consecutive_failures=int(os.getenv("MAX_CONSECUTIVE_FAILURES", "5")),
        circuit_breaker_cooldown=int(os.getenv("CIRCUIT_BREAKER_COOLDOWN", "300")),
        min_tradeable_price=float(os.getenv("MIN_TRADEABLE_PRICE", "0.01")),
        max_tradeable_price=float(os.getenv("MAX_TRADEABLE_PRICE", "0.99")),
        max_spread=float(os.getenv("MAX_SPREAD", "0.10")),
        min_liquidity=float(os.getenv("MIN_LIQUIDITY", "5000.0")),
        bankroll_multiplier=float(os.getenv("BANKROLL_MULTIPLIER", "10.0")),
        ob_imbalance_weight=float(os.getenv("OB_IMBALANCE_WEIGHT", "0.05")),
    )
    return clob, chain, bot
