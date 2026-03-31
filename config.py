"""
Configuration module for the Polymarket trading bot.
Loads settings from environment variables with sensible defaults.
Sourced from patterns in: perpetual-s/polymarket-python-infrastructure, demone456/kalshi-polymarket-bot
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()


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
    chain_id: int = 137
    private_key: str = ""
    proxy_address: str = ""
    ctf_exchange: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    neg_risk_exchange: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    conditional_tokens: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    usdc_address: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"


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


def load_config() -> Tuple[ClobConfig, ChainConfig, BotConfig]:
    clob = ClobConfig(
        api_key=os.getenv("POLYMARKET_API_KEY", ""),
        api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
    )
    chain = ChainConfig(
        rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
        chain_id=int(os.getenv("CHAIN_ID", "137")),
        private_key=os.getenv("PRIVATE_KEY", ""),
        proxy_address=os.getenv("PROXY_ADDRESS", ""),
    )
    bot = BotConfig(
        paper_mode=os.getenv("PAPER_MODE", "true").lower() == "true",
        scan_interval=int(os.getenv("SCAN_INTERVAL_SECONDS", "60")),
        max_position_usdc=float(os.getenv("MAX_POSITION_USDC", "100.0")),
        kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
        min_edge=float(os.getenv("MIN_EDGE", "0.02")),
        max_markets=int(os.getenv("MAX_MARKETS", "10")),
        db_path=os.getenv("DB_PATH", "data/bot.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
    return clob, chain, bot
