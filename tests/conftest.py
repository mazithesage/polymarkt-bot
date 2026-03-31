"""Shared fixtures for tests."""

import os
import tempfile

import pytest

from config import BotConfig, ChainConfig, ClobConfig


@pytest.fixture
def clob_config():
    return ClobConfig(
        api_key="test_key",
        api_secret="test_secret",
        api_passphrase="test_pass",
    )


@pytest.fixture
def chain_config():
    return ChainConfig(
        rpc_url="http://localhost:8545",
        chain_id=137,
        private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        proxy_address="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
    )


@pytest.fixture
def bot_config(tmp_path):
    return BotConfig(
        paper_mode=True,
        scan_interval=5,
        max_position_usdc=100.0,
        kelly_fraction=0.25,
        min_edge=0.02,
        max_markets=5,
        db_path=str(tmp_path / "test_bot.db"),
        log_level="DEBUG",
    )


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")
