import pytest

from redemption import (
    CONDITIONAL_TOKENS_ABI,
    NEG_RISK_ABI,
    REDEEM_GAS_LIMIT,
    RedemptionManager,
    RedemptionResult,
)
from config import ChainConfig


class TestRedemptionResult:
    def test_result_fields(self):
        r = RedemptionResult(
            condition_id="0xabc", tx_hash="0x123",
            status="SUCCESS", amount_redeemed=100.0, gas_used=150000,
        )
        assert r.condition_id == "0xabc"
        assert r.status == "SUCCESS"
        assert r.gas_used == 150000

    def test_not_resolved_result(self):
        r = RedemptionResult(
            condition_id="0xabc", tx_hash="",
            status="NOT_RESOLVED", amount_redeemed=0.0, gas_used=0,
        )
        assert r.status == "NOT_RESOLVED"
        assert r.tx_hash == ""


class TestABIs:
    def test_conditional_tokens_abi_has_redeem(self):
        names = [entry["name"] for entry in CONDITIONAL_TOKENS_ABI if "name" in entry]
        assert "redeemPositions" in names

    def test_conditional_tokens_abi_has_payout(self):
        names = [entry["name"] for entry in CONDITIONAL_TOKENS_ABI if "name" in entry]
        assert "payoutDenominator" in names
        assert "payoutNumerators" in names

    def test_neg_risk_abi_has_redeem(self):
        names = [entry["name"] for entry in NEG_RISK_ABI if "name" in entry]
        assert "redeemPositions" in names


def test_gas_limit_is_reasonable():
    # HACK: this is really just a sanity check that nobody accidentally zeroes it out
    assert REDEEM_GAS_LIMIT >= 200_000
    assert REDEEM_GAS_LIMIT <= 1_000_000


class TestRedemptionManager:
    def test_init(self, chain_config):
        manager = RedemptionManager(chain_config)
        assert manager.chain_config == chain_config
        assert manager._w3 is None

    def test_check_and_redeem_no_connection(self, chain_config):
        # Without a real RPC, should gracefully return NOT_RESOLVED
        manager = RedemptionManager(chain_config)
        result = manager.check_and_redeem("0x" + "ab" * 32)
        assert result.status == "NOT_RESOLVED"
