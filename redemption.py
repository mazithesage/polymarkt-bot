"""On-chain redemption of settled Polymarket positions via CTF contracts on Polygon."""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from config import ChainConfig, USDC_DECIMALS

logger = logging.getLogger("polymarket-bot")

# ERC-20 Transfer(address,address,uint256) event signature
_TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()

# Minimal ABIs — only the functions we actually call
CONDITIONAL_TOKENS_ABI = json.loads("""[
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "getConditionId",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "payoutNumerators",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "payoutDenominator",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")

NEG_RISK_ABI = json.loads("""[
    {
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "amounts", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]""")

# Gas estimation is generous — 300k covers worst-case multicall
REDEEM_GAS_LIMIT = 300_000
# Default max gas price ceiling: 500 gwei — protects against spikes
DEFAULT_MAX_FEE_GWEI = 500


@dataclass
class RedemptionResult:
    condition_id: str
    tx_hash: str
    status: str  # "SUCCESS", "FAILED", "NOT_RESOLVED"
    amount_redeemed: float
    gas_used: int


def get_web3(chain_config: ChainConfig) -> Web3:
    w3 = Web3(Web3.HTTPProvider(chain_config.rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


def is_condition_resolved(w3: Web3, chain_config: ChainConfig, condition_id: str) -> bool:
    ct = w3.eth.contract(
        address=Web3.to_checksum_address(chain_config.conditional_tokens),
        abi=CONDITIONAL_TOKENS_ABI,
    )
    try:
        condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))
        denominator = ct.functions.payoutDenominator(condition_bytes).call()
        return denominator > 0
    except Exception as e:
        logger.warning("Error checking condition resolution: %s", e)
        return False


def _eip1559_gas_params(w3: Web3, max_fee_gwei: int = DEFAULT_MAX_FEE_GWEI) -> dict:
    """Build EIP-1559 gas parameters with a ceiling.

    Uses the node's suggested priority fee and latest base fee to compute
    maxFeePerGas, capped at max_fee_gwei to protect against spikes.
    """
    max_fee_wei = w3.to_wei(max_fee_gwei, "gwei")
    try:
        priority_fee = w3.eth.max_priority_fee_per_gas
    except AttributeError:
        # Fallback for older web3 versions
        priority_fee = w3.to_wei(30, "gwei")

    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", 0)

    # maxFeePerGas = 2 * baseFee + priorityFee, capped at ceiling
    suggested = 2 * base_fee + priority_fee
    max_fee = min(suggested, max_fee_wei)

    # Priority fee must not exceed max fee
    priority_fee = min(priority_fee, max_fee)

    return {
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
    }


def redeem_positions(
    w3: Web3,
    chain_config: ChainConfig,
    condition_id: str,
    index_sets: Optional[List[int]] = None,
    neg_risk: bool = False,
    max_fee_gwei: int = DEFAULT_MAX_FEE_GWEI,
) -> RedemptionResult:
    if index_sets is None:
        index_sets = [1, 2]  # binary market default

    condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))
    account = w3.eth.account.from_key(chain_config.private_key)
    sender = chain_config.proxy_address or account.address
    gas_params = _eip1559_gas_params(w3, max_fee_gwei)

    if neg_risk:
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(chain_config.neg_risk_exchange),
            abi=NEG_RISK_ABI,
        )
        tx = contract.functions.redeemPositions(
            condition_bytes, index_sets
        ).build_transaction({
            "from": Web3.to_checksum_address(sender),
            "nonce": w3.eth.get_transaction_count(Web3.to_checksum_address(sender)),
            "gas": REDEEM_GAS_LIMIT,
            **gas_params,
        })
    else:
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(chain_config.conditional_tokens),
            abi=CONDITIONAL_TOKENS_ABI,
        )
        parent_collection = b"\x00" * 32  # null bytes32 for root collection
        tx = contract.functions.redeemPositions(
            Web3.to_checksum_address(chain_config.usdc_address),
            parent_collection,
            condition_bytes,
            index_sets,
        ).build_transaction({
            "from": Web3.to_checksum_address(sender),
            "nonce": w3.eth.get_transaction_count(Web3.to_checksum_address(sender)),
            "gas": REDEEM_GAS_LIMIT,
            **gas_params,
        })

    signed = w3.eth.account.sign_transaction(tx, chain_config.private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    amount = _decode_usdc_transfer(
        receipt, chain_config.usdc_address, sender
    )

    return RedemptionResult(
        condition_id=condition_id,
        tx_hash=tx_hash.hex(),
        status="SUCCESS" if receipt["status"] == 1 else "FAILED",
        amount_redeemed=amount,
        gas_used=receipt["gasUsed"],
    )


def _decode_usdc_transfer(receipt: dict, usdc_address: str, recipient: str) -> float:
    """Extract total USDC transferred to recipient from transaction receipt logs.

    Scans for ERC-20 Transfer events emitted by the USDC contract where
    topic2 (recipient) matches our address. Returns the sum in USDC
    (human-readable, not base units).
    """
    usdc_addr = usdc_address.lower()
    recipient_padded = "0x" + recipient.lower().replace("0x", "").zfill(64)
    total_base_units = 0

    for log_entry in receipt.get("logs", []):
        log_address = log_entry.get("address", "").lower()
        topics = [t.hex() if isinstance(t, bytes) else t for t in log_entry.get("topics", [])]
        if len(topics) < 3:
            continue
        if log_address != usdc_addr:
            continue
        if topics[0] != _TRANSFER_TOPIC:
            continue
        if topics[2].lower() != recipient_padded:
            continue
        raw_data = log_entry.get("data", "0x")
        if isinstance(raw_data, bytes):
            raw_data = raw_data.hex()
        raw_data = raw_data.replace("0x", "")
        if raw_data:
            total_base_units += int(raw_data, 16)

    return total_base_units / (10 ** USDC_DECIMALS)


class RedemptionManager:

    def __init__(self, chain_config: ChainConfig):
        self.chain_config = chain_config
        self._w3: Optional[Web3] = None

    @property
    def w3(self) -> Web3:
        if self._w3 is None:
            self._w3 = get_web3(self.chain_config)
        return self._w3

    def check_and_redeem(
        self, condition_id: str, neg_risk: bool = False
    ) -> RedemptionResult:
        if not is_condition_resolved(self.w3, self.chain_config, condition_id):
            return RedemptionResult(
                condition_id=condition_id, tx_hash="",
                status="NOT_RESOLVED", amount_redeemed=0.0, gas_used=0,
            )
        return redeem_positions(
            self.w3, self.chain_config, condition_id, neg_risk=neg_risk
        )
