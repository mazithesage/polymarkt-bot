"""
CLOB client wrapper for Polymarket order execution.
Sourced from: Polymarket/py-clob-client (official SDK patterns),
              hbr-l/polypy (simplified wrapper), Jonmaa/btc-polymarket-bot (order flow).

Handles: API authentication (HMAC), order building, EIP-712 signing,
         order submission, and order book queries.
"""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from eth_account import Account
from eth_account.messages import encode_typed_data

from config import ClobConfig, ChainConfig


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    GTC = "GTC"
    FOK = "FOK"
    GTD = "GTD"


@dataclass
class OrderBookSummary:
    token_id: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    bid_depth: float
    ask_depth: float

    @property
    def implied_probability(self) -> float:
        return self.mid_price


@dataclass
class Order:
    token_id: str
    side: Side
    price: float
    size: float
    order_type: OrderType = OrderType.GTC
    expiration: int = 0

    def to_dict(self) -> dict:
        return {
            "tokenID": self.token_id,
            "side": self.side.value,
            "price": str(self.price),
            "size": str(self.size),
            "orderType": self.order_type.value,
            "expiration": str(self.expiration),
        }


@dataclass
class OrderResult:
    order_id: str
    status: str
    token_id: str
    side: str
    price: float
    size: float
    timestamp: float


# EIP-712 domain and types for Polymarket Exchange contract
EXCHANGE_DOMAIN = {
    "name": "Polymarket CTF Exchange",
    "version": "1",
    "chainId": 137,
    "verifyingContract": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
}

ORDER_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Order": [
        {"name": "salt", "type": "uint256"},
        {"name": "maker", "type": "address"},
        {"name": "signer", "type": "address"},
        {"name": "taker", "type": "address"},
        {"name": "tokenId", "type": "uint256"},
        {"name": "makerAmount", "type": "uint256"},
        {"name": "takerAmount", "type": "uint256"},
        {"name": "expiration", "type": "uint256"},
        {"name": "nonce", "type": "uint256"},
        {"name": "feeRateBps", "type": "uint256"},
        {"name": "side", "type": "uint8"},
        {"name": "signatureType", "type": "uint8"},
    ],
}


def _price_to_amounts(price: float, size: float, side: Side) -> Tuple[int, int]:
    """Convert price/size to maker/taker amounts in USDC base units (6 decimals)."""
    usdc_decimals = 10**6
    if side == Side.BUY:
        maker_amount = int(price * size * usdc_decimals)
        taker_amount = int(size * usdc_decimals)
    else:
        maker_amount = int(size * usdc_decimals)
        taker_amount = int(price * size * usdc_decimals)
    return maker_amount, taker_amount


def build_order_message(
    order: Order,
    maker: str,
    signer: str,
    nonce: int = 0,
    fee_rate_bps: int = 0,
) -> dict:
    """Build an EIP-712 typed order message for signing."""
    maker_amount, taker_amount = _price_to_amounts(order.price, order.size, order.side)
    salt = int(time.time() * 1000)
    side_int = 0 if order.side == Side.BUY else 1
    return {
        "salt": salt,
        "maker": maker,
        "signer": signer,
        "taker": "0x0000000000000000000000000000000000000000",
        "tokenId": int(order.token_id) if order.token_id.isdigit() else 0,
        "makerAmount": maker_amount,
        "takerAmount": taker_amount,
        "expiration": order.expiration,
        "nonce": nonce,
        "feeRateBps": fee_rate_bps,
        "side": side_int,
        "signatureType": 0,
    }


def sign_order(order_message: dict, private_key: str, chain_id: int = 137) -> str:
    """Sign an order using EIP-712 typed data."""
    domain = {**EXCHANGE_DOMAIN, "chainId": chain_id}
    signable = encode_typed_data(
        full_message={
            "types": ORDER_TYPES,
            "primaryType": "Order",
            "domain": domain,
            "message": order_message,
        }
    )
    account = Account.from_key(private_key)
    signed = account.sign_message(signable)
    return signed.signature.hex()


def create_hmac_signature(
    api_secret: str, timestamp: str, method: str, path: str, body: str = ""
) -> str:
    """Create HMAC signature for CLOB API authentication."""
    message = timestamp + method.upper() + path + body
    return hmac.new(
        api_secret.encode(), message.encode(), hashlib.sha256
    ).hexdigest()


def create_auth_headers(api_key: str, api_secret: str, api_passphrase: str,
                        method: str, path: str, body: str = "") -> dict:
    """Build authenticated request headers for CLOB API."""
    timestamp = str(int(time.time()))
    signature = create_hmac_signature(api_secret, timestamp, method, path, body)
    return {
        "POLY_API_KEY": api_key,
        "POLY_SIGNATURE": signature,
        "POLY_TIMESTAMP": timestamp,
        "POLY_PASSPHRASE": api_passphrase,
        "Content-Type": "application/json",
    }


class ClobClient:
    """Async wrapper around the Polymarket CLOB API."""

    def __init__(self, clob_config: ClobConfig, chain_config: ChainConfig):
        self.config = clob_config
        self.chain = chain_config
        self.base_url = clob_config.base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict:
        return create_auth_headers(
            self.config.api_key, self.config.api_secret,
            self.config.api_passphrase, method, path, body,
        )

    async def get_order_book(self, token_id: str) -> OrderBookSummary:
        """Fetch order book for a token and return a summary."""
        session = await self._get_session()
        path = f"/book?token_id={token_id}"
        headers = self._auth_headers("GET", path)
        async with session.get(f"{self.base_url}{path}", headers=headers) as resp:
            data = await resp.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
        mid = (best_bid + best_ask) / 2
        return OrderBookSummary(
            token_id=token_id,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=best_ask - best_bid,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )

    async def submit_order(self, order: Order) -> OrderResult:
        """Sign and submit a GTC limit order to the CLOB."""
        account = Account.from_key(self.chain.private_key)
        maker = self.chain.proxy_address or account.address
        signer = account.address

        order_msg = build_order_message(order, maker, signer)
        signature = sign_order(order_msg, self.chain.private_key, self.chain.chain_id)

        payload = {
            "order": {
                **order.to_dict(),
                "salt": str(order_msg["salt"]),
                "maker": maker,
                "signer": signer,
                "signature": signature,
                "nonce": "0",
                "feeRateBps": "0",
                "signatureType": 0,
            },
            "orderType": order.order_type.value,
        }

        session = await self._get_session()
        path = "/order"
        body = json.dumps(payload)
        headers = self._auth_headers("POST", path, body)
        async with session.post(
            f"{self.base_url}{path}", headers=headers, data=body
        ) as resp:
            data = await resp.json()

        return OrderResult(
            order_id=data.get("orderID", ""),
            status=data.get("status", "UNKNOWN"),
            token_id=order.token_id,
            side=order.side.value,
            price=order.price,
            size=order.size,
            timestamp=time.time(),
        )

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        session = await self._get_session()
        path = f"/order/{order_id}"
        headers = self._auth_headers("DELETE", path)
        async with session.delete(f"{self.base_url}{path}", headers=headers) as resp:
            return await resp.json()

    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        session = await self._get_session()
        path = "/orders"
        headers = self._auth_headers("GET", path)
        async with session.get(f"{self.base_url}{path}", headers=headers) as resp:
            return await resp.json()
