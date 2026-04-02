import time

import pytest

from clob_client import (
    Order,
    OrderType,
    Side,
    USDC_BASE_UNITS,
    _price_to_amounts,
    build_order_message,
    create_auth_headers,
    create_hmac_signature,
    sign_order,
)


class TestOrderConstruction:
    def test_buy_order_to_dict(self):
        order = Order(token_id="12345", side=Side.BUY, price=0.55, size=10.0,
                      order_type=OrderType.GTC)
        d = order.to_dict()
        assert d["tokenID"] == "12345"
        assert d["side"] == "BUY"
        assert d["price"] == "0.55"
        assert d["size"] == "10.0"
        assert d["orderType"] == "GTC"

    def test_sell_order_to_dict(self):
        order = Order(token_id="67890", side=Side.SELL, price=0.70, size=5.0)
        assert order.to_dict()["side"] == "SELL"

    def test_order_default_type_is_gtc(self):
        order = Order(token_id="1", side=Side.BUY, price=0.5, size=1.0)
        assert order.order_type == OrderType.GTC

    def test_order_expiration_default_zero(self):
        order = Order(token_id="1", side=Side.BUY, price=0.5, size=1.0)
        assert order.expiration == 0


class TestOrderValidation:
    def test_price_zero_rejected(self):
        with pytest.raises(ValueError, match="price"):
            Order(token_id="1", side=Side.BUY, price=0.0, size=1.0)

    def test_price_one_rejected(self):
        with pytest.raises(ValueError, match="price"):
            Order(token_id="1", side=Side.BUY, price=1.0, size=1.0)

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="price"):
            Order(token_id="1", side=Side.BUY, price=-0.5, size=1.0)

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError, match="size"):
            Order(token_id="1", side=Side.BUY, price=0.5, size=-1.0)

    def test_zero_size_rejected(self):
        with pytest.raises(ValueError, match="size"):
            Order(token_id="1", side=Side.BUY, price=0.5, size=0.0)

    def test_valid_order_accepted(self):
        order = Order(token_id="1", side=Side.BUY, price=0.5, size=1.0)
        assert order.price == 0.5


class TestPriceToAmounts:
    def test_buy_amounts(self):
        maker, taker = _price_to_amounts(0.50, 10.0, Side.BUY)
        assert maker == 5_000_000   # 0.50 * 10.0 * USDC_BASE_UNITS
        assert taker == 10_000_000

    def test_sell_amounts(self):
        maker, taker = _price_to_amounts(0.50, 10.0, Side.SELL)
        assert maker == 10_000_000
        assert taker == 5_000_000

    def test_extreme_price_low(self):
        maker, taker = _price_to_amounts(0.01, 100.0, Side.BUY)
        assert maker == 1_000_000
        assert taker == 100_000_000

    def test_extreme_price_high(self):
        maker, taker = _price_to_amounts(0.99, 100.0, Side.BUY)
        assert maker == 99_000_000
        assert taker == 100_000_000


class TestOrderMessage:
    def test_build_buy_message(self):
        order = Order(token_id="12345", side=Side.BUY, price=0.50, size=10.0)
        msg = build_order_message(
            order,
            maker="0x1234567890abcdef1234567890abcdef12345678",
            signer="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
        )
        assert msg["maker"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert msg["signer"] == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        assert msg["tokenId"] == 12345
        assert msg["side"] == 0  # BUY
        assert msg["signatureType"] == 0
        assert msg["taker"] == "0x0000000000000000000000000000000000000000"

    def test_build_sell_message(self):
        order = Order(token_id="99999", side=Side.SELL, price=0.70, size=5.0)
        msg = build_order_message(order, maker="0x" + "aa" * 20, signer="0x" + "bb" * 20)
        assert msg["side"] == 1

    def test_salt_is_timestamp_based(self):
        order = Order(token_id="1", side=Side.BUY, price=0.5, size=1.0)
        msg = build_order_message(order, maker="0x" + "00" * 20, signer="0x" + "00" * 20)
        assert msg["salt"] > 0
        assert msg["salt"] < int(time.time() * 1000) + 1000


class TestEIP712Signing:
    PRIVATE_KEY = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

    def test_sign_produces_hex_signature(self):
        order = Order(token_id="12345", side=Side.BUY, price=0.50, size=10.0)
        msg = build_order_message(
            order,
            maker="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            signer="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        )
        sig = sign_order(msg, self.PRIVATE_KEY)
        assert isinstance(sig, str)
        assert len(sig) == 130  # 65 bytes hex

    def test_different_orders_different_signatures(self):
        order1 = Order(token_id="111", side=Side.BUY, price=0.50, size=10.0)
        order2 = Order(token_id="222", side=Side.SELL, price=0.70, size=5.0)
        msg1 = build_order_message(
            order1,
            maker="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            signer="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        )
        msg2 = build_order_message(
            order2,
            maker="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            signer="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        )
        assert sign_order(msg1, self.PRIVATE_KEY) != sign_order(msg2, self.PRIVATE_KEY)


class TestNonNumericTokenId:
    def test_non_numeric_token_id_logs_warning(self, caplog):
        """Non-numeric token_id should log a warning and use tokenId=0."""
        import logging
        order = Order(token_id="abc-not-numeric", side=Side.BUY, price=0.50, size=10.0)
        with caplog.at_level(logging.WARNING, logger="polymarket-bot"):
            msg = build_order_message(
                order,
                maker="0x" + "00" * 20,
                signer="0x" + "00" * 20,
            )
        assert msg["tokenId"] == 0
        assert "Non-numeric token_id" in caplog.text

    def test_numeric_token_id_no_warning(self, caplog):
        """Numeric token_id should not log any warning."""
        import logging
        order = Order(token_id="12345", side=Side.BUY, price=0.50, size=10.0)
        with caplog.at_level(logging.WARNING, logger="polymarket-bot"):
            msg = build_order_message(
                order,
                maker="0x" + "00" * 20,
                signer="0x" + "00" * 20,
            )
        assert msg["tokenId"] == 12345
        assert "Non-numeric" not in caplog.text


class TestHMACAuth:
    def test_hmac_signature_deterministic(self):
        sig1 = create_hmac_signature("secret", "12345", "GET", "/book")
        sig2 = create_hmac_signature("secret", "12345", "GET", "/book")
        assert sig1 == sig2

    def test_hmac_different_methods(self):
        assert (create_hmac_signature("secret", "12345", "GET", "/book")
                != create_hmac_signature("secret", "12345", "POST", "/book"))

    def test_auth_headers_contain_required_keys(self):
        headers = create_auth_headers("key", "secret", "pass", "GET", "/book")
        assert headers["POLY_API_KEY"] == "key"
        assert headers["POLY_PASSPHRASE"] == "pass"
        assert "POLY_SIGNATURE" in headers
        assert "POLY_TIMESTAMP" in headers
