import asyncio
from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

from retry import retry_request


async def _async_noop(delay):
    pass


@pytest.mark.asyncio
class TestRetryRequest:
    async def test_success_no_retry(self):
        with aioresponses() as m:
            m.get("http://test.com/ok", payload={"result": "ok"})
            async with aiohttp.ClientSession() as session:
                resp = await retry_request(session, "GET", "http://test.com/ok")
                assert resp.status == 200

    async def test_retries_on_500(self):
        with aioresponses() as m:
            m.get("http://test.com/fail", status=500)
            m.get("http://test.com/fail", status=500)
            m.get("http://test.com/fail", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch("retry.asyncio.sleep", side_effect=_async_noop):
                    resp = await retry_request(
                        session, "GET", "http://test.com/fail",
                        max_retries=3, base_delay=0.0,
                    )
                    assert resp.status == 200

    async def test_retries_on_429(self):
        with aioresponses() as m:
            m.get("http://test.com/rate", status=429)
            m.get("http://test.com/rate", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch("retry.asyncio.sleep", side_effect=_async_noop):
                    resp = await retry_request(
                        session, "GET", "http://test.com/rate",
                        max_retries=2, base_delay=0.0,
                    )
                    assert resp.status == 200

    async def test_respects_retry_after_header(self):
        with aioresponses() as m:
            m.get("http://test.com/wait", status=429, headers={"Retry-After": "2"})
            m.get("http://test.com/wait", payload={"ok": True})
            sleep_calls = []

            async def mock_sleep(delay):
                sleep_calls.append(delay)

            async with aiohttp.ClientSession() as session:
                with patch("retry.asyncio.sleep", side_effect=mock_sleep):
                    resp = await retry_request(
                        session, "GET", "http://test.com/wait",
                        max_retries=2, base_delay=0.5,
                    )
                    assert resp.status == 200
                    assert len(sleep_calls) == 1
                    assert sleep_calls[0] >= 2.0

    async def test_non_retryable_4xx_returns_immediately(self):
        with aioresponses() as m:
            m.get("http://test.com/bad", status=400, payload={"error": "bad request"})
            async with aiohttp.ClientSession() as session:
                resp = await retry_request(session, "GET", "http://test.com/bad", max_retries=3)
                assert resp.status == 400

    async def test_exhausted_retries_raises(self):
        with aioresponses() as m:
            for _ in range(4):
                m.get("http://test.com/down", status=503)
            async with aiohttp.ClientSession() as session:
                with patch("retry.asyncio.sleep", side_effect=_async_noop):
                    with pytest.raises(aiohttp.ClientResponseError):
                        await retry_request(
                            session, "GET", "http://test.com/down",
                            max_retries=3, base_delay=0.0,
                        )

    async def test_retries_on_connection_error(self):
        with aioresponses() as m:
            m.get("http://test.com/conn", exception=aiohttp.ClientConnectionError())
            m.get("http://test.com/conn", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch("retry.asyncio.sleep", side_effect=_async_noop):
                    resp = await retry_request(
                        session, "GET", "http://test.com/conn",
                        max_retries=2, base_delay=0.0,
                    )
                    assert resp.status == 200

    async def test_retries_on_timeout(self):
        with aioresponses() as m:
            m.get("http://test.com/slow", exception=asyncio.TimeoutError())
            m.get("http://test.com/slow", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch("retry.asyncio.sleep", side_effect=_async_noop):
                    resp = await retry_request(
                        session, "GET", "http://test.com/slow",
                        max_retries=2, base_delay=0.0,
                    )
                    assert resp.status == 200

    async def test_post_method(self):
        with aioresponses() as m:
            m.post("http://test.com/submit", payload={"id": "123"})
            async with aiohttp.ClientSession() as session:
                resp = await retry_request(
                    session, "POST", "http://test.com/submit", json={"data": "test"},
                )
                assert resp.status == 200

    async def test_exponential_backoff_delays(self):
        with aioresponses() as m:
            m.get("http://test.com/exp", status=500)
            m.get("http://test.com/exp", status=500)
            m.get("http://test.com/exp", payload={"ok": True})
            sleep_calls = []

            async def mock_sleep(delay):
                sleep_calls.append(delay)

            async with aiohttp.ClientSession() as session:
                with patch("retry.asyncio.sleep", side_effect=mock_sleep):
                    await retry_request(
                        session, "GET", "http://test.com/exp",
                        max_retries=3, base_delay=1.0,
                    )
            assert sleep_calls[0] == pytest.approx(1.0)
            assert sleep_calls[1] == pytest.approx(2.0)
