"""Async HTTP retry with exponential backoff."""

import asyncio
import logging
from typing import Any, Optional

import aiohttp

logger = logging.getLogger("polymarket-bot")

# Poly's API has a habit of 502ing under load
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def retry_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    **kwargs: Any,
) -> aiohttp.ClientResponse:
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = await session.request(method, url, **kwargs)
            if resp.status not in RETRYABLE_STATUS_CODES:
                return resp

            body = await resp.text()
            last_exception = aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=f"HTTP {resp.status}: {body[:200]}",
            )
            if attempt < max_retries:
                delay = _compute_delay(resp, attempt, base_delay)
                logger.warning(
                    f"Retryable HTTP {resp.status} on {method} {url}, "
                    f"attempt {attempt + 1}/{max_retries}, sleeping {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            continue

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Request error on {method} {url}: {e}, "
                    f"attempt {attempt + 1}/{max_retries}, sleeping {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            continue

    raise last_exception  # type: ignore[misc]


def _compute_delay(
    resp: aiohttp.ClientResponse, attempt: int, base_delay: float
) -> float:
    retry_after = resp.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), base_delay)
        except ValueError:
            pass
    return base_delay * (2 ** attempt)
