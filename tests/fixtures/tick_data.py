import os
import arrow
import pytest_asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import AsyncGenerator
from fullon_cache import TickCache
from fullon_orm.models import Tick
from fullon_log import get_component_logger

logger = get_component_logger("fullon.strategies.tests.fixtures")

# Load .env
env_path = Path(__file__).parent.parent.parent / ".env.test"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(f"Test environment file not found at: {env_path}")


@pytest_asyncio.fixture(scope="function")
async def tick_test_data(cache_config) -> AsyncGenerator[dict, None]:
    """
    Fixture providing pre-populated tick data in TickCache.

    Returns a dict with symbol -> list of Ticks.
    """
    data = {}
    test_scenarios = [
        {"symbol": "BTC/USDT", "num_ticks": 20, "base_price": 60100},
        {"symbol": "ETH/USDT", "num_ticks": 20, "base_price": 3010},
    ]

    # Populate cache with test data
    async with TickCache() as cache:
        await cache.flushdb()

        for scenario in test_scenarios:
            symbol = scenario["symbol"]
            ticks = []
            for i in range(scenario["num_ticks"]):
                tick_time = arrow.utcnow().shift(seconds=-scenario["num_ticks"] + i)
                tick = Tick(
                    symbol=symbol,
                    exchange="kraken",
                    price=scenario["base_price"] + (i * 0.5),
                    volume=1.0 + (i * 0.05),
                    time=tick_time.timestamp()
                )
                await cache.set_ticker(tick)
                ticks.append(tick)
            data[symbol] = ticks

    yield data

    # Cleanup after test
    async with TickCache() as cache:
        await cache.flushdb()
