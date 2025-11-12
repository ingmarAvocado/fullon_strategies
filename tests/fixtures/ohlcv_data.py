import os
import arrow
import pandas as pd
import pytest_asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import AsyncGenerator
from fullon_ohlcv.repositories.ohlcv import CandleRepository, TimeseriesRepository
from fullon_ohlcv.models import Candle
from fullon_log import get_component_logger

logger = get_component_logger("fullon.strategies.tests.fixtures")

# Load .env
env_path = Path(__file__).parent.parent.parent / ".env.test"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(f"Test environment file not found at: {env_path}")


def generate_candle_data(start_time, num_candles, base_price, price_increment, period_minutes):
    candles = []
    for i in range(num_candles):
        timestamp = start_time.shift(minutes=i * period_minutes)
        open_price = base_price + (i * price_increment)
        high_price = open_price * 1.002
        low_price = open_price * 0.998
        close_price = (open_price + high_price + low_price) / 3
        volume = 1000 + (i * 10)

        candles.append({
            "timestamp": timestamp.datetime,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        })
    return pd.DataFrame(candles)


@pytest_asyncio.fixture(scope="function")
async def ohlcv_test_data(db_context) -> AsyncGenerator[dict, None]:
    """
    Fixture providing pre-populated OHLCV data for multiple symbols and periods.
    """
    data = {}
    test_scenarios = [
        {"symbol": "BTC/USDT", "period": "minutes", "compression": 1, "num_candles": 150, "base_price": 60000},
        {"symbol": "BTC/USDT", "period": "minutes", "compression": 5, "num_candles": 150, "base_price": 60000},
        {"symbol": "ETH/USDT", "period": "minutes", "compression": 1, "num_candles": 120, "base_price": 3000},
    ]

    end_time = arrow.utcnow().floor('minute')

    for scenario in test_scenarios:
        symbol = scenario["symbol"]
        period = scenario["period"]
        compression = scenario["compression"]
        
        async with CandleRepository(
            exchange="kraken",
            symbol=symbol,
            test=True
        ) as repo:
            await repo.init_symbol()
            start_time = end_time.shift(minutes=-(scenario["num_candles"] * compression))
            df = generate_candle_data(
                start_time,
                scenario["num_candles"],
                scenario["base_price"],
                price_increment=0.1 if symbol == "BTC/USDT" else 0.05,
                period_minutes=compression
            )

            candles = [
                Candle(
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    vol=row['volume']
                ) for _, row in df.iterrows()
            ]
            await repo.save_candles(candles)

            key = f"{symbol}_{compression}{'m' if period == 'minutes' else 'h'}"
            data[key] = df

    yield data

    # Cleanup is implicitly handled by the test database teardown


@pytest_asyncio.fixture(scope="function")
async def empty_ohlcv_data() -> AsyncGenerator[None, None]:
    """
    Fixture providing a repository connection for a symbol with no data.
    This is useful for testing negative paths where data is expected to be absent.
    """
    async with CandleRepository(
        exchange="kraken",
        symbol="EMPTY/SYMBOL",
        test=True
    ) as repo:
        await repo.init_symbol()
        # This context provides a valid repository, but no data is populated.
        pass
    yield
