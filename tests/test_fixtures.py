"""
Tests for the test data fixtures.
"""
import pytest
import pandas as pd
from fullon_orm.models import Tick
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
from fullon_cache import TickCache
import arrow

pytestmark = pytest.mark.asyncio


async def test_ohlcv_test_data_fixture(ohlcv_test_data, db_context):
    """
    Tests that the ohlcv_test_data fixture provides data and verifies database persistence.

    This test verifies that:
    1. The fixture returns properly structured data
    2. Data has been persisted to the database
    3. Data can be fetched back from the database independently
    """
    # Verify fixture returns correct structure
    assert isinstance(ohlcv_test_data, dict)
    assert "BTC/USDT_1m" in ohlcv_test_data
    assert "BTC/USDT_5m" in ohlcv_test_data
    assert "ETH/USDT_1m" in ohlcv_test_data

    # Verify data structure
    df = ohlcv_test_data["BTC/USDT_1m"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100
    assert "close" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "volume" in df.columns

    # Verify database persistence by independently fetching the data
    # This proves the data was actually saved to the database, not just returned in-memory
    end_time = arrow.utcnow().floor('minute')
    start_time = end_time.shift(minutes=-200)

    async with TimeseriesRepository(
        exchange="kraken",
        symbol="BTC/USDT",
        test=True
    ) as repo:
        # Fetch 1-minute candles
        fetched_1m = await repo.fetch_ohlcv_df(
            compression=1,
            period="minutes",
            fromdate=start_time,
            todate=end_time
        )

        # Verify we got data back from the database
        assert len(fetched_1m) > 0, "No 1m candles fetched - data not persisted to database"
        assert len(fetched_1m) >= 100, f"Expected at least 100 candles, got {len(fetched_1m)}"

        # Fetch 5-minute candles
        fetched_5m = await repo.fetch_ohlcv_df(
            compression=5,
            period="minutes",
            fromdate=start_time,
            todate=end_time
        )

        # Verify we got data back from the database
        assert len(fetched_5m) > 0, "No 5m candles fetched - data not persisted to database"
        assert len(fetched_5m) >= 20, f"Expected at least 20 5m candles, got {len(fetched_5m)}"

    # Verify ETH data
    async with TimeseriesRepository(
        exchange="kraken",
        symbol="ETH/USDT",
        test=True
    ) as repo:
        fetched_eth = await repo.fetch_ohlcv_df(
            compression=1,
            period="minutes",
            fromdate=start_time,
            todate=end_time
        )

        assert len(fetched_eth) > 0, "No ETH candles fetched - data not persisted to database"
        assert len(fetched_eth) >= 100, f"Expected at least 100 ETH candles, got {len(fetched_eth)}"


async def test_empty_ohlcv_data_fixture(empty_ohlcv_data):
    """
    Tests that the empty_ohlcv_data fixture provides no data.
    """
    # This fixture doesn't return anything, it just sets up the context.
    # A real test would use this fixture and assert that a data loading operation
    # returns an empty result.
    assert empty_ohlcv_data is None


async def test_tick_test_data_fixture(tick_test_data, cache_config):
    """
    Tests that the tick_test_data fixture persists data to the cache.
    """
    assert isinstance(tick_test_data, dict)
    assert "BTC/USDT" in tick_test_data

    # Verify data is in the cache
    async with TickCache() as cache:
        tick = await cache.get_ticker("BTC/USDT", exchange="kraken")
        assert isinstance(tick, Tick)
        assert tick.symbol == "BTC/USDT"
