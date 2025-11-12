"""
Tests for the test data fixtures.
"""
import pytest
import pandas as pd
from fullon_orm.models import Tick

pytestmark = pytest.mark.asyncio


async def test_ohlcv_test_data_fixture(ohlcv_test_data):
    """
    Tests that the ohlcv_test_data fixture provides data.
    """
    assert isinstance(ohlcv_test_data, dict)
    assert "BTC/USDT_1m" in ohlcv_test_data
    assert "BTC/USDT_5m" in ohlcv_test_data
    assert "ETH/USDT_1m" in ohlcv_test_data

    df = ohlcv_test_data["BTC/USDT_1m"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100
    assert "close" in df.columns


async def test_empty_ohlcv_data_fixture(empty_ohlcv_data):
    """
    Tests that the empty_ohlcv_data fixture provides no data.
    """
    # This fixture doesn't return anything, it just sets up the context.
    # A real test would use this fixture and assert that a data loading operation
    # returns an empty result.
    assert empty_ohlcv_data is None


async def test_tick_test_data_fixture(tick_test_data):
    """
    Tests that the tick_test_data fixture provides data.
    """
    assert isinstance(tick_test_data, dict)
    assert "BTC/USDT" in tick_test_data
    assert "ETH/USDT" in tick_test_data

    ticks = tick_test_data["BTC/USDT"]
    assert isinstance(ticks, list)
    assert len(ticks) > 0
    assert isinstance(ticks[0], Tick)
