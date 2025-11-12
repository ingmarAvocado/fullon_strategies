"""
Test the test infrastructure itself.
"""
import pytest


@pytest.mark.asyncio
async def test_database_creation(db_context):
    """Test that test database is created and accessible."""
    # Just check that we can access the database
    assert db_context is not None
    # Try a simple query
    result = await db_context.session.execute("SELECT 1")
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_strategy_factory(strategy_factory):
    """Test that strategy factory creates strategies."""
    strategy = await strategy_factory(name="Test Strategy", class_name="TestStrategy")
    assert strategy.name == "Test Strategy"
    assert strategy.class_name == "TestStrategy"


@pytest.mark.asyncio
async def test_feed_factory(feed_factory):
    """Test that feed factory creates feeds."""
    feed = await feed_factory(symbol="ETH/USDT", period="5m")
    assert feed.symbol == "ETH/USDT"
    assert feed.period == "5m"


@pytest.mark.asyncio
async def test_symbol_factory(symbol_factory):
    """Test that symbol factory creates symbols."""
    symbol = await symbol_factory(symbol="ADA/USDT", base_asset="ADA")
    assert symbol.symbol == "ADA/USDT"
    assert symbol.base_asset == "ADA"