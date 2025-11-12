"""
Test the test infrastructure itself.
"""
import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_database_creation(db_context):
    """Test that test database is created and accessible."""
    # Just check that we can access the database
    assert db_context is not None
    # Try a simple query
    result = await db_context.session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_strategy_factory(strategy_factory):
    """Test that strategy factory creates strategies."""
    strategy = await strategy_factory(size=200.0)
    assert strategy.str_id is not None  # str_id is auto-generated integer
    assert strategy.size == 200.0


@pytest.mark.asyncio
async def test_feed_factory(feed_factory):
    """Test that feed factory creates feeds."""
    feed = await feed_factory(symbol="ETH/USDT", period="5m")
    assert feed.feed_id is not None  # feed_id is auto-generated
    assert feed.period == "5m"
    # symbol is handled specially in the factory and added via get_symbol method
    assert hasattr(feed, "get_symbol") and feed.get_symbol() == "ETH/USDT"


@pytest.mark.asyncio
async def test_symbol_factory(symbol_factory):
    """Test that symbol factory creates symbols."""
    symbol = await symbol_factory(symbol="ADA/USDT", base="ADA")
    assert symbol.symbol == "ADA/USDT"
    assert symbol.base == "ADA"