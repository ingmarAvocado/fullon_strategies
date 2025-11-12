import pytest
from fullon_orm.models import Exchange, Symbol, Feed, Strategy

pytestmark = pytest.mark.asyncio


async def test_exchange_factory_build(exchange_factory):
    """Test ExchangeFactory builds exchange without persistence."""
    exchange = exchange_factory.build(name="test_exchange")
    assert exchange.ex_id is None
    assert exchange.name == "test_exchange"
    assert exchange.cat_ex_id == 1


async def test_exchange_factory_create(db_context, exchange_factory, setup_base_data):
    """Test ExchangeFactory creates and persists exchange."""
    uid = setup_base_data['user'].uid
    exchange = await exchange_factory.create(db_context, uid=uid, name="binance")
    assert exchange.ex_id is not None
    assert exchange.name == "binance"

    # Verify it's in the DB
    retrieved = await db_context.exchanges.get_by_id(exchange.ex_id)
    assert retrieved is not None
    assert retrieved.name == "binance"


async def test_symbol_factory_build(symbol_factory):
    """Test SymbolFactory builds symbol without persistence."""
    symbol = symbol_factory.build(symbol="ETH/USDT")
    assert symbol.symbol_id is None
    assert symbol.symbol == "ETH/USDT"
    assert symbol.base == "ETH"
    assert symbol.quote == "USDT"


async def test_symbol_factory_create(db_context, symbol_factory, setup_base_data):
    """Test SymbolFactory creates and persists symbol."""
    cat_ex_id = setup_base_data['cat_ex'].cat_ex_id
    symbol = await symbol_factory.create(db_context, symbol="XRP/USD", cat_ex_id=cat_ex_id)
    assert symbol.symbol_id is not None
    assert symbol.symbol == "XRP/USD"
    assert symbol.cat_ex_id == cat_ex_id

    # Verify it's in the DB
    retrieved = await db_context.symbols.get_by_id(symbol.symbol_id)
    assert retrieved is not None
    assert retrieved.symbol == "XRP/USD"


async def test_feed_factory_build(feed_factory):
    """Test FeedFactory builds feed without persistence."""
    feed = feed_factory.build(str_id=1, symbol_id=1, ex_id=1, period="5m")
    assert feed.feed_id is None
    assert feed.period == "5m"
    assert feed.order is not None  # Should have a default


async def test_feed_factory_create(db_context, exchange_factory, symbol_factory, feed_factory, strategy_factory, setup_base_data):
    """Test FeedFactory creates and persists feed."""
    uid = setup_base_data['user'].uid
    cat_ex_id = setup_base_data['cat_ex'].cat_ex_id
    cat_str_id = setup_base_data['cat_str'].cat_str_id
    bot_id = setup_base_data['bot'].bot_id

    exchange = await exchange_factory.create(db_context, uid=uid)
    symbol = await symbol_factory.create(db_context, cat_ex_id=cat_ex_id)
    strategy = await strategy_factory.create(db_context, bot_id=bot_id, cat_str_id=cat_str_id)

    feed = await feed_factory.create(
        db_context,
        str_id=strategy.str_id,
        symbol_id=symbol.symbol_id,
        ex_id=exchange.ex_id,
        period="1h",
        order=2,
    )
    assert feed.feed_id is not None
    assert feed.period == "1h"
    assert feed.order == 2

    # Verify it's in the DB
    # Since there is no feed repository, we can't get it by id.
    # We can check if it's in the strategy's feeds list.
    retrieved_strategy = await db_context.strategies.get_by_id(strategy.str_id)
    assert retrieved_strategy is not None
    # The relationship might not be loaded automatically.
    # The check below depends on the ORM's loading strategy.
    # For now, we'll assume the creation is enough to trust it's in the DB.
    # A more robust test would involve a query.
    from sqlalchemy import select
    from fullon_orm.models import Feed
    stmt = select(Feed).where(Feed.feed_id == feed.feed_id)
    result = await db_context.session.execute(stmt)
    retrieved_feed = result.scalar_one_or_none()
    assert retrieved_feed is not None
    assert retrieved_feed.period == "1h"


async def test_strategy_factory_build(strategy_factory):
    """Test StrategyFactory builds strategy without persistence."""
    strategy = strategy_factory.build(cat_str_id=2)
    assert strategy.str_id is None
    assert strategy.cat_str_id == 2
    assert strategy.bot_id == 1


async def test_strategy_factory_create(db_context, strategy_factory, setup_base_data):
    """Test StrategyFactory creates and persists strategy."""
    bot_id = setup_base_data['bot'].bot_id
    cat_str_id = setup_base_data['cat_str'].cat_str_id
    strategy = await strategy_factory.create(db_context, bot_id=bot_id, cat_str_id=cat_str_id)
    assert strategy.str_id is not None
    assert strategy.bot_id == bot_id
    assert strategy.cat_str_id == cat_str_id

    # Verify it's in the DB
    retrieved = await db_context.strategies.get_by_id(strategy.str_id)
    assert retrieved is not None
    assert retrieved.cat_str_id == cat_str_id


async def test_factories_create_full_hierarchy(
    db_context, exchange_factory, symbol_factory, feed_factory, strategy_factory, setup_base_data
):
    """Test creating a full strategy with feeds using factories."""
    uid = setup_base_data['user'].uid
    cat_ex_id = setup_base_data['cat_ex'].cat_ex_id
    cat_str_id = setup_base_data['cat_str'].cat_str_id
    bot_id = setup_base_data['bot'].bot_id

    exchange = await exchange_factory.create(db_context, uid=uid, name="kraken")
    symbol = await symbol_factory.create(db_context, symbol="BTC/USDT", cat_ex_id=cat_ex_id)
    strategy = await strategy_factory.create(db_context, bot_id=bot_id, cat_str_id=cat_str_id)

    feed1 = await feed_factory.create(
        db_context,
        str_id=strategy.str_id,
        symbol_id=symbol.symbol_id,
        ex_id=exchange.ex_id,
        period="1m",
        order=1,
    )
    feed2 = await feed_factory.create(
        db_context,
        str_id=strategy.str_id,
        symbol_id=symbol.symbol_id,
        ex_id=exchange.ex_id,
        period="5m",
        order=2,
    )

    # Re-fetch strategy to confirm feeds are associated
    # Note: This depends on the ORM's relationship loading logic.
    # We will assume for now that we need to fetch them manually or the ORM handles it.
    from sqlalchemy import select
    from fullon_orm.models import Feed
    stmt = select(Feed).where(Feed.str_id == strategy.str_id)
    result = await db_context.session.execute(stmt)
    feeds = result.scalars().all()
    assert len(feeds) == 2
    assert feeds[0].feed_id == feed1.feed_id
    assert feeds[1].feed_id == feed2.feed_id
    assert feeds[0].order == 1
    assert feeds[1].order == 2
