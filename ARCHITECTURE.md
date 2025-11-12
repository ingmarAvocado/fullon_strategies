# Fullon Strategies - Architecture Decisions

This document captures the core architectural decisions for the fullon_strategies framework. These decisions resolve contradictions between documentation and provide a single source of truth for implementation.

**Last Updated**: 2025-01-11
**Status**: Approved

---

## Overview

The fullon_strategies framework provides a polymorphic strategy system for trading bots. Strategies inherit from `BaseStrategy` and implement custom trading logic while BaseStrategy handles infrastructure, lifecycle management, risk controls, and database operations.

**Key Principles:**
- BaseStrategy owns the execution loop (autonomous operation)
- Strategies are self-contained and independent
- Database operations use repository pattern with context managers
- Feed management through utility class (composition over inheritance)
- Clear separation: BaseStrategy = infrastructure, ChildStrategy = logic

---

## Decision 1: Execution Model

### The Decision

**BaseStrategy has an internal `main_loop()` that drives execution autonomously.**

### Why It Matters

This determines how strategies are started, who controls execution flow, and how strategies interact with the rest of the Fullon ecosystem (fullon_bot, services, etc.).

### Chosen Approach: BaseStrategy-Driven (Internal Loop)

```python
# fullon_bot or launcher
strategy = MyStrategy(strategy_orm)
await strategy.init()
await strategy.run()  # Starts internal main_loop, runs until stopped
```

**BaseStrategy.run() implementation:**
```python
async def run(self):
    """Main execution loop - runs continuously until stopped."""
    self.logger.info("Strategy starting", str_id=self.str_id)

    while not self.shutting_down:
        try:
            # PRE-LOOP phase (already done in init)

            # IN-LOOP phase
            await self.main_loop()

        except Exception as e:
            self.logger.error("Strategy error", error=str(e))
            if self.should_halt(e):
                break

    # POST-LOOP phase
    await self.shutdown()

async def main_loop(self):
    """Single iteration of strategy loop."""
    # Step 0: validate_sync()
    await self.validate_sync()

    # Step 0.5: check_circuit_breakers()
    if not await self.check_circuit_breakers():
        return  # Circuit breaker triggered, skip this iteration

    # Step 1: update_dataframe()
    await self.update_dataframe()

    # Step 2: update_variables()
    await self.update_variables()

    # For each feed...
    for feed_num in self.non_tick_feeds:
        # Step 3: risk_management()
        await self.risk_management(feed_num)

        # Step 4: prepare_indicators() - CHILD IMPLEMENTS
        self.prepare_indicators()

        # Step 5: generate_signals() - CHILD IMPLEMENTS
        self.generate_signals()

        # Step 6: on_signal() or on_position() - CHILD IMPLEMENTS
        if self.position[feed_num] is None and self.signal[feed_num]:
            await self.on_signal(feed_num)
        elif self.position[feed_num] is not None:
            await self.on_position(feed_num)

    # Step 7: update_bot_status()
    await self.update_bot_status()

    # Step 8: emit_telemetry()
    await self.emit_telemetry()

    # Throttle loop to prevent CPU spinning
    await asyncio.sleep(self.loop_interval)
```

### Rejected Alternative: External Orchestration

```python
# REJECTED: External caller drives execution
while running:
    tick = await get_tick()
    await strategy.on_tick(tick)  # fullon_bot calls strategy
```

**Why rejected:**
- Strategy can't control its own lifecycle
- Harder to implement complex multi-step loops
- External caller needs to understand strategy internals
- Doesn't match STRATEGY_LIFECYCLE.md vision

### Implications

1. **fullon_bot role**: Simple launcher that creates strategy and calls `await strategy.run()`
2. **Strategy autonomy**: Each strategy is independent, can have different loop intervals
3. **Lifecycle control**: BaseStrategy fully controls PRE-LOOP, IN-LOOP, POST-LOOP phases
4. **Child implementation**: Child strategies just implement hooks (prepare_indicators, generate_signals, on_signal, on_position)

---

## Decision 2: Feed Numbering System

### The Decision

**`feed_num` is an alias for `Feed.order` from the database.**

### Why It Matters

Feed numbering affects how strategies access data, how dictionaries are keyed, and how child strategies map symbols to feeds.

### Chosen Approach: feed_num = Feed.order

```python
# Feed ORM model (from fullon_orm)
class Feed:
    feed_id: int      # Primary key (auto-increment)
    symbol: str       # e.g., "BTC/USD"
    period: str       # e.g., "1m", "5m", "tick"
    order: int        # Order of this feed in strategy (1, 2, 3, ...)
    # ...

# In BaseStrategy
class BaseStrategy:
    def __init__(self, strategy_orm: Strategy):
        # Load feeds
        self.feeds = strategy_orm.feeds_list

        # Initialize per-feed dictionaries using Feed.order as key
        for feed in self.feeds:
            feed_num = feed.order
            self.dataframe[feed_num] = pd.DataFrame()
            self.position[feed_num] = None
            self.signal[feed_num] = None
            # ...

    async def update_dataframe(self):
        """Update OHLCV data for all feeds."""
        for feed in self.feeds:
            feed_num = feed.order

            if feed.period == "tick":
                continue  # Tick feeds don't have dataframes

            # Load new candles
            new_candles = await self.feed_loader.get_latest_candles(feed)

            # Update dataframe
            self.dataframe[feed_num] = pd.concat([
                self.dataframe[feed_num],
                new_candles
            ]).tail(500)  # Keep last 500 bars
```

### Feed Access Pattern

```python
# Child strategy accessing feed data
class MyStrategy(BaseStrategy):
    def prepare_indicators(self):
        for feed in self.feeds:
            feed_num = feed.order

            if not self.bar_completed[feed_num]:
                continue

            # Access dataframe by feed_num
            df = self.dataframe[feed_num].copy()

            # Calculate indicators
            df.ta.rsi(length=14, append=True)

            self._dataframe[feed_num] = df
            self.bar_completed[feed_num] = False
```

### Why This Pattern?

1. **Simple**: No complex indexing, no separate mapping
2. **Explicit**: Feed.order is set by user/admin when configuring strategy
3. **Deterministic**: Order stays constant, predictable across restarts
4. **Database-backed**: feed.order is persisted, survives crashes

### Implications

1. **Feed.order must be unique per strategy**: Database constraint required
2. **Child strategies can reference feeds by order**: e.g., "feed 1 is BTC, feed 2 is ETH"
3. **Dictionary access is straightforward**: `self.dataframe[feed.order]`
4. **Symbol lookup when needed**: `feed.symbol` available on Feed object

---

## Required Imports for Database Operations

**All database operations in fullon_strategies require proper fullon_orm imports.**

### Core Imports

Every strategy that interacts with the database needs these imports:

```python
from fullon_orm import DatabaseContext
from fullon_orm.models import Order, Trade, Position
```

### Extended Imports

For specific features, import additional models:

```python
# Paper Trading / Backtesting
from fullon_orm.models import DryTrade

# Strategy Configuration (already available in BaseStrategy)
from fullon_orm.models import Bot, Strategy, Feed

# Advanced Features (rarely needed in child strategies)
from fullon_orm.models import Symbol, Exchange
```

### Import Location

- **BaseStrategy**: Already imports all required models in `base_strategy.py`
- **Child Strategies**: Only need to import models if overriding trading methods that use them
- **Examples in this document**: Assume imports are present unless shown explicitly

### Common Import Mistakes

❌ **Don't forget DryTrade when using dry_run mode:**
```python
# Will fail with NameError
dry_trade = DryTrade(...)  # DryTrade not imported!
```

✅ **Always import models you're using:**
```python
from fullon_orm.models import DryTrade

dry_trade = DryTrade(...)  # Works correctly
```

**Reference**: See `/home/ingmar/code/fullon2/fullon_orm/docs/FULLON_ORM_LLM_METHOD_REFERENCE.md` for complete ORM usage guide.

---

## Decision 3: Database Session Management

### The Decision

**Each BaseStrategy method that needs database access creates a fresh `DatabaseContext` for that operation.**

### Why It Matters

Database connection management affects performance, transaction boundaries, error handling, and connection pool exhaustion.

### Chosen Approach: DatabaseContext Per Operation

```python
async def place_order(self, feed_num: int, side: str, size: float,
                     order_type: str = "market", price: Optional[float] = None) -> int:
    """Place an order and save to database."""

    # Get feed information
    feed = self._get_feed_by_order(feed_num)

    # Create fresh database context for this operation
    async with DatabaseContext() as db:
        # Create Order ORM object
        order = Order(
            bot_id=self.strategy_orm.bot_id,
            uid=self.strategy_orm.bot.user_id,
            ex_id=feed.ex_id,
            cat_ex_id=feed.cat_ex_id,
            exchange=feed.exchange.name,
            symbol=feed.symbol,
            order_type=order_type,
            side=side,
            volume=size,  # CRITICAL: Use 'volume' not 'amount'
            price=price,
            status="New",
            reason=f"Strategy signal: {self.signal[feed_num]}"
        )

        # Save via repository
        saved_order = await db.orders.save_order(order)

        # Commit transaction
        await db.commit()

        self.logger.info(
            "Order placed",
            order_id=saved_order.order_id,
            symbol=feed.symbol,
            side=side,
            size=size
        )

        return saved_order.order_id
```

### Another Example: on_trade()

```python
async def on_trade(self, feed_num: int, trade: Trade) -> None:
    """Handle trade execution - save to DB and update position."""

    # Create fresh context
    async with DatabaseContext() as db:
        # Save trade (auto-calculates PnL via FIFO!)
        success = await db.trades.save_trades([trade])

        if not success:
            self.logger.error("Failed to save trade", trade_id=trade.ex_trade_id)
            return

        await db.commit()

    # Update in-memory position
    volume = trade.volume if trade.side == "buy" else -trade.volume
    self.position[feed_num].add_trade(
        volume=volume,
        price=trade.price,
        fee=trade.fee
    )

    # Calculate current PnL
    current_price = self.tick_buffer[feed_num][-1]
    pnl_data = self.position[feed_num].calculate_pnl(current_price)
    self.pnl[feed_num] = pnl_data['unrealized_pnl']

    self.logger.info(
        "Trade processed",
        symbol=trade.symbol,
        side=trade.side,
        volume=trade.volume,
        pnl=self.pnl[feed_num]
    )
```

### Why This Pattern?

1. **Clean boundaries**: Each operation is atomic with clear transaction scope
2. **No stale connections**: Fresh connection per operation prevents staleness
3. **Error isolation**: If one operation fails, doesn't affect others
4. **Connection pooling**: DatabaseContext manages pool, we don't worry about it

### Rejected Alternative: Strategy-Level DB Session

```python
# REJECTED: Single session for entire strategy lifecycle
class BaseStrategy:
    async def init(self):
        self.db = DatabaseContext()
        await self.db.__aenter__()
        # Session stays open for entire strategy lifetime
```

**Why rejected:**
- Long-lived sessions can become stale
- Transaction boundaries unclear
- Harder to recover from errors
- Connection pool exhaustion if many strategies

### Implications

1. **Every DB operation has context manager overhead**: Acceptable, operations are infrequent
2. **No cross-operation transactions**: Each operation is atomic (acceptable for our use case)
3. **Simple error handling**: Each operation fails independently
4. **Connection pool must be sized appropriately**: One connection per operation

---

## Decision 4: Position/Order/Trade Lifecycle

### The Decision

**Strategy calls its own trading methods synchronously. In dry_run mode, strategy simulates fills immediately.**

### Why It Matters

This determines when callbacks are invoked, who calls them, and how dry_run vs live trading differ.

### Chosen Approach: Synchronous Internal Calls

**Live Trading Flow:**
```python
# Child strategy decides to enter position
async def on_signal(self, feed_num: int):
    if self.signal[feed_num] == "buy":
        # Call BaseStrategy method
        order_id = await self.place_order(
            feed_num=feed_num,
            side="buy",
            size=0.5,
            order_type="market"
        )

        # BaseStrategy.place_order() does:
        # 1. Create Order ORM object
        # 2. Save to database
        # 3. Send to exchange (via exchange connector)
        # 4. Return order_id

# Later, exchange connector receives fill confirmation
# Exchange connector calls:
await strategy.on_order(feed_num, order_with_status_filled)

# Then exchange connector reports trade
# Exchange connector calls:
await strategy.on_trade(feed_num, executed_trade)
```

**Dry Run (Paper Trading) Flow:**
```python
# Child strategy decides to enter position
async def on_signal(self, feed_num: int):
    if self.signal[feed_num] == "buy":
        # Call BaseStrategy method (same as live)
        order_id = await self.place_order(
            feed_num=feed_num,
            side="buy",
            size=0.5,
            order_type="market"
        )

        # BaseStrategy.place_order() in dry_run mode does:
        # 1. Create Order ORM object (same)
        # 2. Save to database (same)
        # 3. IMMEDIATELY simulate fill (different!)
        # 4. Call self.on_order() synchronously
        # 5. Create simulated Trade
        # 6. Call self.on_trade() synchronously
        # 7. Return order_id
```

**Implementation in BaseStrategy:**
```python
async def place_order(self, feed_num: int, side: str, size: float,
                     order_type: str = "market", price: Optional[float] = None) -> int:
    """Place order - handles both live and dry_run."""

    feed = self._get_feed_by_order(feed_num)

    # Create and save order to database
    async with DatabaseContext() as db:
        order = Order(
            bot_id=self.strategy_orm.bot_id,
            # ... all fields ...
        )
        saved_order = await db.orders.save_order(order)
        await db.commit()
        order_id = saved_order.order_id

    if self.dry_run:
        # DRY RUN: Simulate immediate fill
        current_price = self.tick_buffer[feed_num][-1]
        fill_price = price if order_type == "limit" else current_price

        # Simulate order filled
        filled_order = saved_order
        filled_order.status = "Filled"
        filled_order.final_volume = size
        await self.on_order(feed_num, filled_order)

        # Simulate trade execution
        simulated_trade = Trade(
            ex_trade_id=f"DRY_{order_id}_{int(time.time())}",
            ex_order_id=f"DRY_ORDER_{order_id}",
            uid=self.strategy_orm.bot.user_id,
            ex_id=feed.ex_id,
            cat_ex_id=feed.cat_ex_id,
            symbol=feed.symbol,
            order_type=order_type,
            side=side,
            volume=size,
            price=fill_price,
            cost=size * fill_price,
            fee=size * fill_price * 0.001,  # Assume 0.1% fee
            leverage=self.leverage[feed_num] or 1.0
        )
        await self.on_trade(feed_num, simulated_trade)

    else:
        # LIVE: Send order to exchange
        # Exchange connector will call on_order() and on_trade() later
        await self.exchange_connector.submit_order(saved_order)

    return order_id
```

### Why This Pattern?

1. **Unified interface**: Child strategy code identical for dry_run and live
2. **Predictable**: Dry run gives immediate feedback, easy to test
3. **Simple**: No complex async callbacks from external services
4. **Testable**: Can test entire flow synchronously in dry_run mode

### Implications

1. **Exchange connector must call on_order/on_trade**: For live mode
2. **Dry run is fully synchronous**: Fills happen immediately
3. **Child strategy doesn't know the difference**: Same code for both modes
4. **Testing is easy**: Set dry_run=True and test entire flow

---

## Decision 5: Data Access Pattern

### The Decision

**Use FeedLoader utility class to load data from fullon_cache (ticks) and fullon_ohlcv (OHLCV).**

### Why It Matters

This determines how strategies access market data and how data is cached/refreshed.

### Chosen Approach: FeedLoader Utility (Composition)

```python
# In BaseStrategy
class BaseStrategy(ABC):
    def __init__(self, strategy_orm: Strategy):
        self.strategy_orm = strategy_orm
        self.feed_loader: Optional[FeedLoader] = None
        # ...

    async def init(self):
        """Initialize strategy - load feeds."""
        from .utils.feed_loader import FeedLoader

        # Create FeedLoader instance
        self.feed_loader = FeedLoader(self.strategy_orm)

        # Load all feeds
        await self.feed_loader.load_feeds()

        # Initialize per-feed dictionaries
        for feed in self.strategy_orm.feeds_list:
            feed_num = feed.order

            if feed.period == "tick":
                self.tick_buffer[feed_num] = []
            else:
                # Get initial OHLCV dataframe
                self.dataframe[feed_num] = self.feed_loader.get_feed(feed.feed_id)

# FeedLoader implementation
class FeedLoader:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.feeds: Dict[int, Union[Tick, pd.DataFrame]] = {}

    async def load_feeds(self):
        """Load all feeds for this strategy."""
        for feed in self.strategy.feeds_list:
            if feed.period == "tick":
                data = await self._load_tick(feed)
            else:
                data = await self._load_ohlcv(feed)

            self.feeds[feed.feed_id] = data

    async def _load_tick(self, feed: Feed) -> Optional[Tick]:
        """Load latest tick from fullon_cache."""
        from fullon_cache import TickCache

        async with TickCache() as cache:
            tick = await cache.get_ticker(feed.symbol)

        return tick

    async def _load_ohlcv(self, feed: Feed) -> pd.DataFrame:
        """Load OHLCV data from fullon_ohlcv using TimeseriesRepository."""
        from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
        import arrow

        # Create repository with feed configuration
        # NOTE: Table/view initialization handled by fullon_ohlcv_service
        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            # Define time range (last 500 bars as example)
            end_time = arrow.utcnow()
            hours_back = 500  # Approximate hours needed based on period
            start_time = end_time.shift(hours=-hours_back)

            # Fetch OHLCV data as DataFrame
            df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=start_time,
                todate=end_time
            )

            return df

    def get_feed(self, feed_id: int) -> Union[Tick, pd.DataFrame]:
        """Get loaded feed data."""
        return self.feeds.get(feed_id)
```

### Why TimeseriesRepository?

Time Series Repository (not CandleRepository) provides intelligent data source selection:

**Three-Tier Architecture:**
1. **Continuous aggregates** - Pre-computed candles in materialized views (50-90% faster)
2. **Candles tables** - Direct candle storage (medium speed)
3. **Trades tables** - On-the-fly aggregation from raw trades (slowest, always available)

**Benefits:**
- Automatic source selection based on availability
- Transparent fallback if preferred source unavailable
- Redis caching for 50-90% performance improvement on repeated queries
- Shared engine pattern reduces database connections by 83%
- Always uses fastest available data source

**Critical Notes:**
- Table/view initialization handled by `fullon_ohlcv_service` - no manual setup needed
- Use `arrow.Arrow` for timestamps (not datetime)
- Always use context manager (`async with`)
- Repository auto-selects optimal source - check `repo.primary_source` for diagnostics

### Why This Pattern?

1. **Separation of concerns**: FeedLoader handles data fetching, BaseStrategy handles logic
2. **Composition over inheritance**: FeedLoader is a utility, not a parent class
3. **Reusable**: FeedLoader can be used outside BaseStrategy if needed
4. **Testable**: Can mock FeedLoader easily

### Rejected Alternative: Direct Data Fetching

```python
# REJECTED: BaseStrategy directly fetches data
class BaseStrategy:
    async def update_dataframe(self):
        from fullon_ohlcv import CandleRepository
        repo = CandleRepository()
        df = await repo.get_candles(...)  # Direct fetching
```

**Why rejected:**
- Mixes concerns (strategy logic + data fetching)
- Harder to test
- Less flexible (can't swap data source easily)

### Implications

1. **FeedLoader is required dependency**: All strategies use it
2. **Initial load in init()**: Feeds loaded once at startup
3. **Refresh in main_loop()**: BaseStrategy calls FeedLoader to get latest data
4. **Caching handled by FeedLoader**: Can implement TTL cache if needed

---

## Continuous Aggregate Materialization Lag

### Understanding the Lag

Continuous aggregates are materialized views that update periodically (not instantly). This means:

- **Very recent data may not yet be in the aggregate** - Materialization happens at intervals
- **Queries spanning current time may miss latest candles** - Aggregate might be minutes behind
- **Fallback to trades table happens automatically** - TimeseriesRepository handles this transparently

### Why This Matters for Strategies

When a strategy queries OHLCV data including "now", it needs to understand that continuous aggregates may not include the most recent candles:

```python
async with TimeseriesRepository(
    exchange="kraken",
    symbol="BTC/USDT",
    test=False
) as repo:
    # Query for last 24 hours including current candle
    df = await repo.fetch_ohlcv_df(
        compression=5,
        period="minutes",
        fromdate=arrow.utcnow().shift(hours=-24),
        todate=arrow.utcnow()  # Including "now" - aggregate may lag
    )
```

### Automatic Fallback Behavior

TimeseriesRepository handles aggregate lag automatically through intelligent source selection:

1. **Try continuous aggregate first** (fastest, 50-90% faster)
2. **Detect if aggregate is incomplete** (missing recent candles)
3. **Fall back to candles table** (medium speed, up-to-date)
4. **Fall back to trades table** (slowest, always complete)

```python
async with TimeseriesRepository(
    exchange="kraken",
    symbol="BTC/USDT",
    test=False
) as repo:
    # Query recent data
    df = await repo.fetch_ohlcv_df(
        compression=5,
        period="minutes",
        fromdate=arrow.utcnow().shift(hours=-1),
        todate=arrow.utcnow()
    )

    # Repository automatically:
    # 1. Tries continuous aggregate
    # 2. Falls back to candles if aggregate incomplete
    # 3. Falls back to trades if needed

    # Check which source was actually used
    if repo.last_used_source != "continuous_aggregate":
        self.logger.debug(
            "Aggregate lag detected, using fallback",
            source=repo.last_used_source,
            feed_num=feed_num
        )
```

### Strategy Impact: No Action Needed for Most Cases

**For most strategies, no special handling is required:**

- TimeseriesRepository automatically selects the best source
- Fallback ensures complete data even with aggregate lag
- Performance is optimized transparently

```python
# Standard strategy pattern - works correctly with aggregate lag
async def update_dataframe(self):
    """Update OHLCV data for all feeds."""
    for feed in self.feeds:
        feed_num = feed.order

        if feed.period == "tick":
            continue

        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            # Get last hour of data
            end_time = arrow.utcnow()
            start_time = end_time.shift(hours=-1)

            # Automatic source selection handles aggregate lag
            df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=start_time,
                todate=end_time
            )

            self.dataframe[feed_num] = df
```

### Advanced Pattern: Split Historical and Real-Time

For strategies that need absolute latest data with maximum performance, split queries:

```python
async def update_dataframe_optimized(self):
    """
    Optimized loading: historical from aggregate, recent from real-time.

    Use this pattern when:
    - Strategy needs last 1000+ bars for indicators
    - Latest 1-10 candles must be real-time accurate
    - Performance is critical
    """
    for feed in self.feeds:
        feed_num = feed.order

        if feed.period == "tick":
            continue

        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            # 1. Load bulk historical data from continuous aggregate (fast)
            # Stop 15 minutes ago to ensure aggregate is materialized
            historical_end = arrow.utcnow().shift(minutes=-15)
            historical_start = historical_end.shift(days=-30)

            historical_df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=historical_start,
                todate=historical_end
            )

            # This will use continuous aggregate (fastest)
            self.logger.debug(
                "Historical data loaded",
                feed_num=feed_num,
                source=repo.last_used_source,  # Should be "continuous_aggregate"
                rows=len(historical_df)
            )

            # 2. Load recent data (last 15 minutes) from candles/trades
            recent_start = historical_end
            recent_end = arrow.utcnow()

            recent_df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=recent_start,
                todate=recent_end
            )

            # This will use candles or trades (most up-to-date)
            self.logger.debug(
                "Recent data loaded",
                feed_num=feed_num,
                source=repo.last_used_source,  # Likely "candles" or "trades"
                rows=len(recent_df)
            )

            # 3. Combine historical + recent
            import pandas as pd
            complete_df = pd.concat([historical_df, recent_df])

            # Remove duplicates (overlap at boundary)
            complete_df = complete_df.drop_duplicates(subset=['timestamp'], keep='last')

            # Store complete dataframe
            self.dataframe[feed_num] = complete_df
```

### Monitoring Aggregate Performance

Track which data sources are being used to detect aggregate lag patterns:

```python
async def update_dataframe_with_metrics(self):
    """Update dataframe with data source monitoring."""
    import time

    for feed in self.feeds:
        feed_num = feed.order

        if feed.period == "tick":
            continue

        start = time.time()

        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            # Fetch data
            df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=arrow.utcnow().shift(hours=-24),
                todate=arrow.utcnow()
            )

            elapsed_ms = (time.time() - start) * 1000

            # Log performance metrics
            self.logger.info(
                "OHLCV updated",
                feed_num=feed_num,
                rows=len(df),
                primary_source=repo.primary_source,      # Best available
                last_used_source=repo.last_used_source,  # Actually used
                elapsed_ms=int(elapsed_ms)
            )

            # Alert if aggregate not being used (possible lag or missing aggregate)
            if repo.primary_source == "continuous_aggregate" and \
               repo.last_used_source != "continuous_aggregate":
                self.logger.warning(
                    "Continuous aggregate available but not used - possible lag",
                    feed_num=feed_num,
                    primary_source=repo.primary_source,
                    last_used_source=repo.last_used_source
                )

            self.dataframe[feed_num] = df
```

### When to Use Each Pattern

| Pattern | Use Case | Performance | Complexity |
|---------|----------|-------------|------------|
| **Standard** (automatic fallback) | Most strategies | Good (automatic optimization) | Simple |
| **Split queries** (historical + recent) | High-performance strategies needing 1000+ bars | Excellent (aggregate for bulk, real-time for recent) | Medium |
| **Monitoring** (track sources) | Production strategies | Good (visibility into performance) | Simple |

### Key Takeaways

1. **Default pattern works for 95% of strategies** - Automatic fallback handles aggregate lag transparently
2. **No manual aggregate checks needed** - TimeseriesRepository handles source selection
3. **Split queries only for specific needs** - Use when loading large historical datasets + real-time data
4. **Monitor `last_used_source` in production** - Helps detect performance issues and aggregate availability

### Common Misconceptions

❌ **WRONG**: "I need to check if aggregates exist before querying"
```python
# Don't do this - TimeseriesRepository handles it
if repo.has_aggregate():
    df = await repo.fetch_from_aggregate(...)
else:
    df = await repo.fetch_from_candles(...)
```

✅ **RIGHT**: "TimeseriesRepository automatically selects the best source"
```python
# Just fetch - repository handles source selection
async with TimeseriesRepository(...) as repo:
    df = await repo.fetch_ohlcv_df(...)
    # Automatically uses best available source
```

❌ **WRONG**: "Aggregate lag means I get incomplete data"
```python
# No - fallback ensures complete data
df = await repo.fetch_ohlcv_df(...)  # Always complete, even with lag
```

✅ **RIGHT**: "Aggregate lag may affect performance, not completeness"
```python
# Correct understanding:
# - Aggregate lag → falls back to candles/trades (slower but complete)
# - No aggregate lag → uses aggregate (faster)
# - Either way, you get complete data
df = await repo.fetch_ohlcv_df(...)
```

---

## Decision 6: Signal Hooks Pattern

### The Decision

**BaseStrategy.main_loop() automatically calls child's on_signal() or on_position() based on position state.**

### Why It Matters

This determines how child strategies interact with BaseStrategy and when trading logic is executed.

### Chosen Approach: Automatic Orchestration

```python
# In BaseStrategy.main_loop()
async def main_loop(self):
    """Single iteration of strategy loop."""

    # ... validation, data updates, risk management ...

    # Child methods - called FOR EACH FEED
    for feed_num in self.non_tick_feeds:
        # Step 4: prepare_indicators() - child implements
        self.prepare_indicators()

        # Step 5: generate_signals() - child implements
        self.generate_signals()

        # Step 6: AUTOMATIC routing based on position state
        if self.position[feed_num] is None and self.signal[feed_num]:
            # No position, but signal exists → try to enter
            await self.on_signal(feed_num)

        elif self.position[feed_num] is not None:
            # Have position → manage it
            await self.on_position(feed_num)

        # If no position and no signal → do nothing
```

### Child Strategy Implementation

```python
class MyStrategy(BaseStrategy):
    def prepare_indicators(self):
        """Calculate indicators for ALL feeds."""
        for feed in self.feeds:
            feed_num = feed.order

            if not self.bar_completed[feed_num]:
                continue  # Skip if no new bar

            df = self.dataframe[feed_num].copy()
            df.ta.rsi(length=14, append=True)
            self._dataframe[feed_num] = df
            self.bar_completed[feed_num] = False

    def generate_signals(self):
        """Generate signals for ALL feeds."""
        for feed in self.feeds:
            feed_num = feed.order

            df = self._dataframe[feed_num]
            rsi = df['RSI_14'].iloc[-1]

            if rsi < 30:
                self.signal[feed_num] = "buy"
            elif rsi > 70:
                self.signal[feed_num] = "sell"
            else:
                self.signal[feed_num] = None

    async def on_signal(self, feed_num: int):
        """Handle signal when NO position exists."""
        if self.signal[feed_num] == "buy":
            # Calculate position size
            size = self.calculate_position_size(feed_num)

            # Set risk parameters
            current_price = self.tick_buffer[feed_num][-1]
            self.stop_loss[feed_num] = current_price * 0.98
            self.take_profit[feed_num] = current_price * 1.06

            # Place order (BaseStrategy handles this)
            await self.place_order(feed_num, "buy", size)

    async def on_position(self, feed_num: int):
        """Manage existing position."""
        # Check if base risk management triggered exit
        if self.exit_signal[feed_num]:
            await self.close_position(feed_num, reason=self.exit_reason[feed_num])
            return

        # Custom position management
        current_price = self.tick_buffer[feed_num][-1]
        pnl_pct = (current_price - self.position[feed_num].price) / self.position[feed_num].price

        # Update trailing stop if profitable
        if pnl_pct > 0.03:
            new_stop = current_price * 0.99
            if new_stop > self.stop_loss[feed_num]:
                self.stop_loss[feed_num] = new_stop
```

### Why This Pattern?

1. **Clear separation**: Child doesn't need to know when methods are called
2. **Automatic routing**: BaseStrategy decides on_signal vs on_position
3. **Simple child code**: Just implement 4 methods
4. **Consistent**: Same pattern for all strategies

### Implications

1. **Child strategies must implement 4 methods**: prepare_indicators, generate_signals, on_signal, on_position
2. **BaseStrategy orchestrates everything**: Child just responds to hooks
3. **No manual position checking**: BaseStrategy handles state management
4. **Works for all feed types**: Tick feeds optional (on_tick hook)

---

## Summary

These six architectural decisions provide a solid foundation for implementing the fullon_strategies framework:

1. ✅ **BaseStrategy-driven execution** - Autonomous strategies with internal loops
2. ✅ **Feed numbering via Feed.order** - Simple, deterministic feed access
3. ✅ **DatabaseContext per operation** - Clean, atomic database operations
4. ✅ **Synchronous internal callbacks** - Unified live/dry_run interface
5. ✅ **FeedLoader utility pattern** - Composition for data access
6. ✅ **Automatic hook orchestration** - BaseStrategy routes to child methods

All other documentation (CLAUDE.md, STRATEGY_LIFECYCLE.md) must align with these decisions.
