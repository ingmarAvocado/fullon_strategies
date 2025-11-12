# Database Access Patterns

This document explains how to access the database in the fullon_strategies framework using the repository pattern with `DatabaseContext`.

**Last Updated**: 2025-01-11

---

## Core Principle

**Each database operation creates a fresh `DatabaseContext` for that operation.**

```python
from fullon_orm import DatabaseContext

async def some_method(self):
    # Create fresh context
    async with DatabaseContext() as db:
        # Perform database operations
        await db.orders.save_order(order)
        await db.commit()
    # Context automatically closes
```

---

## Why DatabaseContext Per Operation?

### Benefits
✅ **Clean transaction boundaries** - Each operation is atomic
✅ **No stale connections** - Fresh connection per operation
✅ **Error isolation** - If one operation fails, others aren't affected
✅ **Connection pooling** - DatabaseContext manages pool automatically
✅ **No resource leaks** - Context manager ensures cleanup

### Rejected Alternative
❌ **Strategy-level session** - Single DB session for entire strategy lifecycle
- Long-lived sessions become stale
- Unclear transaction boundaries
- Harder to recover from errors
- Connection pool exhaustion with many strategies

---

## Cache vs Database

**Key Principle: Cache is source of truth for real-time data, Database is source of truth for persistence.**

fullon_strategies uses both `fullon_cache` (Redis) and `fullon_orm` (PostgreSQL/TimescaleDB) for different purposes. Understanding when to use each is critical.

### When to Use Cache

Use `fullon_cache` for:

✅ **Real-time market data** - Current prices, tickers
✅ **Real-time account data** - Current positions, balances
✅ **Bot status updates** - Running state, heartbeat
✅ **High-frequency reads** - Data accessed every iteration
✅ **Cross-bot communication** - Shared state between strategies

```python
from fullon_cache import TickCache, AccountCache, BotCache

# Get current price (cache)
async with TickCache() as cache:
    tick = await cache.get_ticker("BTC/USDT", "kraken")
    current_price = tick.price

# Get current position (cache)
async with AccountCache() as cache:
    position = await cache.get_position(symbol="BTC/USDT", ex_id="1")
    if position and position.volume != 0:
        current_volume = position.volume

# Update bot status (cache)
async with BotCache() as cache:
    await cache.update_bot("bot_1", {"feed_1": {"status": "running"}})
```

### When to Use Database

Use `fullon_orm` + `DatabaseContext` for:

✅ **Persistence** - Orders, trades that must survive restarts
✅ **Historical data** - Trade history, PnL calculations
✅ **Audit trail** - Complete record of all operations
✅ **Reporting** - Generate reports from historical data
✅ **FIFO calculations** - TradeRepository calculates PnL automatically

```python
from fullon_orm import DatabaseContext
from fullon_orm.models import Order, Trade, DryTrade, Position

# Save order (database)
async with DatabaseContext() as db:
    order = Order(
        bot_id=1,
        symbol="BTC/USDT",
        side="buy",
        volume=0.5,
        status="New"
    )
    saved_order = await db.orders.save_order(order)
    await db.commit()

# Save trade with automatic PnL (database)
async with DatabaseContext() as db:
    trade = Trade(
        bot_id=1,
        symbol="BTC/USDT",
        side="buy",
        volume=0.5,
        price=42000.0,
        fee=21.0
    )
    saved_trade = await db.trades.save_trades([trade])
    await db.commit()
```

### Cache-Database Architecture

**Important: Cache and Database are INDEPENDENT systems with different data flows!**

fullon_strategies uses both cache (Redis) and database (PostgreSQL) for different purposes. They do NOT automatically sync with each other.

#### Data Flow Architecture

1. **Real-Time Market Data (TickCache)**
   - Populated by: `fullon_ticker_service` (reads from exchanges)
   - Source of truth: Live exchange prices
   - NOT synced from database
   - Strategy usage: Read current prices

2. **Real-Time Account Data (AccountCache)**
   - Populated by: `fullon_account_service` (reads from exchanges)
   - Source of truth: Live exchange positions/balances
   - NOT synced from database
   - Strategy usage: Validate current positions

3. **Persistent Data (Database)**
   - Populated by: Strategy writes (orders, trades)
   - Source of truth: Historical data, audit trail
   - NOT synced to cache
   - Strategy usage: Persistence, reporting, PnL calculation

#### Data Flow Diagram

```
Exchange APIs
     ↓
[fullon_ticker_service] → [TickCache (Redis)] → [Strategy reads prices]
[fullon_account_service] → [AccountCache (Redis)] → [Strategy reads positions]

[Strategy writes] → [Database (PostgreSQL)] → [Persisted orders/trades]
                         ↓
                    [Historical queries]
```

**Key Point:** Cache and Database are separate pipelines. Don't expect data written to database to appear in cache, or vice versa.

#### When to Use Cache vs Database

**Use Cache (Read-Only for Strategies):**
✅ Current market prices (TickCache.get_ticker)
✅ Current account positions (AccountCache.get_position)
✅ Bot status updates (BotCache.update_bot)
✅ High-frequency reads (every loop iteration)

**Use Database (Write Operations for Strategies):**
✅ Save orders (OrderRepository.save_order)
✅ Save trades (TradeRepository.save_trades)
✅ Query historical data (TradeRepository.get_trades)
✅ Generate reports (aggregate queries)

**Anti-Patterns (Don't Do This):**
❌ Write orders to database expecting them to appear in OrdersCache
❌ Query database for current prices (use TickCache instead)
❌ Expect cache to persist across restarts (it's ephemeral)
❌ Manually sync cache and database (they're independent)

```python
# ✅ CORRECT: Read from cache, write to database
async def update_variables(self):
    # Read current price from TickCache (populated by ticker_service)
    async with TickCache() as cache:
        tick = await cache.get_ticker("BTC/USDT", "kraken")
        if tick:
            self.current_price = tick.price

async def place_order(self, feed_num: int, side: str, size: float):
    # Write order to database for persistence
    async with DatabaseContext() as db:
        order = Order(...)
        await db.orders.save_order(order)
        await db.commit()
    # Order now persisted in database
    # Cache and database are INDEPENDENT - no automatic sync

# ❌ WRONG: Expecting database write to update cache
async def place_order(self, feed_num: int, side: str, size: float):
    # Save to database
    async with DatabaseContext() as db:
        order = Order(...)
        await db.orders.save_order(order)
        await db.commit()

    # This order will NOT appear in OrdersCache automatically!
    # Cache is fed by external services, not database writes
```

### Decision Table

| Data Type | Read From | Write To | Reason |
|-----------|-----------|----------|--------|
| Current price | Cache (TickCache) | N/A (exchange → ticker_service → cache) | Real-time exchange data |
| Current position | Cache (AccountCache) | N/A (exchange → account_service → cache) | Real-time exchange positions |
| Bot status | Cache (BotCache) | Cache (BotCache) | Bot monitoring, heartbeat |
| New order | N/A | Database (OrderRepository) | Persistence, audit trail |
| Filled trade | N/A | Database (TradeRepository) | Persistence, PnL calculation |
| Historical trades | Database (TradeRepository) | N/A | Reporting, PnL analysis |
| Strategy config | Database (StrategyRepository) | Database (StrategyRepository) | Persistence across restarts |

### Common Patterns

#### Pattern 1: Get Real-Time Price

```python
# Always use cache for current prices
async def get_current_price(self, feed_num: int) -> float:
    feed = self._get_feed_by_order(feed_num)

    async with TickCache() as cache:
        tick = await cache.get_ticker(feed.symbol, feed.exchange.name)
        if tick:
            return tick.price

    # Fallback to last known price from buffer
    if self.tick_buffer[feed_num]:
        return self.tick_buffer[feed_num][-1]

    return 0.0
```

#### Pattern 2: Validate Position

```python
# Cross-check in-memory position with cache
async def validate_position(self, feed_num: int) -> bool:
    feed = self._get_feed_by_order(feed_num)

    # Get cached position (synced from exchange by account_service)
    async with AccountCache() as cache:
        # CRITICAL: get_position() takes POSITIONAL args (symbol, ex_id)
        # Always returns Position, never None
        cached_pos = await cache.get_position(
            feed.symbol,      # Positional arg 1
            str(feed.ex_id)   # Positional arg 2
        )

    # Compare with in-memory position
    # Check volume to see if position actually exists (get_position always returns Position)
    if self.position[feed_num] and cached_pos.volume != 0:
        volume_diff = abs(self.position[feed_num].volume - cached_pos.volume)
        if volume_diff > 0.0001:
            self.logger.warning(
                "Position mismatch",
                feed_num=feed_num,
                memory=self.position[feed_num].volume,
                cache=cached_pos.volume
            )
            return False

    return True
```

#### Pattern 3: Place Order (Cache + Database)

```python
# Write to database, read from cache
async def place_order(self, feed_num: int, side: str, size: float) -> int:
    feed = self._get_feed_by_order(feed_num)

    # 1. Get current price from cache
    async with TickCache() as cache:
        tick = await cache.get_ticker(feed.symbol, feed.exchange.name)
        current_price = tick.price if tick else 0.0

    # 2. Write order to database for persistence
    async with DatabaseContext() as db:
        order = Order(
            bot_id=self.strategy_orm.bot_id,
            uid=self.strategy_orm.bot.user_id,
            ex_id=feed.ex_id,
            symbol=feed.symbol,
            side=side,
            volume=size,
            price=current_price,
            status="New"
        )
        saved_order = await db.orders.save_order(order)
        await db.commit()

    # Order now persisted in database
    # Cache and database are INDEPENDENT - no automatic sync
    return saved_order.order_id
```

### Best Practices

**DO:**
✅ Read prices from TickCache for current market data
✅ Read positions from AccountCache for current account state
✅ Write orders/trades to database for persistence
✅ Update bot status in BotCache for monitoring
✅ Trust that cache syncs automatically - don't duplicate writes

**DON'T:**
❌ Query database for current prices (too slow, stale data)
❌ Query database for current positions during trading loop
❌ Manually sync cache and database (automatic syncing handles this)
❌ Write to cache what should be persisted (use database for persistence)
❌ Use database for high-frequency reads (use cache for real-time data)

---

## Historical OHLCV Data vs Persistent Trading Data

fullon_strategies uses **TWO separate database systems** for different data types.

### fullon_orm (PostgreSQL) - Persistent Trading Data

**Use for:**
- Orders (Order model)
- Trades (Trade model)
- Positions (Position model - in-memory, calculated from trades)
- Strategy configuration (Strategy, Feed, Bot models)
- Audit trail and compliance

**Access via:**
```python
from fullon_orm import DatabaseContext
from fullon_orm.models import Order, Trade

async with DatabaseContext() as db:
    order = await db.orders.save_order(order_obj)
    await db.commit()
```

### fullon_ohlcv (TimescaleDB) - Historical Time-Series Data

**Use for:**
- Historical candle data (OHLCV)
- Trade history from exchanges (raw trades)
- Continuous aggregates (pre-computed candles)
- Time-series queries and analytics

**Access via:**
```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

# NOTE: Table/view initialization handled by fullon_ohlcv_service
async with TimeseriesRepository(exchange="kraken", symbol="BTC/USDT", test=False) as repo:
    df = await repo.fetch_ohlcv_df(
        compression=5,
        period="minutes",
        fromdate=arrow.utcnow().shift(days=-7),
        todate=arrow.utcnow()
    )
```

### Decision Table

| Data Type | Database | Repository | Model | Purpose |
|-----------|----------|------------|-------|---------|
| Strategy order | fullon_orm | OrderRepository | Order | Trading execution |
| Executed trade | fullon_orm | TradeRepository | Trade | PnL calculation (FIFO) |
| Position state | fullon_orm | N/A | Position (in-memory) | Current position tracking |
| Historical candles | fullon_ohlcv | TimeseriesRepository | DataFrame | Indicator calculations |
| Exchange trades | fullon_ohlcv | TradeRepository | Trade (OHLCV) | Raw trade data |
| Continuous aggregates | fullon_ohlcv | TimeseriesRepository | Materialized view | Fast OHLCV queries |

### Common Pattern: Combine Both

```python
# Strategy trading logic
async def on_signal(self, feed_num: int):
    """Enter position based on historical analysis."""

    feed = self._get_feed_by_order(feed_num)

    # 1. Load historical data from fullon_ohlcv
    async with TimeseriesRepository(
        exchange=feed.exchange.name,
        symbol=feed.symbol.symbol,
        test=False
    ) as repo:
        # Get last 100 candles for analysis
        df = await repo.fetch_ohlcv_df(
            compression=5,
            period="minutes",
            fromdate=arrow.utcnow().shift(hours=-10),
            todate=arrow.utcnow()
        )

    # 2. Calculate indicators
    df.ta.rsi(length=14, append=True)
    rsi = df['RSI_14'].iloc[-1]

    # 3. Place order via fullon_orm
    if rsi < 30:
        async with DatabaseContext() as db:
            order = Order(
                bot_id=self.strategy_orm.bot_id,
                symbol=feed.symbol.symbol,
                side="buy",
                volume=0.5
            )
            await db.orders.save_order(order)
            await db.commit()
```

### Key Differences

**fullon_orm (PostgreSQL):**
- Row-oriented (individual records)
- ACID transactions (strong consistency)
- Persistent state (survives restarts)
- Used for: trading operations, audit trail

**fullon_ohlcv (TimescaleDB):**
- Time-series oriented (temporal queries)
- TimescaleDB optimizations (hypertables, continuous aggregates)
- Analytics-focused (fast aggregations)
- Used for: historical analysis, backtesting, indicators

---

## Repository Lifecycle Management

When loading OHLCV data using `TimeseriesRepository`, strategies need to decide whether to create repositories per-operation or reuse them across multiple operations.

### Pattern 1: Per-Operation Repository (Recommended)

Create a fresh repository for each data fetch operation:

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

async def update_dataframe(self):
    """Fetch new candles for all feeds."""
    for feed in self.feeds:
        feed_num = feed.order

        if feed.period == "tick":
            continue

        # Create fresh repository per operation
        # NOTE: Table/view initialization handled by fullon_ohlcv_service
        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            # Define time range
            end_time = arrow.utcnow()
            start_time = end_time.shift(hours=-1)  # Last hour

            # Fetch data
            df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=start_time,
                todate=end_time
            )

            # Update dataframe
            self.dataframe[feed_num] = df
```

**Advantages:**
- Clean transaction boundaries
- Automatic connection cleanup via context manager
- No stale connection issues
- Shared engine pattern still provides connection pooling
- Simple error handling (context manager handles cleanup)

**When to Use:**
- Default choice for most strategies
- Periodic updates (every minute, every bar)
- Standard trading strategies

### Pattern 2: Reused Repository (Advanced)

For high-frequency updates, create repositories once and reuse them:

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

class HighFrequencyStrategy(BaseStrategy):
    async def init(self):
        """Initialize strategy with persistent repositories."""
        await super().init()

        # Create repositories for all OHLCV feeds
        self._repos = {}

        for feed in self.feeds:
            feed_num = feed.order

            if feed.period == "tick":
                continue

            # Create repository (without context manager)
            # NOTE: Table/view initialization handled by fullon_ohlcv_service
            repo = TimeseriesRepository(
                exchange=feed.exchange.name,
                symbol=feed.symbol.symbol,
                test=False
            )

            # Manual initialization
            await repo.initialize()

            # Store for reuse
            self._repos[feed_num] = repo

    async def update_dataframe(self):
        """Fetch new candles using reused repositories."""
        for feed in self.feeds:
            feed_num = feed.order

            if feed_num not in self._repos:
                continue

            # Reuse repository
            repo = self._repos[feed_num]

            # Define time range
            end_time = arrow.utcnow()
            start_time = end_time.shift(minutes=-5)

            # Fetch data
            df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=start_time,
                todate=end_time
            )

            # Update dataframe
            self.dataframe[feed_num] = df

    async def cleanup(self):
        """Close all repositories before shutdown."""
        for repo in self._repos.values():
            await repo.close()

        await super().cleanup()
```

**Advantages:**
- Eliminates repository initialization overhead
- Faster for very high-frequency updates (sub-minute)
- Useful for tick-level strategies

**Disadvantages:**
- Manual lifecycle management required
- Must implement cleanup() to close repositories
- Potential for stale connections if errors occur
- More complex error handling

**When to Use:**
- High-frequency strategies (updates every few seconds)
- Backtesting with rapid iteration
- Performance-critical applications (after profiling shows repository creation overhead)

### Connection Pooling: Shared Engine Pattern

**Important**: Regardless of which pattern you use, TimeseriesRepository uses a shared engine pattern that pools database connections:

```python
# Multiple repositories share the same connection pool
async with TimeseriesRepository(exchange="kraken", symbol="BTC/USDT", test=False) as repo1:
    df1 = await repo1.fetch_ohlcv_df(...)

async with TimeseriesRepository(exchange="kraken", symbol="ETH/USDT", test=False) as repo2:
    df2 = await repo2.fetch_ohlcv_df(...)

# Both repo1 and repo2 use the same underlying connection pool
# This provides 83% reduction in database connections
```

**Key Insight**: Even with Pattern 1 (per-operation), you get connection pooling benefits. The overhead is repository object creation, NOT connection establishment.

### Performance Comparison

Based on fullon_ohlcv benchmarks:

| Pattern | Repository Creation | Connection Usage | Best For |
|---------|-------------------|------------------|----------|
| Per-Operation | ~0.1-0.5ms overhead | Shared pool (83% reduction) | Standard strategies (1-60s updates) |
| Reused | One-time initialization | Shared pool (83% reduction) | High-frequency (< 1s updates) |

**Recommendation**: Start with Pattern 1 (per-operation). Only switch to Pattern 2 if profiling shows repository creation is a bottleneck.

### Error Handling Comparison

**Pattern 1 (Per-Operation):**
```python
async def update_dataframe(self):
    """Simple error handling with context manager."""
    for feed in self.feeds:
        try:
            async with TimeseriesRepository(...) as repo:
                df = await repo.fetch_ohlcv_df(...)
                self.dataframe[feed.order] = df
        except Exception as e:
            self.logger.error("Failed to fetch OHLCV", feed_num=feed.order, error=str(e))
            # Context manager automatically closes connection
```

**Pattern 2 (Reused):**
```python
async def update_dataframe(self):
    """Complex error handling with manual cleanup."""
    for feed in self.feeds:
        feed_num = feed.order
        repo = self._repos.get(feed_num)

        if not repo:
            continue

        try:
            df = await repo.fetch_ohlcv_df(...)
            self.dataframe[feed_num] = df
        except Exception as e:
            self.logger.error("Failed to fetch OHLCV", feed_num=feed_num, error=str(e))

            # Manual recovery: close and recreate repository
            try:
                await repo.close()
            except:
                pass

            # Recreate repository
            new_repo = TimeseriesRepository(
                exchange=feed.exchange.name,
                symbol=feed.symbol.symbol,
                test=False
            )
            await new_repo.initialize()
            self._repos[feed_num] = new_repo
```

**Key Takeaway**: Pattern 1 is simpler and sufficient for most use cases. Use Pattern 2 only when profiling shows a clear performance bottleneck.

---

## Available Repositories

```python
async with DatabaseContext() as db:
    # Order management
    db.orders          # OrderRepository

    # Trade management
    db.trades          # TradeRepository

    # Strategy management
    db.strategies      # StrategyRepository

    # Bot management
    db.bots            # BotRepository

    # Feed management
    db.feeds           # FeedRepository

    # Symbol management
    db.symbols         # SymbolRepository

    # Exchange management
    db.exchanges       # ExchangeRepository
```

---

## Common Patterns

### Pattern 1: Save Order

```python
async def place_order(self, feed_num: int, side: str, size: float,
                     order_type: str = "market", price: Optional[float] = None) -> int:
    """Place order and save to database."""

    # Get feed information
    feed = next((f for f in self.feeds if f.order == feed_num), None)
    if not feed:
        raise ValueError(f"Feed {feed_num} not found")

    # Create fresh database context
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
            volume=size,  # CRITICAL: Use 'volume' NOT 'amount'
            price=price,
            status="New",
            futures=False,
            leverage=self.leverage[feed_num] if feed_num in self.leverage else None,
            reason=f"Strategy signal: {self.signal[feed_num]}"
        )

        # Save via repository
        saved_order = await db.orders.save_order(order)

        # Commit transaction
        await db.commit()

        self.logger.info(
            "Order saved",
            order_id=saved_order.order_id,
            symbol=feed.symbol,
            side=side,
            volume=size
        )

        return saved_order.order_id
```

### Pattern 2: Update Order Status

```python
async def on_order(self, feed_num: int, order: Order) -> None:
    """Callback when order status changes."""

    # Fresh context for update operation
    async with DatabaseContext() as db:
        # Update order status
        await db.orders.update_order_status(
            order.order_id,
            order.status,
            ex_order_id=order.ex_order_id
        )

        # If partially filled, update final_volume
        if order.final_volume:
            await db.orders.update_order_final_volume(
                order.order_id,
                order.final_volume
            )

        await db.commit()

    # Update in-memory tracking
    self.last_orders[feed_num].append(order)
    self.last_orders[feed_num] = self.last_orders[feed_num][-220:]  # Keep last 220

    self.logger.info(
        "Order status updated",
        order_id=order.order_id,
        status=order.status
    )
```

### Pattern 3: Save Trade (with automatic PnL calculation)

```python
async def on_trade(self, feed_num: int, trade: Trade) -> None:
    """Handle trade execution - saves to DB and updates position."""

    # Fresh context for trade save
    async with DatabaseContext() as db:
        # Save trade (auto-calculates PnL via FIFO matching!)
        success = await db.trades.save_trades([trade])

        if not success:
            self.logger.error(
                "Failed to save trade",
                trade_id=trade.ex_trade_id,
                symbol=trade.symbol
            )
            return

        await db.commit()

    # Update in-memory position
    if feed_num not in self.position or self.position[feed_num] is None:
        # Initialize position
        feed = next((f for f in self.feeds if f.order == feed_num), None)
        self.position[feed_num] = Position(
            symbol=trade.symbol,
            ex_id=str(trade.ex_id)
        )

    # Add trade to position (converts to signed volume)
    volume = trade.volume if trade.side.lower() == "buy" else -trade.volume
    self.position[feed_num].add_trade(
        volume=volume,
        price=trade.price,
        fee=trade.fee
    )

    # Calculate current PnL
    current_price = self.tick_buffer[feed_num][-1] if self.tick_buffer[feed_num] else trade.price
    pnl_data = self.position[feed_num].calculate_pnl(current_price)
    self.pnl[feed_num] = pnl_data['unrealized_pnl']

    self.logger.info(
        "Trade processed",
        trade_id=trade.ex_trade_id,
        symbol=trade.symbol,
        side=trade.side,
        volume=trade.volume,
        price=trade.price,
        unrealized_pnl=self.pnl[feed_num]
    )
```

### Pattern 4: Query Orders

```python
async def get_open_orders(self, feed_num: int) -> List[Order]:
    """Get all open orders for a feed."""

    feed = next((f for f in self.feeds if f.order == feed_num), None)
    if not feed:
        return []

    # Fresh context for query
    async with DatabaseContext() as db:
        # Query open orders for this bot
        open_orders = await db.orders.get_open_orders(
            bot_id=self.strategy_orm.bot_id
        )

        # Filter by symbol
        feed_orders = [
            order for order in open_orders
            if order.symbol == feed.symbol
        ]

    return feed_orders
```

### Pattern 5: Query Trades

```python
async def get_recent_trades(self, feed_num: int, limit: int = 100) -> List[Trade]:
    """Get recent trades for a feed."""

    feed = next((f for f in self.feeds if f.order == feed_num), None)
    if not feed:
        return []

    # Fresh context for query
    async with DatabaseContext() as db:
        # Get trades for this exchange
        all_trades = await db.trades.get_trades(
            ex_id=feed.ex_id,
            symbol=feed.symbol
        )

    # Return most recent trades
    return sorted(all_trades, key=lambda t: t.time, reverse=True)[:limit]
```

### Pattern 6: Save Dry Trade (Paper Trading)

```python
async def save_dry_trade(self, feed_num: int, side: str, volume: float,
                        price: float, closingtrade: bool = False) -> bool:
    """Save simulated trade for backtesting/paper trading."""

    feed = next((f for f in self.feeds if f.order == feed_num), None)
    if not feed:
        return False

    # Fresh context
    async with DatabaseContext() as db:
        # Create dry trade
        dry_trade = DryTrade(
            bot_id=self.strategy_orm.bot_id,
            uid=self.strategy_orm.bot.user_id,
            ex_id=feed.ex_id,
            cat_ex_id=feed.cat_ex_id,
            symbol=feed.symbol,
            side=side,
            volume=volume,
            price=price,
            cost=volume * price,
            fee=volume * price * 0.001,  # Assume 0.1% fee
            closingtrade=closingtrade,
            reason=f"Dry run: {self.signal[feed_num]}"
        )

        # Save (auto-calculates PnL if closingtrade=True)
        success = await db.trades.save_dry_trade(dry_trade)

        await db.commit()

    return success
```

---

## Error Handling

### Pattern 1: Basic Try-Except

```python
async def place_order(self, feed_num: int, side: str, size: float) -> Optional[int]:
    """Place order with error handling."""

    try:
        async with DatabaseContext() as db:
            order = Order(...)
            saved_order = await db.orders.save_order(order)
            await db.commit()
            return saved_order.order_id

    except Exception as e:
        self.logger.error(
            "Failed to place order",
            feed_num=feed_num,
            side=side,
            size=size,
            error=str(e)
        )
        return None
```

### Pattern 2: Rollback on Error

```python
async def complex_operation(self):
    """Operation that may need rollback."""

    try:
        async with DatabaseContext() as db:
            # Multiple operations
            await db.orders.save_order(order1)
            await db.orders.save_order(order2)

            # Some business logic that might fail
            if not self.validate_orders():
                raise ValueError("Order validation failed")

            await db.commit()

    except Exception as e:
        # Context manager automatically rolls back on exception
        self.logger.error("Operation failed, rolled back", error=str(e))
        raise
```

### Pattern 3: Retry Logic

```python
from asyncio import sleep

async def save_with_retry(self, order: Order, max_retries: int = 3) -> Optional[int]:
    """Save order with retry logic."""

    for attempt in range(max_retries):
        try:
            async with DatabaseContext() as db:
                saved_order = await db.orders.save_order(order)
                await db.commit()
                return saved_order.order_id

        except Exception as e:
            self.logger.warning(
                "Save failed, retrying",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(e)
            )

            if attempt < max_retries - 1:
                await sleep(2 ** attempt)  # Exponential backoff
            else:
                self.logger.error("Save failed after retries", error=str(e))
                return None
```

---

## Critical Field Names

### Order Model

**CRITICAL: Use `volume` field, NOT `amount`**

```python
# ✅ CORRECT
order = Order(
    volume=0.5,  # Correct field name
    side="buy"
)

# ❌ WRONG
order = Order(
    amount=0.5,  # This field doesn't exist!
    side="buy"
)
```

**All Order fields:**
```python
Order(
    bot_id=int,              # Required
    uid=int,                 # Required
    ex_id=int,               # Required
    cat_ex_id=int,           # Required
    exchange=str,            # Required (denormalized)
    symbol=str,              # Required (e.g., "BTC/USD")
    order_type=str,          # Required ("market", "limit", "stop")
    side=str,                # Required ("buy", "sell")
    volume=float,            # Required - USE THIS, NOT 'amount'
    final_volume=float,      # Optional - actual filled volume
    price=float,             # Optional - None for market orders
    plimit=float,            # Optional - price limit for conditional
    tick=float,              # Optional - minimum price movement
    futures=bool,            # Optional - default False
    leverage=float,          # Optional - None for spot
    status=str,              # Optional - default "New"
    command=str,             # Optional - strategy command
    reason=str,              # Optional - human-readable reason
    ex_order_id=str,         # Optional - set by exchange
)
```

### Trade Model

**All Trade fields:**
```python
Trade(
    ex_trade_id=str,         # Required - exchange's trade ID
    ex_order_id=str,         # Required - exchange's order ID
    uid=int,                 # Required
    ex_id=int,               # Required
    cat_ex_id=int,           # Required
    symbol=str,              # Required
    order_type=str,          # Required
    side=str,                # Required ("buy", "sell")
    volume=float,            # Required
    price=float,             # Required
    cost=float,              # Required (volume * price)
    fee=float,               # Required
    leverage=float,          # Optional - default 1.0
    cur_volume=float,        # Optional - current position volume
    cur_avg_price=float,     # Optional - current avg position price
    cur_avg_cost=float,      # Optional - current avg position cost
    cur_fee=float,           # Optional - cumulative fees
    roi=float,               # AUTO-CALCULATED by save_trades()
    roi_pct=float,           # AUTO-CALCULATED by save_trades()
    total_fee=float,         # Optional - total fees including funding
)
```

---

## Position Model (In-Memory Only)

**Position is NOT persisted to database** - it's an in-memory dataclass for tracking.

```python
from fullon_orm.models import Position

# Create position
position = Position(
    symbol="BTC/USD",
    ex_id="1"
)

# Add trades
position.add_trade(volume=0.5, price=67000.0, fee=10.0)   # Buy
position.add_trade(volume=0.3, price=68000.0, fee=6.0)    # Add to position
position.add_trade(volume=-0.2, price=69000.0, fee=5.0)   # Partial close

# Calculate PnL
current_price = 69500.0
pnl_data = position.calculate_pnl(current_price)

print(pnl_data)
# {
#   'unrealized_pnl': 1234.56,
#   'realized_pnl': 456.78,
#   'total_pnl': 1691.34,
#   'pnl_percentage': 5.23
# }

# Check position state
print(f"Volume: {position.volume}")          # Current volume
print(f"Avg price: {position.avg_price}")     # Average entry price
print(f"Is open: {position.is_open}")         # True if volume != 0
print(f"Is long: {position.is_long}")         # True if volume > 0
print(f"Is short: {position.is_short}")       # True if volume < 0
```

---

## Repository Methods Reference

### OrderRepository

```python
async with DatabaseContext() as db:
    # Create/Update
    await db.orders.save_order(order: Order) -> Order
    await db.orders.update_order_status(oid: int, status: str, ex_order_id: Optional[str]) -> None
    await db.orders.update_order_final_volume(order_id: int, final_volume: float) -> None
    await db.orders.update_orders_status(bot_id: int, status: str, restrict: Optional[str]) -> None

    # Query
    await db.orders.get_order(ex_order_id: int) -> Optional[Order]
    await db.orders.get_open_orders(uid: Optional[int], ex_id: Optional[int], bot_id: Optional[int]) -> List[Order]
    await db.orders.get_all_orders(uid: Optional[int], ex_id: Optional[int], status: Optional[str]) -> List[Order]
    await db.orders.get_last_order(bot_id: int) -> Optional[Order]
```

### TradeRepository

```python
async with DatabaseContext() as db:
    # Create/Update (Live Trades)
    await db.trades.save_trades(trades: List[Trade]) -> bool  # Auto-calculates PnL!
    await db.trades.update_trade(trade: Trade) -> None
    await db.trades.delete_trade(trade_id: int) -> bool

    # Create/Update (Dry Trades)
    await db.trades.save_dry_trade(dry_trade: DryTrade) -> bool  # Auto-calculates PnL!
    await db.trades.update_dry_trade(trade_id: int, changes: Dict[str, Any]) -> bool
    await db.trades.delete_dry_trades(bot_id: int) -> bool

    # Query
    await db.trades.get_trades(ex_id: int, last: bool, uncalculated: bool, symbol: Optional[str]) -> List[Trade]
    await db.trades.get_last_dry_trade(bot_id: int, symbol: str, ex_id: int) -> Optional[DryTrade]
```

---

## Best Practices

### DO:
✅ Use `async with DatabaseContext() as db:` for every database operation
✅ Commit transactions with `await db.commit()` before context closes
✅ Use `volume` field in Order (NOT `amount`)
✅ Let TradeRepository calculate PnL automatically
✅ Use Position model for in-memory tracking
✅ Handle exceptions and log errors
✅ Use retry logic for critical operations

### DON'T:
❌ Keep DatabaseContext open for entire strategy lifecycle
❌ Share DatabaseContext across methods
❌ Use `amount` field in Order (doesn't exist)
❌ Try to persist Position model to database (it's in-memory only)
❌ Forget to call `await db.commit()`
❌ Ignore database errors silently

---

## Complete Example

```python
class MyStrategy(BaseStrategy):
    async def on_signal(self, feed_num: int):
        """Complete example with database operations."""

        if self.signal[feed_num] != "buy":
            return

        # Get feed info
        feed = next((f for f in self.feeds if f.order == feed_num), None)
        if not feed:
            self.logger.error("Feed not found", feed_num=feed_num)
            return

        # Calculate position size
        size = self.calculate_position_size(feed_num)
        current_price = self.tick_buffer[feed_num][-1]

        # Set risk parameters
        self.stop_loss[feed_num] = current_price * 0.98
        self.take_profit[feed_num] = current_price * 1.06

        # Place order with database persistence
        try:
            async with DatabaseContext() as db:
                # Create order
                order = Order(
                    bot_id=self.strategy_orm.bot_id,
                    uid=self.strategy_orm.bot.user_id,
                    ex_id=feed.ex_id,
                    cat_ex_id=feed.cat_ex_id,
                    exchange=feed.exchange.name,
                    symbol=feed.symbol,
                    order_type="market",
                    side="buy",
                    volume=size,
                    status="New",
                    reason=f"RSI oversold: {self.signal[feed_num]}"
                )

                # Save order
                saved_order = await db.orders.save_order(order)
                await db.commit()

                self.logger.info(
                    "Order placed",
                    order_id=saved_order.order_id,
                    symbol=feed.symbol,
                    size=size,
                    price=current_price
                )

                # Initialize position tracking
                self.position[feed_num] = Position(
                    symbol=feed.symbol,
                    ex_id=str(feed.ex_id)
                )

                # If dry run, simulate immediate fill
                if self.dry_run:
                    await self._simulate_fill(feed_num, saved_order, current_price)

        except Exception as e:
            self.logger.error(
                "Failed to place order",
                feed_num=feed_num,
                error=str(e)
            )

    async def _simulate_fill(self, feed_num: int, order: Order, fill_price: float):
        """Simulate order fill for dry run mode."""

        # Update order status
        async with DatabaseContext() as db:
            await db.orders.update_order_status(
                order.order_id,
                "Filled",
                ex_order_id=f"DRY_{order.order_id}"
            )
            await db.commit()

        # Create simulated trade
        trade = Trade(
            ex_trade_id=f"DRY_TRADE_{order.order_id}_{int(time.time())}",
            ex_order_id=f"DRY_{order.order_id}",
            uid=order.uid,
            ex_id=order.ex_id,
            cat_ex_id=order.cat_ex_id,
            symbol=order.symbol,
            order_type=order.order_type,
            side=order.side,
            volume=order.volume,
            price=fill_price,
            cost=order.volume * fill_price,
            fee=order.volume * fill_price * 0.001,
            leverage=1.0
        )

        # Save trade (calls on_trade internally)
        await self.on_trade(feed_num, trade)
```

---

## Summary

Database access in fullon_strategies follows a simple pattern:

1. **Fresh context per operation**: `async with DatabaseContext() as db:`
2. **Use repositories**: `db.orders`, `db.trades`, etc.
3. **Commit explicitly**: `await db.commit()`
4. **Handle errors**: Try-except with logging
5. **Use correct field names**: `volume` not `amount`
6. **Let TradeRepository calculate PnL**: Automatic FIFO matching
7. **Position is in-memory only**: Not persisted to database

This pattern keeps database operations clean, predictable, and error-resistant.
