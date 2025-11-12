---
description: Fullon Strategies - Dynamic strategy system for trading bots
globs:
  - "**/*.py"
  - "**/pyproject.toml"
  - "**/poetry.lock"
  - "tests/**/*.py"
  - "examples/**/*.py"
alwaysApply: true
---

## Project Overview

**Type:** Python Library
**Description:** Polymorphic strategy framework for trading bots with dynamic loading and feed management
**Primary Goal:** Provide base classes and utilities for building trading strategies that can access tick/OHLCV data
**Status:** Initial development - foundation phase

**⚠️ IMPORTANT**: Before implementing features, read [ARCHITECTURE.md](./ARCHITECTURE.md) for core architectural decisions that guide all development.

## Current Development Priorities

1. **Build BaseStrategy lifecycle** - Internal main_loop() with PRE-LOOP, IN-LOOP, POST-LOOP phases
2. **Implement trading methods** - open_position, close_position, on_order, on_trade, place_order
3. **Add risk management** - Stop loss, take profit, trailing stops, circuit breakers
4. **Create example strategies** - RSI, EMA crossover following lifecycle pattern
5. **Achieve comprehensive test coverage** - Real database tests following fullon_ohlcv patterns

**Note**: FeedLoader ✅ (done), StrategyLoader ✅ (done)

## Tech Stack

- **Language:** Python 3.13+ with type hints
- **Data Processing:** pandas (OHLCV DataFrames)
- **Async:** asyncio for async strategy methods
- **Testing:** pytest with real PostgreSQL/TimescaleDB (no mocking)
- **Dependencies:** fullon_cache (ticks), fullon_ohlcv (OHLCV), fullon_orm (models), fullon_log (logging)

## Architectural Decisions

**📖 READ [ARCHITECTURE.md](./ARCHITECTURE.md) FIRST** - Contains detailed rationale and examples.

**Quick Summary:**

1. **Execution Model**: BaseStrategy has internal `main_loop()` that drives execution autonomously
   - Pattern: `await strategy.run()` starts the strategy and runs until stopped
   - BaseStrategy orchestrates PRE-LOOP → IN-LOOP → POST-LOOP phases

2. **Feed Numbering**: `feed_num = Feed.order` from database
   - Simple alias, no complex indexing
   - Access data via `self.dataframe[feed.order]`

3. **Database Sessions**: `DatabaseContext` per operation
   - Each trading method creates fresh context: `async with DatabaseContext() as db:`
   - Clean transaction boundaries, no stale connections

4. **Position/Order/Trade Lifecycle**: Synchronous internal calls
   - Strategy calls `self.place_order()` → internally calls `self.on_order()` and `self.on_trade()`
   - Dry run simulates fills immediately

5. **Data Access**: FeedLoader utility (composition over inheritance)
   - `self.feed_loader = FeedLoader(strategy_orm)`
   - Loads from fullon_cache (ticks) and fullon_ohlcv (OHLCV)

6. **Signal Hooks**: BaseStrategy automatically routes to child methods
   - If no position + signal exists → calls `on_signal(feed_num)`
   - If position exists → calls `on_position(feed_num)`

## Architecture Overview

### Core Components

1. **BaseStrategy** (`src/fullon_strategies/base_strategy.py`)
   - Abstract parent class for all strategies
   - Has internal `main_loop()` that orchestrates execution (PRE-LOOP, IN-LOOP, POST-LOOP)
   - Child strategies implement: `prepare_indicators()`, `generate_signals()`, `on_signal()`, `on_position()`
   - Provides trading methods: `open_position()`, `close_position()`, `place_order()`, `on_order()`, `on_trade()`
   - Handles risk management: stop loss, take profit, trailing stops, circuit breakers
   - Manages feed data via FeedLoader utility

2. **FeedLoader** (`src/fullon_strategies/utils/feed_loader.py`)
   - Loads tick data from fullon_cache.TickCache
   - Loads OHLCV data from fullon_ohlcv.CandleRepository
   - Caches loaded feeds in memory
   - Provides `get_feed(feed_id)` method for strategy access

3. **StrategyLoader** (`src/fullon_strategies/strategy_loader.py`)
   - Dynamically loads strategy classes by name
   - Enables polymorphic execution (different strategies at runtime)
   - Used by fullon_bot to load strategies from database

### Key Design Patterns

- **Polymorphism**: Each strategy inherits from BaseStrategy but implements custom logic
- **Dynamic Loading**: Strategies loaded by class name at runtime
- **Composition**: FeedLoader is a utility used by strategies, not inheritance
- **Async-First**: All strategy methods are async

## Critical Requirements

### Async Operations
- **EVERYTHING is async** - All operations use async/await
- Strategy methods (on_tick, on_bar) are async
- FeedLoader methods are async
- Uses asyncio Tasks for continuous execution

### Testing Philosophy (FOLLOW fullon_ohlcv PATTERN)
- **NO MOCKING for database tests** - Use real PostgreSQL/TimescaleDB
- **Per-worker test databases** - Each pytest worker gets unique DB
- **Real data**: Load actual OHLCV data, create real ticks
- **Factory pattern**: Use factories for test data creation
- **Conftest structure**: Match fullon_ohlcv conftest.py pattern

### Code Quality Standards
- Follow PEP8 strictly
- Type hints for all functions/methods
- Comprehensive docstrings
- Robust error handling with fullon_log
- Clear separation of concerns

## Key Dependencies

### Required Libraries (MUST USE)
- **fullon_log**: Structured logging (MUST use get_component_logger)
- **fullon_orm**: Database models (Strategy, Feed, Symbol, Tick)
- **fullon_cache**: TickCache for real-time tick data
- **fullon_ohlcv**: CandleRepository for historical OHLCV data

### External Dependencies
- **pandas**: DataFrame operations (OHLCV data)
- **asyncio**: Async operations

## Data Flow

### Strategy Execution Flow

```
1. fullon_bot loads Strategy ORM object from database
2. StrategyLoader.load(strategy.class_name) → Returns strategy class
3. strategy_instance = StrategyClass(strategy_orm_object)
4. await strategy_instance.init()
   ├─> FeedLoader loads all feeds (tick + OHLCV)
   ├─> Initialize per-feed variables (position, dataframe, signals, etc.)
   └─> Strategy is ready
5. await strategy_instance.run()
   ├─> Starts internal main_loop()
   ├─> PRE-LOOP: Validation, service checks, parameter validation
   ├─> IN-LOOP: Continuous execution
   │   ├─> validate_sync()
   │   ├─> check_circuit_breakers()
   │   ├─> update_dataframe()
   │   ├─> update_variables()
   │   ├─> risk_management() [for each feed]
   │   ├─> prepare_indicators() [child implements]
   │   ├─> generate_signals() [child implements]
   │   ├─> on_signal() or on_position() [child implements]
   │   ├─> update_bot_status()
   │   └─> emit_telemetry()
   └─> POST-LOOP: Graceful shutdown when stopped
```

### FeedLoader Flow

```python
# FeedLoader loads data during strategy.init()
# Child strategy accesses via feed_num (which equals Feed.order)

# For tick feed (period == "tick")
async with TickCache() as cache:
    tick = await cache.get_ticker(feed.symbol)
    self.tick_buffer[feed.order].append(tick)  # feed_num = feed.order

# For OHLCV feed (period == "1m", "5m", etc.)
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

async with TimeseriesRepository(
    exchange=feed.exchange.name,
    symbol=feed.symbol.symbol,
    test=False
) as repo:
    # NOTE: init_symbol() handled automatically by fullon_ohlcv_service
    end_time = arrow.utcnow()
    start_time = end_time.shift(hours=-500)

    df = await repo.fetch_ohlcv_df(
        compression=feed.compression,
        period=feed.period,
        fromdate=start_time,
        todate=end_time
    )

    self.dataframe[feed.order] = df  # feed_num = feed.order
```

---

## Loading Historical OHLCV Data

**fullon_strategies uses `fullon_ohlcv.TimeseriesRepository` for loading historical candle data.**

### Why TimeseriesRepository?

TimeseriesRepository provides intelligent three-tier data selection:
1. **Continuous aggregates** (fastest, 50-90% faster when available)
2. **Candles tables** (medium speed, direct storage)
3. **Trades tables** (slowest, always available as fallback)

The repository automatically selects the fastest available source for each query.

### Basic Pattern

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

async def load_historical_data(feed: Feed) -> pd.DataFrame:
    """Load OHLCV data for a feed."""

    # Create repository with context manager
    # NOTE: Table/view initialization handled by fullon_ohlcv_service
    async with TimeseriesRepository(
        exchange=feed.exchange.name,  # e.g., "kraken"
        symbol=feed.symbol.symbol,    # e.g., "BTC/USDT"
        test=False  # Use production database
    ) as repo:
        # Define time range
        end_time = arrow.utcnow()
        start_time = end_time.shift(days=-30)  # Last 30 days

        # Fetch OHLCV as DataFrame
        df = await repo.fetch_ohlcv_df(
            compression=feed.compression,  # e.g., 5
            period=feed.period,           # e.g., "minutes"
            fromdate=start_time,
            todate=end_time
        )

        return df
```

### Feed Configuration Mapping

| Feed Field | TimeseriesRepository | Example |
|------------|---------------------|---------|
| `feed.exchange.name` | `exchange` parameter | "kraken" |
| `feed.symbol.symbol` | `symbol` parameter | "BTC/USDT" |
| `feed.compression` | `compression` parameter | 5 |
| `feed.period` | `period` parameter | "minutes" |

### Data Source Inspection

```python
async with TimeseriesRepository(exchange="kraken", symbol="BTC/USDT") as repo:
    # Check available data sources
    print(repo.data_sources)
    # {'continuous_aggregate': 'BTC_USDT_candles1m_view',
    #  'candles': 'BTC_USDT_candles1m',
    #  'trades': 'BTC_USDT_trades'}

    print(repo.primary_source)  # "continuous_aggregate" (fastest available)

    # After query
    ohlcv = await repo.fetch_ohlcv_df(...)
    print(repo.last_used_source)  # Which source was actually used
```

### Performance Features

- **Redis Caching**: Repeated queries 50-90% faster
- **Shared Engine**: Multiple repositories share connection pool (83% reduction)
- **Automatic Fallback**: If aggregates unavailable, falls back to candles or trades
- **Intelligent Source Selection**: Always uses fastest available data source

### Important Notes

- Always use `async with TimeseriesRepository(...) as repo:` context manager
- Use `arrow.Arrow` for timestamps (from arrow library)
- Repository automatically selects fastest available data source
- Table/view initialization handled by fullon_ohlcv_service
- Cache is transparent - no configuration needed

---

### Feed Numbering

**feed_num = Feed.order (from database)**

```python
# Feed ORM model
class Feed:
    feed_id: int      # Primary key (auto-increment)
    symbol: str       # e.g., "BTC/USD"
    period: str       # e.g., "1m", "5m", "tick"
    order: int        # Order of feed in strategy (1, 2, 3, ...)

# In BaseStrategy
for feed in self.feeds:
    feed_num = feed.order  # Simple alias
    self.dataframe[feed_num] = pd.DataFrame()
    self.position[feed_num] = None
    self.signal[feed_num] = None
```

## Development Workflow

### Before Writing Code
1. Review fullon_ohlcv patterns (especially tests/conftest.py)
2. Understand the Strategy ORM model (from fullon_orm)
3. Review Feed ORM model (symbol, period, order fields)
4. Understand TickCache and CandleRepository interfaces

### When Implementing Features
1. **Always write tests first** (TDD approach)
2. **Use real database** connections in tests (like fullon_ohlcv)
3. **Follow factory pattern** for test data
4. **Use fullon_log** for all logging (get_component_logger)
5. **Ensure async safety** (proper locking, no race conditions)

### Testing Commands
- Run tests: `pytest`
- Check coverage: `pytest --cov=fullon_strategies`
- Parallel tests: `pytest -n auto` (each worker gets own DB)
- Run specific test: `pytest tests/path/to/test.py::TestClass::test_method`

## Common Operations

### Required Imports

**CRITICAL: Always import required ORM models before using them in your strategies.**

#### Standard Imports for All Strategies

```python
from fullon_strategies import BaseStrategy
from fullon_orm import DatabaseContext
from fullon_orm.models import Order, Trade, Position
```

#### Additional Imports as Needed

```python
# Paper Trading / Backtesting
from fullon_orm.models import DryTrade

# Strategy Configuration (usually already available in BaseStrategy)
from fullon_orm.models import Bot, Strategy, Feed

# Advanced Features
from fullon_orm.models import Symbol, Exchange
```

#### Common Import Mistakes

❌ **Forgetting to import DryTrade when using dry_run mode:**
```python
# This will fail with NameError
if self.dry_run:
    dry_trade = DryTrade(...)  # DryTrade not imported!
```

✅ **Always import models you're using:**
```python
from fullon_orm.models import DryTrade

if self.dry_run:
    dry_trade = DryTrade(...)  # Works correctly
```

**Reference**: See `fullon_orm/docs/FULLON_ORM_LLM_METHOD_REFERENCE.md` for complete ORM usage patterns.

---

### BaseStrategy Usage
```python
from fullon_strategies import BaseStrategy
from fullon_orm.models import Strategy
import pandas as pd

class RSIStrategy(BaseStrategy):
    def __init__(self, strategy_orm: Strategy):
        super().__init__(strategy_orm)
        self.rsi_period = 14
        self.oversold = 30
        self.overbought = 70
        self._dataframe = {}  # Store indicators separately

    def prepare_indicators(self):
        """IN-LOOP Step 4: Calculate indicators when new bar completes."""
        for feed in self.feeds:
            feed_num = feed.order

            # Only calculate if new bar completed
            if not self.bar_completed[feed_num]:
                continue

            df = self.dataframe[feed_num].copy()
            df.ta.rsi(length=self.rsi_period, append=True)
            self._dataframe[feed_num] = df
            self.bar_completed[feed_num] = False

    def generate_signals(self):
        """IN-LOOP Step 5: Generate buy/sell signals."""
        for feed in self.feeds:
            feed_num = feed.order

            df = self._dataframe[feed_num]
            rsi = df[f'RSI_{self.rsi_period}'].iloc[-1]

            if rsi < self.oversold:
                self.signal[feed_num] = "buy"
            elif rsi > self.overbought:
                self.signal[feed_num] = "sell"
            else:
                self.signal[feed_num] = None

    async def on_signal(self, feed_num: int):
        """IN-LOOP Step 6a: Handle signal when no position."""
        if self.signal[feed_num] == "buy":
            size = self.calculate_position_size(feed_num)
            current_price = self.tick_buffer[feed_num][-1]

            # Set risk parameters
            self.stop_loss[feed_num] = current_price * 0.98
            self.take_profit[feed_num] = current_price * 1.06

            # Place order (BaseStrategy handles DB)
            await self.place_order(feed_num, "buy", size)

    async def on_position(self, feed_num: int):
        """IN-LOOP Step 6b: Manage existing position."""
        # Check if risk management triggered exit
        if self.exit_signal[feed_num]:
            await self.close_position(feed_num, reason=self.exit_reason[feed_num])
```

### FeedLoader Usage
```python
from fullon_strategies.utils import FeedLoader
from fullon_orm import DatabaseContext

async def load_strategy_feeds(str_id: int):
    async with DatabaseContext() as db:
        strategy = await db.strategies.get_by_id(str_id)

        # Create feed loader
        loader = FeedLoader(strategy)

        # Load all feeds
        await loader.load_feeds()

        # Access feeds
        for feed in strategy.feeds_list:
            data = loader.get_feed(feed.feed_id)
            if feed.period == "tick":
                print(f"Tick: {data.price}")
            else:
                print(f"OHLCV: {data.tail()}")
```

### StrategyLoader Usage
```python
from fullon_strategies import StrategyLoader
from fullon_orm import DatabaseContext

async def run_strategy(str_id: int):
    async with DatabaseContext() as db:
        strategy_orm = await db.strategies.get_by_id(str_id)

    # Load strategy class dynamically
    strategy_class = StrategyLoader.load(strategy_orm.cat_strategy.class_name)

    # Create instance
    strategy = strategy_class(strategy_orm)

    # Initialize (loads feeds, sets up variables)
    await strategy.init()

    # Execute (starts internal main_loop, runs until stopped)
    await strategy.run()
```

### DatabaseContext Pattern

**All database operations use `DatabaseContext` per operation:**

```python
# Example: BaseStrategy.place_order()
async def place_order(self, feed_num: int, side: str, size: float) -> int:
    """Place order with database persistence."""
    feed = self._get_feed_by_order(feed_num)

    # Create fresh context for this operation
    async with DatabaseContext() as db:
        # Create Order ORM object
        order = Order(
            bot_id=self.strategy_orm.bot_id,
            uid=self.strategy_orm.bot.user_id,
            ex_id=feed.ex_id,
            symbol=feed.symbol,
            side=side,
            volume=size,  # CRITICAL: Use 'volume' NOT 'amount'
            status="New"
        )

        # Save via repository
        saved_order = await db.orders.save_order(order)

        # Commit transaction
        await db.commit()

        return saved_order.order_id

# Example: BaseStrategy.on_trade()
async def on_trade(self, feed_num: int, trade: Trade) -> None:
    """Handle trade execution."""

    # Fresh context per operation
    async with DatabaseContext() as db:
        # Save trade (auto-calculates PnL via FIFO!)
        await db.trades.save_trades([trade])
        await db.commit()

    # Update in-memory position
    volume = trade.volume if trade.side == "buy" else -trade.volume
    self.position[feed_num].add_trade(volume, trade.price, trade.fee)
```

## Test Structure (FOLLOW fullon_ohlcv)

### Directory Structure
```
tests/
├── conftest.py          # Per-worker DB, factories, fixtures (FOLLOW fullon_ohlcv)
├── unit/                # Unit tests (FeedLoader, StrategyLoader logic)
├── integration/         # Integration tests (with real fullon_cache, fullon_ohlcv)
└── strategies/          # Example strategy tests
```

### Conftest Pattern (COPY from fullon_ohlcv)
- Per-worker test database creation
- Unique DB name per test file
- Cleanup after tests
- Factory fixtures

### Test Data Pattern
- Use TickCache to set test ticks
- Use CandleRepository to create test OHLCV data
- Test strategy loading, feed access, execution

## Known Issues & TODOs

1. **Strategy lifecycle management** - Start, stop, pause mechanisms
2. **Error handling** - Strategy errors shouldn't crash bot
3. **Performance monitoring** - Track strategy execution time
4. **Memory management** - OHLCV DataFrames can get large

### Testing Warnings - RESOLVED ✅

Previously encountered deprecation warnings have been fixed. See [docs/TESTING_KNOWN_ISSUES.md](./docs/TESTING_KNOWN_ISSUES.md) for details.

**Resolution:**
- **uvloop deprecation warning**: ✅ Fixed by setting up EventLoopPolicy in conftest.py before fullon_orm loads
- **pytest-asyncio event loop warning**: ✅ Fixed by filtering the cosmetic warning in pytest.ini

All tests run without warnings in both serial and parallel modes.

## Security Considerations

- Strategies run in bot process - validate strategy code before loading
- Logging should not expose sensitive data (API keys, passwords)
- Proper exception handling to prevent bot crashes

## Questions to Ask When Stuck

1. Is this operation truly async?
2. Am I using the Feed ORM model correctly?
3. Are my tests using real database connections (like fullon_ohlcv)?
4. Am I using fullon_log for logging?
5. Have I added proper error handling?
6. Does my strategy inherit from BaseStrategy?

## Dependencies Summary

**MUST USE:**
- `fullon_log` - All logging via get_component_logger("fullon.strategies.xxx")
- `fullon_orm` - Strategy, Feed, Symbol, Tick models
- `fullon_cache` - TickCache for real-time tick data
- `fullon_ohlcv` - CandleRepository for OHLCV data

**EXTERNAL:**
- `pandas` - DataFrame operations (OHLCV data)
- `asyncio` - Async operations

Remember: This library is the core of the trading logic - strategies must be fast, reliable, and well-tested!
