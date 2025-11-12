# Strategy Lifecycle Documentation

**This document describes BaseStrategy's INTERNAL lifecycle.**

BaseStrategy has an autonomous `main_loop()` that orchestrates all lifecycle phases. Child strategies simply implement hooks (prepare_indicators, generate_signals, on_signal, on_position) which are called automatically by BaseStrategy at the appropriate times.

**📖 Also read:** [ARCHITECTURE.md](./ARCHITECTURE.md) for core architectural decisions and rationale.

## Overview

Strategies follow a three-phase lifecycle managed by BaseStrategy:

1. **PRE-LOOP** - Initialization and validation (runs once at startup)
2. **IN-LOOP** - Continuous execution and monitoring (loops until stopped)
3. **POST-LOOP** - Graceful shutdown and cleanup (runs once at shutdown)

**Execution:** `await strategy.run()` starts the internal loop and blocks until shutdown.

## Architecture

```
BaseStrategy (parent)
    ↓ inherits
ChildStrategy (custom implementation)
```

- **BaseStrategy**: Handles infrastructure, risk management, lifecycle orchestration, main_loop()
- **ChildStrategy**: Implements custom trading logic, signal generation, indicator calculations

### Feed Numbering

**Throughout this document, `feed_num` refers to `Feed.order` from the database.**

```python
# Feed ORM model
class Feed:
    feed_id: int      # Primary key
    symbol: str       # e.g., "BTC/USD"
    period: str       # e.g., "1m", "5m", "tick"
    order: int        # Order of feed in strategy (1, 2, 3, ...)

# In BaseStrategy, feed_num = feed.order
for feed in self.feeds:
    feed_num = feed.order  # Simple alias
    self.dataframe[feed_num] = pd.DataFrame()
    self.position[feed_num] = None
```

See [FEED_MAPPING.md](./FEED_MAPPING.md) for detailed feed numbering documentation.

---

## PRE-LOOP Phase

Initialization phase that runs once before the main loop starts.

### Step 1: BaseStrategy.__init__()

**Responsibilities:**
- Check if required services are running:
  - `fullon_ohlcv_service` - Historical candle data
  - `fullon_ticker_service` - Real-time tick data
  - `fullon_account_service` - Account/position data
- Ensure candles are filled to current time (no gaps, last bar is current)
- Load strategy parameters from bot (receives `fullon_orm.Bot` object)
- Deduce configuration from bot → strategy → feeds relationship

**Initialize Strategy Variables:**

For each feed, set up:

```python
# Tick data (for all feeds)
self.tick_buffer[feed_num] = []  # Last 100 ticks (max)
self.timestamp[feed_num] = []    # Last timestamps (Arrow format)

# Risk management (for non-tick feeds)
self.take_profit[feed_num] = Decimal     # Take profit price
self.trailing_stop[feed_num] = Decimal   # Trailing stop distance
self.stop_loss[feed_num] = Decimal       # Stop loss price
self.maxtime_open[feed_num] = int        # Max seconds position can be open

# Position tracking (for non-tick feeds)
self.position[feed_num] = fullon_orm.Position  # Validated vs exchange
self.open_trades[feed_num] = List[Trade]       # List of buys for this position
self.pnl[feed_num] = Decimal                   # Validated vs exchange
self.leverage[feed_num] = Decimal              # Leverage for this feed

# Strategy parameters
self.params = StrategyParams()  # e.g., self.params.RSI, self.params.SMA, self.params.EMA

# Market data (for non-tick feeds)
self.dataframe[feed_num] = pd.DataFrame  # OHLCV data
self.funds[feed_num] = Decimal           # Available funds for this feed

# Order tracking (for non-tick feeds)
self.last_orders[feed_num] = []  # Last 220 orders (max)

# State flags (for non-tick feeds)
self.bar_completed[feed_num] = bool    # True when new candle formed
self.exit_signal[feed_num] = bool      # True to close position now
self.signal[feed_num] = str            # "buy", "sell", or None

# Configuration
self.dry_run = bool  # True for paper trading
```

**Loading Historical OHLCV Data:**

During initialization, BaseStrategy loads historical data for all OHLCV feeds using FeedLoader:

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

async def init(self):
    """Initialize strategy and load feeds."""

    for feed in self.feeds:
        feed_num = feed.order

        if feed.period == "tick":
            # Initialize tick buffer
            self.tick_buffer[feed_num] = []
        else:
            # Load historical OHLCV data using TimeseriesRepository
            # NOTE: Table/view initialization handled by fullon_ohlcv_service
            async with TimeseriesRepository(
                exchange=feed.exchange.name,
                symbol=feed.symbol.symbol,
                test=False
            ) as repo:
                # Load last 500 bars (adjust based on strategy needs)
                end_time = arrow.utcnow()
                start_time = end_time.shift(hours=-500)

                df = await repo.fetch_ohlcv_df(
                    compression=feed.compression,
                    period=feed.period,
                    fromdate=start_time,
                    todate=end_time
                )

                # Store in dataframe dict using feed_num as key
                self.dataframe[feed_num] = df

                # Initialize position tracking
                self.position[feed_num] = None
                self.signal[feed_num] = None
                self.bar_completed[feed_num] = False

                self.logger.info(
                    "Feed loaded",
                    feed_num=feed_num,
                    symbol=feed.symbol.symbol,
                    rows=len(df),
                    primary_source=repo.primary_source
                )
```

**Key Points:**
- Uses TimeseriesRepository for intelligent data source selection (aggregates > candles > trades)
- Context manager ensures proper cleanup
- Arrow.Arrow used for timestamps
- Table/view initialization handled by fullon_ohlcv_service automatically
- Loads sufficient history for indicator calculations (500 bars default)
- Repository automatically uses fastest available data source

### Step 2: BaseStrategy.validate_parameters()

**Responsibilities:**
- Ensure all parameters are within valid ranges
- Check for conflicting parameters (e.g., stop_loss > take_profit)
- Validate feed configurations
- Verify required params exist

### Step 3: BaseStrategy.initialize_telemetry()

**Responsibilities:**
- Setup performance monitoring
- Initialize logging with correlation IDs
- Setup metrics collection
- Configure alerting thresholds

### Step 4: ChildStrategy.__init__() [or custom name]

**Responsibilities:**
- Create strategy-specific class variables
- Initialize any external data sources (Twitter API, news feeds, etc.)
- Run one-time setup code
- **This step is OPTIONAL** - can be empty

**Example:**
```python
class MyStrategy(BaseStrategy):
    def __init__(self, bot):
        super().__init__(bot)
        self.twitter_api = TwitterClient()
        self.sentiment_threshold = 0.7
        self.custom_indicator_period = 20
```

### Step 5: BaseStrategy.run() - Start Execution

**This is the entry point that starts the internal lifecycle loop.**

**Responsibilities:**
- Start internal `main_loop()` that orchestrates all lifecycle phases
- Loop continuously until `self.shutting_down = True`
- Coordinate all IN-LOOP steps automatically
- Call child strategy hooks at appropriate times

**Pattern:**
```python
# fullon_bot or launcher
strategy = MyStrategy(strategy_orm)
await strategy.init()  # PRE-LOOP phase
await strategy.run()    # Starts main_loop, blocks until shutdown
```

**BaseStrategy.run() implementation:**
```python
async def run(self):
    """Main execution - runs internal loop until stopped."""
    self.logger.info("Strategy starting", str_id=self.str_id)

    while not self.shutting_down:
        try:
            await self.main_loop()  # Single iteration
        except Exception as e:
            self.logger.error("Loop error", error=str(e))
            if self.should_halt(e):
                break
        await asyncio.sleep(self.loop_interval)  # Throttle

    await self.shutdown()  # POST-LOOP phase
```

---

## IN-LOOP Phase

**Continuous execution phase - runs in `main_loop()` until strategy is stopped.**

Each iteration of `main_loop()` executes all steps below, then sleeps briefly before next iteration.

### Step 0: BaseStrategy.validate_sync()

**Responsibilities:**
- Validate bars and ticks are time-synchronized
- Detect gaps in data
- Ensure no stale data
- Verify timestamp consistency across feeds

### Step 0.5: BaseStrategy.check_circuit_breakers()

**Responsibilities:**
- Check max drawdown limit
- Check daily loss limit
- Check position concentration limits
- Check API rate limiting
- **HALT strategy if breakers triggered**

### Step 1: BaseStrategy.update_dataframe()

**Responsibilities:**
- Check if new candles have formed
- Add new candles to `self.dataframe[feed_num]`
- Maintain rolling window (e.g., last 500 bars)
- Set `self.bar_completed[feed_num] = True` if new bar formed

**Implementation:**

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

async def update_dataframe(self):
    """Update OHLCV dataframes with new candles."""

    for feed in self.feeds:
        feed_num = feed.order

        # Skip tick feeds
        if feed.period == "tick":
            continue

        # Get current dataframe
        df = self.dataframe[feed_num]

        # Determine time range for new data
        if df.empty:
            # No data yet - load initial history
            end_time = arrow.utcnow()
            start_time = end_time.shift(hours=-500)
        else:
            # Get last timestamp and fetch from there
            last_timestamp = df.index[-1]  # Assuming timestamp is index
            start_time = arrow.get(last_timestamp)
            end_time = arrow.utcnow()

        # Fetch new candles using TimeseriesRepository
        # NOTE: Table/view initialization handled by fullon_ohlcv_service
        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            new_candles_df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=start_time,
                todate=end_time
            )

        # Check if new data exists
        if not new_candles_df.empty:
            # Append new candles
            self.dataframe[feed_num] = pd.concat([
                self.dataframe[feed_num],
                new_candles_df
            ])

            # Maintain rolling window (keep last 500 bars)
            self.dataframe[feed_num] = self.dataframe[feed_num].tail(500)

            # Mark that new bar completed
            self.bar_completed[feed_num] = True
        else:
            # No new data
            self.bar_completed[feed_num] = False
```

**Notes:**
- TimeseriesRepository automatically uses fastest available data source
- Redis caching makes repeated queries 50-90% faster
- Context manager ensures proper cleanup
- Rolling window prevents unlimited memory growth
- Table/view initialization handled automatically by fullon_ohlcv_service

### Step 2: BaseStrategy.update_variables()

**Responsibilities:**
Update all strategy variables that need continuous updating using fullon_cache:

```python
from fullon_cache import TickCache, AccountCache

async def update_variables(self):
    """Update all per-feed variables from cache.

    Cache is populated by external services (fullon_ticker_service,
    fullon_account_service) that read from exchanges. NOT synced from database.
    """

    for feed in self.feeds:
        feed_num = feed.order

        # Update tick data from TickCache
        try:
            async with TickCache() as tick_cache:
                # get_ticker() uses positional args: symbol, exchange
                tick = await tick_cache.get_ticker(feed.symbol, feed.exchange.name)

                if tick:
                    # Update tick buffer
                    self.tick_buffer[feed_num].append(tick.price)
                    self.tick_buffer[feed_num] = self.tick_buffer[feed_num][-100:]  # Keep last 100

                    # Update timestamps
                    self.timestamp[feed_num].append(tick.time)
        except Exception as e:
            self.logger.error("Failed to get tick from cache",
                            feed_num=feed_num, error=str(e))

        # Re-validate position from AccountCache (cache is synced from exchange)
        try:
            async with AccountCache() as account_cache:
                # CRITICAL: get_position() takes POSITIONAL args (symbol, ex_id)
                # Always returns Position object (never None)
                cached_position = await account_cache.get_position(
                    feed.symbol,      # Positional arg 1
                    str(feed.ex_id)   # Positional arg 2
                )

                # Check volume to determine if position actually exists
                # (get_position always returns Position, even if empty with volume=0)
                if cached_position.volume != 0:
                    # Sync in-memory position with cached position
                    if self.position[feed_num] is None:
                        self.position[feed_num] = cached_position
                    else:
                        # Cross-check volume matches
                        if abs(self.position[feed_num].volume - cached_position.volume) > 0.0001:
                            self.logger.warning(
                                "Position mismatch with cache",
                                feed_num=feed_num,
                                memory_volume=self.position[feed_num].volume,
                                cached_volume=cached_position.volume
                            )

                    # Calculate current PnL using cached position
                    if tick and self.position[feed_num]:
                        pnl_data = self.position[feed_num].calculate_pnl(tick.price)
                        self.pnl[feed_num] = pnl_data['unrealized_pnl']
        except Exception as e:
            self.logger.error("Failed to validate position from cache",
                            feed_num=feed_num, error=str(e))
```

### Step 3: BaseStrategy.risk_management(feed_num)

**Responsibilities:**
- **Only runs if we have an open position**
- Check if stop_loss triggered → set `self.exit_signal[feed_num] = True`
- Check if take_profit triggered → set `self.exit_signal[feed_num] = True`
- Check if trailing_stop triggered → set `self.exit_signal[feed_num] = True`
- Check if maxtime_open exceeded → set `self.exit_signal[feed_num] = True`

**Example Logic:**
```python
if self.position[feed_num] is not None:
    current_price = self.tick_buffer[feed_num][-1]

    # Stop loss check
    if current_price <= self.stop_loss[feed_num]:
        self.exit_signal[feed_num] = True
        self.exit_reason[feed_num] = "stop_loss"

    # Take profit check
    if current_price >= self.take_profit[feed_num]:
        self.exit_signal[feed_num] = True
        self.exit_reason[feed_num] = "take_profit"

    # Trailing stop check
    if self.check_trailing_stop(feed_num, current_price):
        self.exit_signal[feed_num] = True
        self.exit_reason[feed_num] = "trailing_stop"

    # Max time check
    time_open = current_time - self.position[feed_num].open_time
    if time_open > self.maxtime_open[feed_num]:
        self.exit_signal[feed_num] = True
        self.exit_reason[feed_num] = "max_time"
```

### Step 3.5: BaseStrategy.update_portfolio_metrics()

**Responsibilities:**
- Calculate Sharpe ratio
- Update maximum drawdown
- Track win/loss ratio
- Update exposure metrics
- Calculate risk-adjusted returns

### Step 4: ChildStrategy.prepare_indicators()

**Responsibilities:**
- **Child strategy implements this**
- **ONLY runs when `self.bar_completed[feed_num] = True`** (new bar formed)
- Access immutable `self.dataframe[feed_num]`
- Create custom dataframe with technical indicators
- Store in `self._dataframe[feed_num]` or similar
- **Skip this step entirely if no new bar** - critical for performance!

**Example:**
```python
def prepare_indicators(self):
    for feed_num in self.non_tick_feeds:
        # CRITICAL: Only calculate if new bar completed
        if not self.bar_completed[feed_num]:
            continue  # Skip - no new data, use cached indicators

        # Don't modify base dataframe
        df = self.dataframe[feed_num].copy()

        # Add custom indicators using pandas-ta or custom calculations
        df.ta.rsi(length=self.params.RSI_PERIOD, append=True)
        df.ta.ema(length=self.params.EMA_FAST, append=True)
        df.ta.ema(length=self.params.EMA_SLOW, append=True)
        df.ta.macd(append=True)

        # Store in custom dataframe
        self._dataframe[feed_num] = df

        # Reset flag after processing
        self.bar_completed[feed_num] = False
```

### Step 5: ChildStrategy.generate_signals()

**Responsibilities:**
- **Child strategy implements this**
- Analyze indicators, price action, external data
- Optionally call LLM for signal generation
- Set `self.signal[feed_num]` to "buy", "sell", or None

**Example:**
```python
def generate_signals(self):
    for feed_num in self.non_tick_feeds:
        df = self._dataframe[feed_num]

        # RSI oversold/overbought
        rsi = df['RSI_14'].iloc[-1]

        # EMA crossover
        ema_fast = df[f'EMA_{self.params.EMA_FAST}'].iloc[-1]
        ema_slow = df[f'EMA_{self.params.EMA_SLOW}'].iloc[-1]

        # Generate signal
        if rsi < 30 and ema_fast > ema_slow:
            self.signal[feed_num] = "buy"
        elif rsi > 70 and ema_fast < ema_slow:
            self.signal[feed_num] = "sell"
        else:
            self.signal[feed_num] = None
```

### Step 5.5: ChildStrategy.on_tick() [OPTIONAL]

**Responsibilities:**
- **Child strategy MAY implement this**
- Called on every tick (if strategy needs tick-level granularity)
- Can be skipped if strategy only needs bar-level data

**Example:**
```python
async def on_tick(self, feed_num, tick):
    """Optional: Handle individual ticks"""
    # Example: Check for rapid price movement
    if len(self.tick_buffer[feed_num]) >= 10:
        recent_ticks = self.tick_buffer[feed_num][-10:]
        price_change = (recent_ticks[-1] - recent_ticks[0]) / recent_ticks[0]

        if abs(price_change) > 0.02:  # 2% rapid movement
            self.logger.warning(f"Rapid price movement detected: {price_change*100:.2f}%")
```

### Step 6a: ChildStrategy.on_signal()

**Called automatically by BaseStrategy.main_loop() when:**
- `self.position[feed_num] is None` (no open position)
- AND `self.signal[feed_num]` is set ("buy" or "sell")

**Responsibilities:**
- **Child strategy implements this**
- Evaluate signal and decide whether to enter position
- Set position size, leverage, risk parameters
- Submit entry order via `await self.place_order()`

**BaseStrategy routing logic:**
```python
# In BaseStrategy.main_loop()
for feed_num in self.non_tick_feeds:
    if self.position[feed_num] is None and self.signal[feed_num]:
        # No position but signal exists → enter
        await self.on_signal(feed_num)  # Calls child's implementation
```

**Example:**
```python
async def on_signal(self, feed_num):
    """Enter position based on signal"""
    if self.signal[feed_num] == "buy":
        # Calculate position size based on risk
        position_size = self.calculate_position_size(feed_num)

        # Set risk parameters
        entry_price = self.tick_buffer[feed_num][-1]
        self.stop_loss[feed_num] = entry_price * 0.98  # 2% stop loss
        self.take_profit[feed_num] = entry_price * 1.06  # 6% take profit

        # Submit order
        await self.place_order(feed_num, "buy", position_size)
```

### Step 6b: ChildStrategy.on_position()

**Called automatically by BaseStrategy.main_loop() when:**
- `self.position[feed_num] is not None` (open position exists)

**Responsibilities:**
- **Child strategy implements this**
- Monitor position health
- Adjust risk parameters (trailing stop)
- Decide if manual exit needed (beyond base risk management)
- Handle exit_signal if set by risk_management()

**BaseStrategy routing logic:**
```python
# In BaseStrategy.main_loop()
for feed_num in self.non_tick_feeds:
    if self.position[feed_num] is not None:
        # Have position → manage it
        await self.on_position(feed_num)  # Calls child's implementation
```

**Example:**
```python
async def on_position(self, feed_num):
    """Manage existing position"""
    # Check if base risk management triggered exit
    if self.exit_signal[feed_num]:
        await self.close_position(feed_num, reason=self.exit_reason[feed_num])
        return

    # Custom position management logic
    position = self.position[feed_num]
    current_price = self.tick_buffer[feed_num][-1]

    # Update trailing stop if in profit
    if current_price > position.entry_price * 1.03:  # 3% in profit
        new_stop = current_price * 0.99  # Trail at 1%
        if new_stop > self.stop_loss[feed_num]:
            self.stop_loss[feed_num] = new_stop
            self.logger.info(f"Updated trailing stop to {new_stop}")
```

### Step 7: BaseStrategy.update_bot_status()

**Responsibilities:**
- Update bot status in BotCache with per-feed information
- Provide heartbeat signal to monitoring systems
- Track running status, positions, PnL per feed
- Enable real-time bot monitoring through cache

**Implementation Pattern:**

```python
from fullon_cache import BotCache

async def update_bot_status(self):
    """Update bot status in cache with per-feed information.

    BotCache stores bot status in Redis for real-time monitoring.
    BotCache.update_bot() expects feed-based structure matching our architecture.
    """
    status_data = {}

    for feed in self.feeds:
        feed_num = feed.order

        # Build status for this feed
        # CRITICAL: Use "symbols" (plural, list) per BotCache API
        status_data[f"feed_{feed_num}"] = {
            "status": "running" if not self.shutting_down else "stopping",
            "symbols": [feed.symbol],  # List of symbols (plural)
            "period": feed.period,
            "exchange": feed.exchange.name,
            "has_position": (
                self.position[feed_num] is not None and
                self.position[feed_num].volume != 0
            ),
            "signal": self.signal.get(feed_num),
            "exit_signal": self.exit_signal.get(feed_num, False),
            "pnl": float(self.pnl.get(feed_num, 0)),
            "last_update": datetime.utcnow().isoformat()
        }

        # Add position details if exists
        if self.position[feed_num] and self.position[feed_num].volume != 0:
            pos = self.position[feed_num]
            status_data[f"feed_{feed_num}"]["position"] = {
                "volume": float(pos.volume),
                "entry_price": float(pos.entry_price),
                "current_price": float(self.tick_buffer[feed_num][-1]) if self.tick_buffer[feed_num] else 0.0,
                "unrealized_pnl": float(pos.unrealized_pnl)
            }

    # Write to BotCache
    try:
        async with BotCache() as cache:
            await cache.update_bot(str(self.strategy_orm.bot_id), status_data)
    except Exception as e:
        self.logger.error("Failed to update bot status in cache", error=str(e))
```

**Feed-Based Structure:**

The status_data dictionary uses `feed_{order}` keys matching our `feed_num = Feed.order` architecture:

```python
# Example status_data structure
{
    "feed_1": {
        "status": "running",
        "symbols": ["BTC/USDT"],  # List of symbols (plural)
        "period": "1m",
        "exchange": "kraken",
        "has_position": True,
        "signal": None,
        "exit_signal": False,
        "pnl": 125.50,
        "position": {
            "volume": 0.5,
            "entry_price": 42000.0,
            "current_price": 42500.0,
            "unrealized_pnl": 250.0
        }
    },
    "feed_2": {
        "status": "running",
        "symbols": ["ETH/USDT"],  # List of symbols (plural)
        "period": "5m",
        "exchange": "kraken",
        "has_position": False,
        "signal": "buy",
        "exit_signal": False,
        "pnl": -50.25
    }
}
```

**Cache Integration Notes:**

- BotCache stores bot status in Redis for real-time monitoring
- Monitoring systems read from cache for instant access to bot state
- Cache and database are INDEPENDENT - cache NOT synced from database
- Feed-based structure enables per-symbol monitoring
- Heartbeat implicit in update timestamp via `last_update` field
- Use "symbols" (plural, list) to match BotCache API expectations

### Step 8: BaseStrategy.emit_telemetry()

**Responsibilities:**
- Send performance metrics to monitoring system
- Log execution times
- Update dashboards
- Emit alerts if needed

---

## Required Model Imports

**All trading methods use fullon_orm models. Import these at the top of your strategy file.**

### Core Imports (Required for All Strategies)

Every strategy that uses trading methods needs these imports:

```python
from fullon_orm import DatabaseContext
from fullon_orm.models import Order, Trade, Position
```

These models are used by:
- **Order**: Created by `place_order()`, managed by `on_order()`
- **Trade**: Created by exchange fills, handled by `on_trade()`
- **Position**: In-memory position tracking, used throughout lifecycle

### Extended Imports (For Specific Features)

```python
# Paper Trading / Backtesting
from fullon_orm.models import DryTrade

# Strategy Configuration (already imported in BaseStrategy)
from fullon_orm.models import Bot, Strategy, Feed

# Advanced Features
from fullon_orm.models import Symbol, Exchange
```

### Import Notes

- **BaseStrategy** already imports all models internally - child strategies only need imports if they directly instantiate models
- **DryTrade** is required if strategy runs in `dry_run=True` mode
- Most child strategies only need `Order`, `Trade`, `Position` if overriding trading methods

### Example Import Block

```python
# my_strategy.py
from fullon_strategies import BaseStrategy
from fullon_orm import DatabaseContext
from fullon_orm.models import Order, Trade, Position, DryTrade

class MyStrategy(BaseStrategy):
    # Your strategy implementation
    pass
```

---

## BaseStrategy Trading Methods

These utility methods are provided by BaseStrategy and used by child strategies in `on_signal()` and `on_position()` for position and order management. All database operations use `fullon_orm` repositories.

### BaseStrategy.open_position()

```python
async def open_position(self, feed_num: int, side: str, size: float, price: float) -> Position:
    """
    Open a new position for a feed.

    Args:
        feed_num: Feed number to open position for
        side: "buy" (long) or "sell" (short)
        size: Position size (volume)
        price: Entry price

    Returns:
        Position object (in-memory tracking)

    Responsibilities:
        - Create Order object and save via OrderRepository
        - Initialize in-memory Position object
        - Update self.position[feed_num]
        - Log position opening
        - Handle database commit via DatabaseContext

    Example:
        position = await self.open_position(
            feed_num=1,
            side="buy",
            size=0.5,
            price=67000.0
        )
    """
```

**Implementation Notes:**
- Uses `fullon_orm.DatabaseContext` for database access
- Creates `Order` with proper fields: `bot_id`, `uid`, `ex_id`, `cat_ex_id`, `symbol`, `order_type`, `side`, `volume`, `price`, `status`, `reason`
- **CRITICAL**: Use `volume` field in Order, NOT `amount`
- Saves order via `db.orders.save_order(order)`
- Creates `Position(symbol, ex_id, side)` for in-memory tracking
- Position is NOT persisted to database (it's a non-persistent model)

### BaseStrategy.close_position()

```python
async def close_position(self, feed_num: int, reason: str) -> bool:
    """
    Close an existing position for a feed.

    Args:
        feed_num: Feed number to close position for
        reason: Reason for closing (e.g., "stop_loss", "take_profit", "manual")

    Returns:
        True if successful, False otherwise

    Responsibilities:
        - Calculate closing order size from self.position[feed_num].volume
        - Determine opposite side ("sell" if long, "buy" if short)
        - Create closing Order and save via OrderRepository
        - Clear position tracking variables:
            - self.position[feed_num] = None
            - self.exit_signal[feed_num] = False
            - self.exit_reason[feed_num] = None
            - self.stop_loss[feed_num] = None
            - self.take_profit[feed_num] = None
        - Log position closing with reason
        - Handle database commit

    Example:
        success = await self.close_position(
            feed_num=1,
            reason="take_profit"
        )
    """
```

**Implementation Notes:**
- Checks if position exists before closing
- Calculates closing size: `abs(self.position[feed_num].volume)`
- Determines opposite side: `"sell" if self.position[feed_num].is_long else "buy"`
- Creates Order with `reason` field set to provided reason
- Clears all position-related variables for the feed
- Position PnL is calculated automatically by TradeRepository when trades are saved

### BaseStrategy.on_order()

```python
async def on_order(self, feed_num: int, order: Order) -> None:
    """
    Callback when order status changes (e.g., filled, cancelled, rejected).

    Args:
        feed_num: Feed number for this order
        order: Order object with updated status

    Responsibilities:
        - Update order status in database via OrderRepository
        - Update self.last_orders[feed_num] circular buffer (max 220)
        - Log order status change with details
        - Handle order rejection/cancellation appropriately
        - Update order final_volume if partially filled

    Example:
        # Called by exchange connector when order fills
        await strategy.on_order(feed_num=1, order=filled_order)
    """
```

**Implementation Notes:**
- Uses `db.orders.update_order_status(order_id, status, ex_order_id)`
- Can also use `db.orders.update_order_final_volume(order_id, final_volume)` for partial fills
- Maintains circular buffer: `self.last_orders[feed_num].append(order); self.last_orders[feed_num] = self.last_orders[feed_num][-220:]`
- Logs order transitions: pending → filled, pending → cancelled, etc.
- Does NOT create/update Position directly (that happens in on_trade)

### BaseStrategy.on_trade()

```python
async def on_trade(self, feed_num: int, trade: Trade) -> None:
    """
    Callback when trade executes (order fill confirmation from exchange).

    Args:
        feed_num: Feed number for this trade
        trade: Trade object with execution details

    Responsibilities:
        - Save trade to database via TradeRepository (auto-calculates PnL!)
        - Update in-memory Position with position.add_trade()
        - Calculate current PnL using position.calculate_pnl()
        - Update self.pnl[feed_num] with unrealized PnL
        - Update self.open_trades[feed_num] for position tracking
        - Log trade execution with P&L details
        - Handle database commit

    Example:
        # Called by exchange connector after trade executes
        trade = Trade(
            ex_trade_id="T123",
            ex_order_id="O456",
            symbol="BTC/USD",
            side="buy",
            volume=0.5,
            price=67000.0,
            cost=33500.0,
            fee=33.5
        )
        await strategy.on_trade(feed_num=1, trade=trade)
    """
```

**Implementation Notes:**
- **CRITICAL**: `db.trades.save_trades([trade])` automatically calculates realized PnL using FIFO matching!
- The TradeRepository handles complex PnL calculation including leverage
- For position tracking: convert trade to signed volume: `volume = trade.volume if trade.side == "buy" else -trade.volume`
- Update Position: `self.position[feed_num].add_trade(volume=volume, price=trade.price, fee=trade.fee)`
- Calculate current PnL: `pnl_data = self.position[feed_num].calculate_pnl(current_price)`
- Update: `self.pnl[feed_num] = pnl_data['unrealized_pnl']`
- Position may be closed if volume reaches zero

### BaseStrategy.place_order()

```python
async def place_order(self, feed_num: int, side: str, size: float,
                     order_type: str = "market", price: Optional[float] = None) -> int:
    """
    Place an order for a feed.

    Args:
        feed_num: Feed number to place order for
        side: "buy" or "sell"
        size: Order size (volume)
        order_type: "market", "limit", "stop", etc. (default: "market")
        price: Limit price (required for limit orders, None for market)

    Returns:
        order_id: Database ID of created order

    Responsibilities:
        - Create Order object with proper fields
        - Save order via OrderRepository
        - Log order placement
        - Return order_id for tracking
        - Handle database commit

    Example:
        order_id = await self.place_order(
            feed_num=1,
            side="buy",
            size=0.5,
            order_type="limit",
            price=67000.0
        )
    """
```

**Implementation Notes:**
- Creates Order with all required fields from strategy/bot context
- Gets `bot_id`, `uid`, `ex_id`, `cat_ex_id`, `exchange`, `symbol` from strategy configuration
- **CRITICAL**: Uses `volume` field, NOT `amount`
- Sets `status="New"` for new orders
- For market orders: `price=None`
- For limit orders: `price` must be specified
- Returns `order.order_id` after saving for tracking

---

## POST-LOOP Phase

Graceful shutdown phase that runs when strategy is stopped.

### Step 1: BaseStrategy.signal_shutdown()

**Responsibilities:**
- Set `self.shutting_down = True`
- Broadcast shutdown signal to all components
- Prevent new operations from starting

### Step 2: BaseStrategy.cancel_pending_orders()

**Responsibilities:**
- Identify all pending orders across all feeds
- Cancel each pending order
- Wait for confirmations
- Log cancellation results

### Step 3: BaseStrategy.close_positions() [OPTIONAL]

**Responsibilities:**
- **Only if `config.close_on_shutdown = True`**
- Close all open positions
- Submit market orders for immediate execution
- Wait for fills
- Log final position states

### Step 4: BaseStrategy.persist_final_state()

**Responsibilities:**
- Save strategy state to database
- Persist any cached data
- Write checkpoint for potential restart
- Save final parameter values

### Step 5: BaseStrategy.generate_final_report()

**Responsibilities:**
- Calculate total PnL per feed and aggregate
- Count trades executed (wins/losses)
- Calculate win/loss ratio
- Determine maximum drawdown experienced
- List any errors/warnings encountered
- Report final positions (if not closed)
- Generate performance metrics:
  - Total execution time
  - API calls made
  - Average execution latency
  - Sharpe ratio
  - Total return percentage

**Example Report:**
```json
{
  "session_id": "abc123",
  "start_time": "2025-01-10T10:00:00Z",
  "end_time": "2025-01-10T18:30:00Z",
  "duration_hours": 8.5,
  "total_pnl": 1234.56,
  "pnl_by_feed": {
    "feed_1": 856.23,
    "feed_2": 378.33
  },
  "trades_executed": 45,
  "wins": 28,
  "losses": 17,
  "win_rate": 0.622,
  "max_drawdown": -234.12,
  "sharpe_ratio": 1.85,
  "final_positions": [],
  "errors": 2,
  "warnings": 8
}
```

### Step 6: BaseStrategy.cleanup_resources()

**Responsibilities:**
- Close database connections
- Disconnect from exchanges
- Flush Redis caches
- Close WebSocket connections
- Release file handles
- Clean up temporary files

### Step 7: BaseStrategy.archive_session_data()

**Responsibilities:**
- Archive logs to long-term storage
- Save strategy state snapshot
- Backup important metrics
- Compress and store for analysis

### Step 8: BaseStrategy.send_shutdown_notification()

**Responsibilities:**
- Send shutdown report via configured channels (email, Slack, etc.)
- Update monitoring dashboards
- Log final status
- Mark strategy as stopped in database

---

## Variable Reference

### Per-Feed Variables

| Variable | Type | Description | Tick Feed | OHLCV Feed |
|----------|------|-------------|-----------|------------|
| `tick_buffer[feed_num]` | List[Decimal] | Last 100 ticks | ✓ | ✓ |
| `timestamp[feed_num]` | List[Arrow] | Last timestamps | ✓ | ✓ |
| `take_profit[feed_num]` | Decimal | Take profit price | ✗ | ✓ |
| `trailing_stop[feed_num]` | Decimal | Trailing stop distance | ✗ | ✓ |
| `stop_loss[feed_num]` | Decimal | Stop loss price | ✗ | ✓ |
| `maxtime_open[feed_num]` | int | Max seconds open | ✗ | ✓ |
| `position[feed_num]` | Position | Current position | ✗ | ✓ |
| `open_trades[feed_num]` | List[Trade] | Trades in position | ✗ | ✓ |
| `pnl[feed_num]` | Decimal | Profit/Loss | ✗ | ✓ |
| `leverage[feed_num]` | Decimal | Position leverage | ✗ | ✓ |
| `dataframe[feed_num]` | DataFrame | OHLCV data | ✗ | ✓ |
| `funds[feed_num]` | Decimal | Available funds | ✗ | ✓ |
| `last_orders[feed_num]` | List[Order] | Last 220 orders | ✗ | ✓ |
| `bar_completed[feed_num]` | bool | New bar formed | ✗ | ✓ |
| `exit_signal[feed_num]` | bool | Close position now | ✗ | ✓ |
| `signal[feed_num]` | str | Buy/sell/None | ✗ | ✓ |

### Global Variables

| Variable | Type | Description |
|----------|------|-------------|
| `params` | StrategyParams | Strategy parameters (RSI, SMA, EMA, etc.) |
| `dry_run` | bool | Paper trading mode |
| `shutting_down` | bool | Shutdown in progress |

---

## Error Handling Strategy

### Service Failures
- Implement exponential backoff for reconnections
- Maximum retry attempts: 5
- Fallback to cached data when available
- Log all service failures

### Data Synchronization Issues
- Detect gaps in candle data → fill from backup source
- Detect stale ticks → request refresh
- Validate order book depth → adjust position size if insufficient

### Race Conditions
- Use per-feed locks for critical operations
- Implement asyncio.Lock for concurrent feed updates
- Queue conflicting operations

### Exchange API Failures
- Implement retry logic with exponential backoff
- Use fallback exchanges if available
- Implement order status polling for confirmation
- Log all API errors

---

## Performance Considerations

### Memory Optimization
- Use CircularBuffer for tick/order history instead of lists
- Implement lazy loading for dataframes
- Clear old data periodically

### CPU Optimization
- Batch indicator calculations
- Cache expensive computations (LRU cache)
- Throttle loop to prevent CPU spinning
- Use efficient data structures

### Network Optimization
- Batch API requests when possible
- Use WebSockets for real-time data
- Implement connection pooling
- Compress large data transfers

### Loop Timing
- Minimum loop time to prevent spinning
- Priority processing for critical updates (stop losses)
- Regular processing for indicators
- Batch non-urgent operations

---

## Testing Strategy

### Unit Tests
- Test each lifecycle method independently
- Mock external dependencies
- Test edge cases and error conditions

### Integration Tests
- Test full lifecycle with mock exchange
- Test service failure scenarios
- Test data sync issues
- Test concurrent operations

### Replay Testing
- Record live data
- Replay through strategy
- Verify expected behavior
- Compare against historical performance

---

## Configuration Example

```python
{
    "strategy_class": "RSIStrategy",
    "params": {
        "RSI_PERIOD": 14,
        "RSI_OVERSOLD": 30,
        "RSI_OVERBOUGHT": 70,
        "EMA_FAST": 10,
        "EMA_SLOW": 20
    },
    "risk": {
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.06,
        "trailing_stop_pct": 0.01,
        "max_time_open_seconds": 86400,
        "max_drawdown_pct": 0.15,
        "daily_loss_limit": 1000.00
    },
    "execution": {
        "dry_run": false,
        "close_on_shutdown": true,
        "loop_interval_seconds": 1.0,
        "max_position_size_pct": 0.25
    }
}
```

---

## Summary

The strategy lifecycle provides a robust framework for trading strategy execution with:

- ✅ **Clear separation of concerns** - Base handles infrastructure, child handles logic
- ✅ **Comprehensive risk management** - Built-in stop loss, take profit, trailing stops
- ✅ **Multi-feed support** - Handle multiple feeds per strategy
- ✅ **Mixed feed types** - Support both tick and OHLCV feeds
- ✅ **Graceful shutdown** - Proper cleanup and reporting
- ✅ **Error resilience** - Circuit breakers and fallbacks
- ✅ **Performance monitoring** - Telemetry and metrics throughout
- ✅ **State persistence** - Checkpoint and recovery capabilities

This lifecycle design ensures production-ready strategies that are robust, maintainable, and performant.
