# Feed Mapping and Numbering

This document explains how feeds are numbered and accessed in the fullon_strategies framework.

**Last Updated**: 2025-01-11

---

## Core Concept

**`feed_num` is simply an alias for `Feed.order` from the database.**

There is no complex indexing or mapping - it's a direct 1:1 relationship.

---

## Feed ORM Model

```python
# From fullon_orm.models
class Feed(BaseModel):
    __tablename__ = "feeds"

    feed_id = Column(Integer, primary_key=True, autoincrement=True)
    str_id = Column(Integer, ForeignKey("strategy.str_id"))
    symbol_id = Column(Integer, ForeignKey("symbols.symbol_id"))
    ex_id = Column(Integer, ForeignKey("exchanges.ex_id"))
    cat_ex_id = Column(Integer, ForeignKey("cat_exchanges.cat_ex_id"))
    period = Column(String(10))  # e.g., "1m", "5m", "1h", "tick"
    compression = Column(Integer, default=1)
    order = Column(Integer)  # THIS IS feed_num!

    # Relationships
    strategy = relationship("Strategy", back_populates="feeds_list")
    symbol = relationship("Symbol")
    exchange = relationship("Exchange")
```

**Key Field: `Feed.order`**
- User-defined ordering (1, 2, 3, ...)
- Determines which feed is "first", "second", etc.
- Used to key all per-feed dictionaries in BaseStrategy

---

## BaseStrategy Feed Initialization

```python
class BaseStrategy(ABC):
    def __init__(self, strategy_orm: Strategy):
        self.strategy_orm = strategy_orm
        self.feeds = strategy_orm.feeds_list  # List[Feed]

        # Initialize per-feed dictionaries using Feed.order as key
        self.dataframe: Dict[int, pd.DataFrame] = {}
        self.position: Dict[int, Optional[Position]] = {}
        self.tick_buffer: Dict[int, List[float]] = {}
        self.signal: Dict[int, Optional[str]] = {}
        self.bar_completed: Dict[int, bool] = {}
        self.stop_loss: Dict[int, Optional[Decimal]] = {}
        self.take_profit: Dict[int, Optional[Decimal]] = {}
        # ... many more per-feed variables

        for feed in self.feeds:
            feed_num = feed.order  # The key!

            # Initialize based on feed type
            if feed.period == "tick":
                self.tick_buffer[feed_num] = []
            else:
                self.dataframe[feed_num] = pd.DataFrame()
                self.position[feed_num] = None
                self.signal[feed_num] = None
                self.bar_completed[feed_num] = False
                # ... etc.
```

---

## Child Strategy Usage

### Pattern 1: Iterate All Feeds

```python
class MyStrategy(BaseStrategy):
    def prepare_indicators(self):
        """Calculate indicators for all OHLCV feeds."""
        for feed in self.feeds:
            feed_num = feed.order  # Get feed_num

            # Skip tick feeds
            if feed.period == "tick":
                continue

            # Only calculate if new bar completed
            if not self.bar_completed[feed_num]:
                continue

            # Access dataframe by feed_num
            df = self.dataframe[feed_num].copy()

            # Calculate indicators
            df.ta.rsi(length=14, append=True)

            # Store result
            self._dataframe[feed_num] = df
            self.bar_completed[feed_num] = False
```

### Pattern 2: Access Specific Feed by Order

```python
class MyStrategy(BaseStrategy):
    def __init__(self, strategy_orm: Strategy):
        super().__init__(strategy_orm)

        # Child strategy can reference feeds by their order
        # Assumption: User configured feeds in specific order
        # Feed order=1 is BTC, feed order=2 is ETH
        self.btc_feed_num = 1
        self.eth_feed_num = 2

    def generate_signals(self):
        """Generate signals for specific feeds."""
        # Access BTC dataframe
        btc_df = self._dataframe[self.btc_feed_num]
        btc_rsi = btc_df['RSI_14'].iloc[-1]

        # Access ETH dataframe
        eth_df = self._dataframe[self.eth_feed_num]
        eth_rsi = eth_df['RSI_14'].iloc[-1]

        # Cross-feed strategy logic
        if btc_rsi < 30 and eth_rsi < 30:
            # Both oversold
            self.signal[self.btc_feed_num] = "buy"
            self.signal[self.eth_feed_num] = "buy"
```

### Pattern 3: Get Feed by Symbol

```python
class MyStrategy(BaseStrategy):
    def get_feed_by_symbol(self, symbol: str) -> Optional[Feed]:
        """Helper to find feed by symbol."""
        for feed in self.feeds:
            if feed.symbol == symbol:
                return feed
        return None

    async def on_signal(self, feed_num: int):
        """Enter position - find feed to get symbol info."""
        # Get the feed object
        feed = next((f for f in self.feeds if f.order == feed_num), None)

        if feed:
            self.logger.info(
                "Entering position",
                symbol=feed.symbol,
                period=feed.period,
                feed_num=feed_num
            )

            await self.place_order(feed_num, "buy", 0.5)
```

---

## Feed Order Configuration

### Database Configuration

Feeds are configured in the database with explicit `order` values:

```sql
-- Example: Strategy with 2 feeds
INSERT INTO feeds (str_id, symbol_id, ex_id, period, order) VALUES
    (1, 100, 1, '1m', 1),   -- BTC/USD 1m is feed #1
    (1, 101, 1, '5m', 2);   -- ETH/USD 5m is feed #2
```

### User Configuration

When users create strategies, they specify feed order:

```python
# Via fullon_orm
from fullon_orm import DatabaseContext
from fullon_orm.models import Feed

async def create_strategy_with_feeds(str_id: int):
    async with DatabaseContext() as db:
        # Add BTC feed as order=1
        btc_feed = Feed(
            str_id=str_id,
            symbol_id=100,  # BTC/USD
            ex_id=1,
            period="1m",
            order=1  # First feed
        )
        await db.feeds.add(btc_feed)

        # Add ETH feed as order=2
        eth_feed = Feed(
            str_id=str_id,
            symbol_id=101,  # ETH/USD
            ex_id=1,
            period="5m",
            order=2  # Second feed
        )
        await db.feeds.add(eth_feed)

        await db.commit()
```

---

## Common Access Patterns

### 1. Process All Non-Tick Feeds

```python
# In BaseStrategy or child
for feed in self.feeds:
    feed_num = feed.order

    if feed.period == "tick":
        continue  # Skip tick feeds

    # Process OHLCV feed
    df = self.dataframe[feed_num]
    # ...
```

### 2. Get Current Price for Feed

```python
def get_current_price(self, feed_num: int) -> float:
    """Get latest price for a feed."""
    if self.tick_buffer[feed_num]:
        return self.tick_buffer[feed_num][-1]
    elif feed_num in self.dataframe and not self.dataframe[feed_num].empty:
        return self.dataframe[feed_num]['close'].iloc[-1]
    else:
        return 0.0
```

### 3. Check if Feed Has Position

```python
def has_position(self, feed_num: int) -> bool:
    """Check if feed has an open position."""
    return (
        feed_num in self.position and
        self.position[feed_num] is not None and
        self.position[feed_num].volume != 0
    )
```

### 4. Get Feed Metadata

```python
def get_feed_info(self, feed_num: int) -> Dict[str, Any]:
    """Get feed information."""
    feed = next((f for f in self.feeds if f.order == feed_num), None)

    if not feed:
        return {}

    return {
        "feed_id": feed.feed_id,
        "feed_num": feed.order,
        "symbol": feed.symbol,
        "period": feed.period,
        "exchange": feed.exchange.name,
        "ex_id": feed.ex_id
    }
```

---

## Feed Mapping in Cache Operations

Cache operations (via `fullon_cache`) use feed-based structures that directly align with our `feed_num = Feed.order` architecture.

### BotCache Feed Structure

When updating bot status in cache, use `feed_{order}` keys matching feed numbering:

```python
from fullon_cache import BotCache

async def update_bot_status(self):
    """Update bot status with per-feed data."""
    status_data = {}

    for feed in self.feeds:
        feed_num = feed.order  # Get feed_num

        # Build status using feed_num as key
        # CRITICAL: Use "symbols" (plural, list) per BotCache API
        status_data[f"feed_{feed_num}"] = {
            "status": "running",
            "symbols": [feed.symbol],  # List of symbols (plural)
            "period": feed.period,
            "has_position": self.position[feed_num] is not None,
            "signal": self.signal.get(feed_num),
            "pnl": float(self.pnl.get(feed_num, 0))
        }

    # Write to cache using feed-based structure
    try:
        async with BotCache() as cache:
            await cache.update_bot(str(self.strategy_orm.bot_id), status_data)
    except Exception as e:
        self.logger.error("Failed to update bot status", error=str(e))
```

**Result in cache:**
```json
{
  "bot_1": {
    "feed_1": {
      "status": "running",
      "symbols": ["BTC/USDT"],
      "period": "1m",
      "has_position": true,
      "signal": null,
      "pnl": 125.50
    },
    "feed_2": {
      "status": "running",
      "symbols": ["ETH/USDT"],
      "period": "5m",
      "has_position": false,
      "signal": "buy",
      "pnl": -50.25
    }
  }
}
```

### TickCache Feed Access

Get tickers for specific feeds using symbol and exchange from feed:

```python
from fullon_cache import TickCache

async def update_prices(self):
    """Update prices for all feeds from cache."""
    for feed in self.feeds:
        feed_num = feed.order

        try:
            # Get ticker from cache using feed metadata
            # get_ticker() uses POSITIONAL args: symbol, exchange
            async with TickCache() as cache:
                tick = await cache.get_ticker(
                    feed.symbol,          # Positional arg 1
                    feed.exchange.name    # Positional arg 2
                )

                if tick:
                    # Store using feed_num as key
                    self.tick_buffer[feed_num].append(tick.price)
                    self.timestamp[feed_num].append(tick.time)
        except Exception as e:
            self.logger.error("Failed to get tick from cache",
                            feed_num=feed_num, error=str(e))
```

### AccountCache Feed Access

Validate positions for specific feeds:

```python
from fullon_cache import AccountCache

async def validate_positions(self):
    """Validate positions from cache for all feeds."""
    for feed in self.feeds:
        feed_num = feed.order

        try:
            # Get position from cache using feed metadata
            # CRITICAL: get_position() uses POSITIONAL args (symbol, ex_id)
            # Always returns Position object (never None)
            async with AccountCache() as cache:
                cached_position = await cache.get_position(
                    feed.symbol,      # Positional arg 1
                    str(feed.ex_id)   # Positional arg 2
                )

                # Cross-check with in-memory position (keyed by feed_num)
                # Check volume to determine if position actually exists
                if self.position[feed_num] and cached_position.volume != 0:
                    volume_diff = abs(self.position[feed_num].volume - cached_position.volume)
                    if volume_diff > 0.0001:
                        self.logger.warning(
                            "Position mismatch",
                            feed_num=feed_num,
                            symbol=feed.symbol,
                            memory_volume=self.position[feed_num].volume,
                            cache_volume=cached_position.volume
                        )
        except Exception as e:
            self.logger.error("Failed to validate position from cache",
                            feed_num=feed_num, error=str(e))
```

### Cache Key Pattern

Cache operations follow this pattern:

1. **Iterate feeds**: `for feed in self.feeds:`
2. **Get feed_num**: `feed_num = feed.order`
3. **Use feed metadata**: `feed.symbol`, `feed.exchange.name`, `feed.ex_id`
4. **Key by feed_num**: Store results in dictionaries using `feed_num`

```python
# Pattern: Cache access with feed mapping
for feed in self.feeds:
    feed_num = feed.order  # 1. Get feed_num

    # 2. Access cache with feed metadata
    async with SomeCache() as cache:
        data = await cache.get_something(
            symbol=feed.symbol,           # Feed metadata
            exchange=feed.exchange.name   # Feed metadata
        )

    # 3. Store using feed_num as key
    self.some_dict[feed_num] = data
```

### Multi-Feed Cache Updates

When updating cache for multiple feeds, build complete structure:

```python
async def update_all_feed_status(self):
    """Update status for all feeds in a single cache write."""
    # Build complete status structure
    status_data = {}

    for feed in self.feeds:
        feed_num = feed.order

        status_data[f"feed_{feed_num}"] = {
            "symbol": feed.symbol,
            "period": feed.period,
            "current_price": (
                self.tick_buffer[feed_num][-1]
                if self.tick_buffer[feed_num] else 0.0
            ),
            "position_volume": (
                self.position[feed_num].volume
                if self.position[feed_num] else 0.0
            ),
            "unrealized_pnl": (
                self.position[feed_num].unrealized_pnl
                if self.position[feed_num] else 0.0
            )
        }

    # Single cache write with all feeds
    async with BotCache() as cache:
        await cache.update_bot(str(self.strategy_orm.bot_id), status_data)
```

### Cache-Feed Mapping Summary

| Cache Type | Input (Feed Metadata) | Output Key | Usage |
|------------|----------------------|------------|-------|
| TickCache | `feed.symbol`, `feed.exchange.name` | Store at `feed_num` | Get current prices |
| AccountCache | `feed.symbol`, `feed.ex_id` | Store at `feed_num` | Validate positions |
| BotCache | `feed.order` | `feed_{feed_num}` | Update bot status |
| OrdersCache | `feed.symbol`, `feed.ex_id` | N/A (queried later) | Check pending orders |
| TradesCache | `feed.symbol`, `feed.ex_id` | N/A (queried later) | Check recent fills |

**Key Insight**: Cache operations use feed metadata (symbol, exchange) for lookup, but results are stored in-memory using `feed_num` (Feed.order) as the dictionary key.

---

## Feed Mapping to TimeseriesRepository

When loading historical OHLCV data, strategies use `fullon_ohlcv.TimeseriesRepository` which requires mapping Feed ORM attributes to repository parameters.

### Feed to Repository Parameter Mapping

| Feed Attribute | TimeseriesRepository Parameter | Example |
|----------------|-------------------------------|---------|
| `feed.exchange.name` | `exchange` | "kraken" |
| `feed.symbol.symbol` | `symbol` | "BTC/USDT" |
| `feed.compression` | `compression` | 1, 5, 15 |
| `feed.period` | `period` | "minutes", "hours", "days" |
| N/A (automatic) | `test` | `False` (production DB) |

### Period Configuration

Feed period stored in database is a shorthand like "1m", "5m", "1h":

| Feed.period | Compression | Period | Description |
|-------------|-------------|---------|-------------|
| "1m" | 1 | "minutes" | 1-minute candles |
| "5m" | 5 | "minutes" | 5-minute candles |
| "15m" | 15 | "minutes" | 15-minute candles |
| "1h" | 1 | "hours" | 1-hour candles |
| "4h" | 4 | "hours" | 4-hour candles |
| "1d" | 1 | "days" | 1-day candles |

### Basic OHLCV Loading Pattern

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

async def load_feed_ohlcv(self, feed: Feed) -> pd.DataFrame:
    """Load historical OHLCV data for a feed."""

    # Map Feed attributes to TimeseriesRepository parameters
    # NOTE: Table/view initialization handled by fullon_ohlcv_service
    async with TimeseriesRepository(
        exchange=feed.exchange.name,    # Feed -> Repository
        symbol=feed.symbol.symbol,      # Feed -> Repository
        test=False                      # Production database
    ) as repo:
        # Define time range using arrow
        end_time = arrow.utcnow()
        start_time = end_time.shift(hours=-500)  # Last ~500 bars

        # Fetch OHLCV as DataFrame
        df = await repo.fetch_ohlcv_df(
            compression=feed.compression,   # Feed -> Repository
            period=feed.period,            # Feed -> Repository
            fromdate=start_time,
            todate=end_time
        )

        # Store using feed_num as key
        feed_num = feed.order
        self.dataframe[feed_num] = df

        return df
```

### Multi-Feed Loading in init()

Strategies typically load OHLCV data for all feeds during initialization:

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow

async def init(self):
    """Initialize strategy and load all OHLCV feeds."""

    for feed in self.feeds:
        feed_num = feed.order

        # Skip tick feeds (loaded from cache during execution)
        if feed.period.lower() == "tick":
            self.tick_buffer[feed_num] = []
            continue

        # Load historical OHLCV for this feed
        # NOTE: Table/view initialization handled by fullon_ohlcv_service
        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            # Calculate time range based on strategy needs
            # Example: Load 500 bars of historical data
            end_time = arrow.utcnow()
            start_time = end_time.shift(hours=-500)

            # Fetch OHLCV data
            df = await repo.fetch_ohlcv_df(
                compression=feed.compression,
                period=feed.period,
                fromdate=start_time,
                todate=end_time
            )

            # Store using feed_num as key
            self.dataframe[feed_num] = df

            # Initialize per-feed tracking
            self.position[feed_num] = None
            self.signal[feed_num] = None
            self.bar_completed[feed_num] = False

            self.logger.info(
                "Feed OHLCV loaded",
                feed_num=feed_num,
                symbol=feed.symbol.symbol,
                period=feed.period,
                rows=len(df)
            )
```

### Data Source Inspection

TimeseriesRepository provides properties to inspect which data sources were used:

```python
async with TimeseriesRepository(
    exchange=feed.exchange.name,
    symbol=feed.symbol.symbol,
    test=False
) as repo:
    # Available data sources for this symbol
    print(repo.data_sources)
    # {'continuous_aggregate': 'BTC_USDT_candles1m_view',
    #  'candles': 'BTC_USDT_candles1m',
    #  'trades': 'BTC_USDT_trades'}

    # Primary source (fastest available)
    print(repo.primary_source)  # "continuous_aggregate"

    # Fetch data
    df = await repo.fetch_ohlcv_df(
        compression=feed.compression,
        period=feed.period,
        fromdate=arrow.utcnow().shift(days=-1),
        todate=arrow.utcnow()
    )

    # Check which source was actually used
    print(repo.last_used_source)  # "continuous_aggregate"

    # Log performance for monitoring
    self.logger.info(
        "OHLCV loaded",
        feed_num=feed.order,
        rows=len(df),
        source=repo.last_used_source
    )
```

### Three-Tier Data Architecture

TimeseriesRepository automatically selects the fastest available data source:

1. **Continuous Aggregates** (fastest, 50-90% faster)
   - Pre-computed views for common periods
   - Transparent Redis caching
   - Automatic fallback if unavailable

2. **Candles Tables** (medium speed)
   - Direct TimescaleDB storage
   - Available for all active symbols

3. **Trades Tables** (slowest, always available)
   - Raw trade data aggregated on-the-fly
   - Fallback when candles not yet created

**Key Insight**: Strategies don't need to worry about which source is used - TimeseriesRepository handles optimization automatically.

### Period Conversion Helper

```python
def parse_feed_period(period: str) -> tuple[int, str]:
    """
    Parse Feed.period string into compression and period.

    Args:
        period: Feed period like "1m", "5m", "1h", "4h", "1d"

    Returns:
        (compression, period_name) tuple

    Examples:
        "1m" -> (1, "minutes")
        "5m" -> (5, "minutes")
        "1h" -> (1, "hours")
        "4h" -> (4, "hours")
        "1d" -> (1, "days")
    """
    period_map = {
        'm': 'minutes',
        'h': 'hours',
        'd': 'days'
    }

    # Extract number and unit
    num_str = period[:-1]  # e.g., "5" from "5m"
    unit = period[-1]      # e.g., "m" from "5m"

    compression = int(num_str)
    period_name = period_map.get(unit, 'minutes')

    return compression, period_name

# Usage in strategy
for feed in self.feeds:
    if feed.period != "tick":
        compression, period = parse_feed_period(feed.period)

        async with TimeseriesRepository(
            exchange=feed.exchange.name,
            symbol=feed.symbol.symbol,
            test=False
        ) as repo:
            df = await repo.fetch_ohlcv_df(
                compression=compression,
                period=period,
                fromdate=start_time,
                todate=end_time
            )
```

### Feed-Repository Mapping Summary

The pattern for loading OHLCV data from feeds:

```python
# Pattern: Feed -> TimeseriesRepository -> DataFrame -> feed_num storage
for feed in self.feeds:
    feed_num = feed.order  # 1. Get feed_num

    if feed.period == "tick":
        continue  # Skip tick feeds

    # 2. Create repository with feed metadata
    async with TimeseriesRepository(
        exchange=feed.exchange.name,    # Feed metadata
        symbol=feed.symbol.symbol,      # Feed metadata
        test=False                      # Production
    ) as repo:
        # 3. Fetch OHLCV using feed configuration
        df = await repo.fetch_ohlcv_df(
            compression=feed.compression,  # Feed config
            period=feed.period,           # Feed config
            fromdate=start_time,
            todate=end_time
        )

        # 4. Store using feed_num as key
        self.dataframe[feed_num] = df
```

**Critical Notes:**
- Always use context manager: `async with TimeseriesRepository(...) as repo:`
- Use `arrow.Arrow` for timestamps (not datetime)
- Set `test=False` for production database
- Table/view initialization handled automatically by fullon_ohlcv_service
- NO manual `init_symbol()` calls needed
- Repository automatically selects fastest data source

---

## Best Practices

### DO:
✅ Use `feed.order` as the dictionary key throughout your strategy
✅ Iterate `self.feeds` to process all feeds
✅ Document which feed order corresponds to which symbol in child strategy
✅ Use helper methods to get feeds by symbol if needed

### DON'T:
❌ Use `feed.feed_id` as the dictionary key (it's an auto-increment primary key)
❌ Assume feed order will be sequential (1, 2, 3...) - user could set any order
❌ Hardcode feed numbers without documenting what they represent
❌ Create complex indexing schemes - keep it simple

---

## Examples

### Example 1: Single-Feed Strategy

```python
class SinglePairStrategy(BaseStrategy):
    """Strategy that trades only one pair."""

    def __init__(self, strategy_orm: Strategy):
        super().__init__(strategy_orm)

        # Only one feed expected
        if len(self.feeds) != 1:
            raise ValueError("This strategy requires exactly 1 feed")

        # Get the feed number
        self.feed_num = self.feeds[0].order

    def generate_signals(self):
        """Simple - only one feed to check."""
        df = self._dataframe[self.feed_num]
        rsi = df['RSI_14'].iloc[-1]

        if rsi < 30:
            self.signal[self.feed_num] = "buy"
        elif rsi > 70:
            self.signal[self.feed_num] = "sell"
```

### Example 2: Multi-Feed Spread Strategy

```python
class SpreadStrategy(BaseStrategy):
    """Trade based on price spread between two pairs."""

    def __init__(self, strategy_orm: Strategy):
        super().__init__(strategy_orm)

        # Requires exactly 2 feeds
        if len(self.feeds) != 2:
            raise ValueError("Spread strategy requires exactly 2 feeds")

        # Document feed assignments
        self.primary_feed_num = 1  # Feed order=1 (e.g., BTC/USD)
        self.secondary_feed_num = 2  # Feed order=2 (e.g., ETH/USD)

    def generate_signals(self):
        """Generate signals based on spread."""
        # Get dataframes
        df1 = self._dataframe[self.primary_feed_num]
        df2 = self._dataframe[self.secondary_feed_num]

        # Calculate spread
        price1 = df1['close'].iloc[-1]
        price2 = df2['close'].iloc[-1]
        spread_pct = (price1 / price2 - 1.0) * 100

        # Generate signals based on spread
        if spread_pct > 5:
            # Spread too wide - sell primary, buy secondary
            self.signal[self.primary_feed_num] = "sell"
            self.signal[self.secondary_feed_num] = "buy"
        elif spread_pct < -5:
            # Spread too narrow - buy primary, sell secondary
            self.signal[self.primary_feed_num] = "buy"
            self.signal[self.secondary_feed_num] = "sell"
```

### Example 3: Basket Strategy

```python
class BasketStrategy(BaseStrategy):
    """Trade a basket of symbols with equal weighting."""

    def __init__(self, strategy_orm: Strategy):
        super().__init__(strategy_orm)

        # Support any number of feeds
        self.logger.info(f"Basket strategy with {len(self.feeds)} feeds")

    def generate_signals(self):
        """Generate signals for all feeds in basket."""
        # Calculate average RSI across all feeds
        rsi_values = []

        for feed in self.feeds:
            feed_num = feed.order

            if feed.period == "tick":
                continue

            df = self._dataframe[feed_num]
            rsi = df['RSI_14'].iloc[-1]
            rsi_values.append(rsi)

        # Basket-wide signal based on average
        avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 50

        if avg_rsi < 30:
            # Buy all feeds in basket
            for feed in self.feeds:
                if feed.period != "tick":
                    self.signal[feed.order] = "buy"
        elif avg_rsi > 70:
            # Sell all feeds in basket
            for feed in self.feeds:
                if feed.period != "tick":
                    self.signal[feed.order] = "sell"
```

---

## Summary

Feed numbering in fullon_strategies is intentionally simple:

- **`feed_num = Feed.order`** (from database)
- User controls feed ordering when configuring strategy
- All per-feed dictionaries use `feed.order` as key
- Child strategies iterate `self.feeds` and access data via `feed.order`

This design keeps the code clean, predictable, and easy to understand.
