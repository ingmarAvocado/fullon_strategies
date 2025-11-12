# 🔧 Method Reference

Quick reference of available methods based on examples.

## Key Features

- **Three-Tier Data Architecture**: Automatic selection between continuous aggregates, candles, and trades
- **Redis Caching**: 50-90% performance improvement for repeated OHLCV queries
- **Shared Engine Pattern**: 83% reduction in database connections
- **Async-First**: All operations use async/await for optimal performance
- **TimescaleDB Optimized**: Hypertables and continuous aggregates for time-series data

## TradeRepository Methods

```python
from fullon_ohlcv.repositories.ohlcv import TradeRepository
```

### Initialization
- `TradeRepository(exchange, symbol, test=True)` - Create repository
- `async with TradeRepository(...) as repo:` - Context manager (recommended)
- `await repo.initialize()` - Initialize connection, schema, and TimescaleDB extension
- `await repo.init_symbol()` - Create all database tables/views for symbol
- `await repo.close()` - Manual cleanup

**What init_symbol() creates**:
1. `{symbol}_trades` - Raw trade data table (TimescaleDB hypertable)
2. `{symbol}_candles1m` - 1-minute candles table (TimescaleDB hypertable)
3. `{symbol}_candles1m_view` - Continuous aggregate materialized view

### Save Data
- `await repo.save_trades(trades: List[Trade]) -> bool` - Save trade list

### Query Data
- `await repo.get_recent_trades(limit=100) -> List[Trade]` - Get recent trades
- `await repo.get_trades_in_range(start, end, limit=10) -> List[Trade]` - Get trades by time range

### Timestamps
- `await repo.get_oldest_timestamp() -> Optional[datetime]` - Get oldest trade timestamp
- `await repo.get_latest_timestamp() -> Optional[datetime]` - Get latest trade timestamp

## CandleRepository Methods

```python
from fullon_ohlcv.repositories.ohlcv import CandleRepository
```

### Initialization
- `CandleRepository(exchange, symbol, test=True)` - Create repository
- `async with CandleRepository(...) as repo:` - Context manager (recommended)
- `await repo.initialize()` - Initialize connection, schema, and TimescaleDB extension
- `await repo.init_symbol()` - Create all database tables/views for symbol (see TradeRepository for details)
- `await repo.close()` - Manual cleanup

### Save Data  
- `await repo.save_candles(candles: List[Candle]) -> bool` - Save candle list

### Timestamps
- `await repo.get_oldest_timestamp() -> Optional[arrow.Arrow]` - Get oldest candle timestamp
- `await repo.get_latest_timestamp() -> Optional[arrow.Arrow]` - Get latest candle timestamp

## TimeseriesRepository Methods

```python
from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
import arrow
```

### Initialization
- `TimeseriesRepository(exchange, symbol, test=True)` - Create repository
- `async with TimeseriesRepository(...) as repo:` - Context manager (recommended)
- `await repo.initialize()` - Initialize connection, schema, and TimescaleDB extension
- `await repo.init_symbol()` - Create all database tables/views for symbol (see TradeRepository for details)
- `await repo.close()` - Manual cleanup

### Three-Tier Data Source Architecture

TimeseriesRepository intelligently selects the best available data source for OHLCV queries:

**Priority Order (fastest to slowest)**:
1. **Continuous Aggregates** (`{symbol}_candles1m_view`) - Pre-computed, fastest (50-90% faster)
2. **Candles Tables** (`{symbol}_candles1m`) - Direct candle storage, medium speed
3. **Trades Tables** (`{symbol}_trades`) - On-the-fly aggregation, slowest but always available

**Properties**:
- `repo.data_sources` - Dict of available data sources: `{"continuous_aggregate": "table_name", "candles": "table_name", "trades": "table_name"}`
- `repo.primary_source` - String indicating the best available source being used
- `repo.last_used_source` - String showing which source was used in the last query

### OHLCV Data Generation
- `await repo.fetch_ohlcv(compression: int, period: str, fromdate: arrow.Arrow, todate: arrow.Arrow) -> List[tuple]` - Generate OHLCV candles from existing trade data
- `await repo.fetch_ohlcv_df(compression: int, period: str, fromdate: arrow.Arrow, todate: arrow.Arrow) -> pd.DataFrame` - Generate OHLCV candles as pandas DataFrame (requires pandas)

**Parameters**:
- `compression`: Number of time periods per candle (e.g., 5 for 5-minute candles)
- `period`: Time period unit - "minutes", "hours", "days"
- `fromdate`: Start timestamp (arrow.Arrow object)
- `todate`: End timestamp (arrow.Arrow object)

**Returns**:
- List of tuples: `(timestamp, open, high, low, close, volume)`
- Automatically uses the fastest available data source

### Timestamps
- `await repo.get_oldest_timestamp() -> Optional[arrow.Arrow]` - Get oldest data timestamp
- `await repo.get_latest_timestamp() -> Optional[arrow.Arrow]` - Get latest data timestamp

## Utilities

```python
from fullon_ohlcv.utils import install_uvloop
```

### Performance
- `install_uvloop()` - Install uvloop for better async performance (call before asyncio.run())

## CacheManager (Redis Caching)

```python
from fullon_ohlcv.utils.cache import CacheManager
```

### Overview
TimeseriesRepository automatically uses Redis caching for OHLCV queries when Redis is available. Caching provides 50-90% performance improvements for repeated queries.

### Features
- Automatic cache key generation based on exchange, symbol, timeframe, and date range
- Cache invalidation when new data is added
- Different timeframes cached separately
- Graceful degradation when Redis is unavailable

### Methods
- `cache_mgr = CacheManager()` - Create cache manager instance
- `await cache_mgr.get_stats() -> dict` - Get cache statistics (hits, misses, hit rate)
- `cache_mgr.enabled` - Boolean indicating if caching is enabled

### Cache Statistics
```python
cache_mgr = CacheManager()
if cache_mgr.enabled:
    stats = await cache_mgr.get_stats()
    print(f"Cache hits: {stats.get('keyspace_hits', 0)}")
    print(f"Cache misses: {stats.get('keyspace_misses', 0)}")
    print(f"Hit rate: {stats.get('hit_rate', 0):.3f}")
```

### Behavior
- **Cache Miss**: First query hits database, result cached for subsequent queries
- **Cache Hit**: Subsequent identical queries served from Redis (much faster)
- **Cache Invalidation**: Adding new trades/candles invalidates related cache entries
- **Timeframe Isolation**: 1m, 5m, 15m candles cached separately

## Shared Engine Pattern

### Overview
Multiple repository instances for the same database automatically share a single connection pool, dramatically reducing PostgreSQL connections.

### Benefits
- **Connection Reduction**: 150 connections → ~25 connections for 10 repositories
- **Resource Efficiency**: ~83% reduction in database connections
- **Automatic**: No configuration needed, works transparently

### Configuration
Controlled via environment variables or config:
```python
from fullon_ohlcv.utils.config import config

# Shared pool settings
config.database.shared_pool_size = 20
config.database.shared_pool_max_overflow = 5
# Total connections per database: 25 (20 + 5)
```

### Example
```python
# Create multiple repositories - they automatically share engines
repos = [
    CandleRepository("binance", f"SYM{i}/USDT", test=True)
    for i in range(10)
]

# All 10 repositories share the same engine and connection pool
# Without sharing: 10 repos × 15 connections = 150 connections
# With sharing: 1 shared pool = 25 connections (83% reduction)

# Verify sharing
assert all(repo.engine is repos[0].engine for repo in repos)
```

## Models

```python
from fullon_ohlcv.models import Trade, Candle
```

### Trade Model
```python
Trade(
    timestamp=datetime.now(timezone.utc),
    price=50000.0,
    volume=0.1,
    side="BUY",      # "BUY" or "SELL"
    type="MARKET"    # "MARKET" or "LIMIT"
)
```

### Candle Model
```python
Candle(
    timestamp=datetime.now(timezone.utc),
    open=3000.0,
    high=3010.0,
    low=2995.0,
    close=3005.0,
    vol=150.5
)
```

## Common Patterns

### Context Manager (Recommended)
```python
async with TradeRepository("binance", "BTC/USDT", test=True) as repo:
    # Initialize symbol tables first
    await repo.init_symbol()

    # Then save/query data
    success = await repo.save_trades(trades)
```

### Manual Management
```python
repo = CandleRepository("binance", "ETH/USDT", test=True)
await repo.initialize()
try:
    # Initialize symbol tables
    await repo.init_symbol()

    # Then save/query data
    success = await repo.save_candles(candles)
finally:
    await repo.close()
```

### Performance Setup
```python
from fullon_ohlcv.utils import install_uvloop
install_uvloop()  # Call before asyncio.run() for better performance
asyncio.run(main())
```

### Bulk Operations
```python
# Save multiple trades/candles at once
trades = [Trade(...), Trade(...), Trade(...)]
success = await repo.save_trades(trades)
```

### TimeseriesRepository OHLCV Generation
```python
import arrow
from datetime import datetime, timezone, timedelta

# Generate 5-minute OHLCV candles from trade data
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=24)

async with TimeseriesRepository("binance", "BTC/USDT", test=True) as repo:
    # Initialize symbol tables
    await repo.init_symbol()

    # Fetch OHLCV data as list of tuples
    ohlcv_data = await repo.fetch_ohlcv(
        compression=5,
        period="minutes",
        fromdate=arrow.get(start_time),
        todate=arrow.get(end_time)
    )
    # Returns list of tuples: (timestamp, open, high, low, close, volume)
    for ts, open_price, high_price, low_price, close_price, volume in ohlcv_data:
        print(f"Candle: {ts} O:{open_price} H:{high_price} L:{low_price} C:{close_price} V:{volume}")

    # Or get as pandas DataFrame (requires pandas)
    df = await repo.fetch_ohlcv_df(
        compression=5,
        period="minutes",
        fromdate=arrow.get(start_time),
        todate=arrow.get(end_time)
    )
    print(df.head())  # Display first 5 rows
```

### Check TimeseriesRepository Data Sources
```python
async with TimeseriesRepository("binance", "BTC/USDT", test=True) as repo:
    await repo.initialize()
    await repo.init_symbol()

    # Check which data sources are available
    print(f"Available sources: {repo.data_sources}")
    # Example: {'continuous_aggregate': 'BTC_USDT_candles1m_view',
    #           'candles': 'BTC_USDT_candles1m',
    #           'trades': 'BTC_USDT_trades'}

    print(f"Primary source: {repo.primary_source}")
    # Example: 'continuous_aggregate'

    # Fetch data
    ohlcv = await repo.fetch_ohlcv(compression=5, period="minutes", ...)

    # Check which source was actually used
    print(f"Used source: {repo.last_used_source}")
    # Example: 'continuous_aggregate'
```

### Monitor Cache Performance
```python
from fullon_ohlcv.utils.cache import CacheManager

async with TimeseriesRepository("binance", "BTC/USDT", test=True) as repo:
    await repo.initialize()
    await repo.init_symbol()

    # First query (cache miss)
    data1 = await repo.fetch_ohlcv(compression=5, period="minutes", ...)

    # Second query (cache hit - much faster)
    data2 = await repo.fetch_ohlcv(compression=5, period="minutes", ...)

    # Check cache statistics
    cache_mgr = CacheManager()
    if cache_mgr.enabled:
        stats = await cache_mgr.get_stats()
        print(f"Cache hit rate: {stats.get('hit_rate', 0):.1%}")
```

### Create Multiple Repositories with Shared Engine
```python
# All repositories for the same database share a connection pool
repos = []
for symbol in ["BTC/USDT", "ETH/USDT", "ADA/USDT"]:
    repo = TradeRepository("binance", symbol, test=True)
    await repo.initialize()
    repos.append(repo)

# Verify all repositories share the same engine
print(f"All share engine: {all(r.engine is repos[0].engine for r in repos)}")
# True - they all use the same connection pool

# Clean up
for repo in repos:
    await repo.close()
```

## Database Management

```bash
# Create database with TimescaleDB
python src/fullon_ohlcv/install_ohlcv.py my_database_name

# Delete database
python src/fullon_ohlcv/install_ohlcv.py --delete my_database_name
```

## Performance Tips

1. **Use uvloop**: Always call `install_uvloop()` before `asyncio.run()` for 2-4x async performance boost
2. **Enable Redis caching**: Set up Redis server to get automatic caching for OHLCV queries
3. **Use continuous aggregates**: Call `init_symbol()` to create pre-computed aggregates for fastest queries
4. **Batch operations**: Save multiple trades/candles at once rather than one at a time
5. **Context managers**: Use `async with` pattern for automatic resource cleanup
6. **Monitor data sources**: Check `repo.last_used_source` to verify optimal source is being used

## Related Examples

- `examples/trade_repository_example.py` - Basic trade operations
- `examples/candle_repository_example.py` - Basic candle operations
- `examples/timeseries_repository_example.py` - Three-tier architecture and performance comparison
- `examples/init_symbol_example.py` - Database initialization
- `examples/cache_demo.py` - Redis caching demonstration
- `examples/shared_engine_demo.py` - Connection pooling benefits

See example files for complete working code!