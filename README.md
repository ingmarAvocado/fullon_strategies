# fullon_strategies

Dynamic strategy system for the Fullon trading platform.

## Overview

`fullon_strategies` provides a polymorphic strategy framework for building trading bots. Strategies can dynamically load, access feeds (tick or OHLCV), and execute trading logic.

## Key Features

- **BaseStrategy**: Parent class for all trading strategies
- **Dynamic Loading**: Load strategy classes by name at runtime
- **FeedLoader Utility**: Access tick data (from fullon_cache) and OHLCV data (from fullon_ohlcv)
- **Polymorphic Design**: Each strategy implements its own `on_tick()` and `on_bar()` methods
- **ORM Integration**: Uses fullon_orm Feed, Symbol, and Tick models

## Installation

```bash
poetry install
```

## Dependencies

**Required:**
- `fullon_log` - Structured logging
- `fullon_orm` - Database models (Feed, Symbol, Strategy, Tick)
- `fullon_cache` - Real-time tick data from Redis
- `fullon_ohlcv` - Historical OHLCV candle data

**External:**
- `pandas` - DataFrame operations for OHLCV data

## Architecture

### Strategy Execution Flow

```
Bot (fullon_bot)
  └─> Loads Strategy (strategy_loader)
       └─> Strategy.init()
            ├─> FeedLoader.load_feeds()  # Load tick/OHLCV data
            │    ├─> TickCache (for period="tick")
            │    └─> CandleRepository (for period="1m", "5m", etc.)
            └─> Strategy.on_tick() / on_bar()  # Execute trading logic
```

### Core Components

1. **BaseStrategy** (`src/fullon_strategies/base_strategy.py`)
   - Abstract parent class
   - Provides `on_tick()` and `on_bar()` hooks
   - Manages feed access

2. **FeedLoader** (`src/fullon_strategies/utils/feed_loader.py`)
   - Loads tick data from fullon_cache
   - Loads OHLCV data from fullon_ohlcv
   - Provides unified interface for strategies

3. **StrategyLoader** (`src/fullon_strategies/strategy_loader.py`)
   - Dynamically loads strategy classes by name
   - Enables polymorphic strategy execution

## Usage

### Creating a Strategy

```python
from fullon_strategies import BaseStrategy
from fullon_orm.models import Tick
import pandas as pd

class MyStrategy(BaseStrategy):
    """Custom trading strategy."""

    async def on_tick(self, tick: Tick):
        """Called when new tick data arrives."""
        if tick.price > 50000:
            await self.place_order("buy", 0.1)

    async def on_bar(self, df: pd.DataFrame):
        """Called when new OHLCV bar completes."""
        if df['close'].iloc[-1] > df['close'].iloc[-2]:
            await self.place_order("buy", 0.1)
```

### Loading Feeds

```python
from fullon_strategies.utils import FeedLoader
from fullon_orm.models import Strategy, Feed

# Load strategy from database
async with DatabaseContext() as db:
    strategy = await db.strategies.get_by_id(str_id)

    # Create feed loader
    loader = FeedLoader(strategy)

    # Load all feeds
    await loader.load_feeds()

    # Access loaded data
    for feed in strategy.feeds_list:
        if feed.period == "tick":
            tick = loader.get_feed(feed.feed_id)
            print(f"Tick: {tick.price}")
        else:
            df = loader.get_feed(feed.feed_id)
            print(f"OHLCV: {df.tail()}")
```

## Testing

Tests follow the fullon_ohlcv pattern:
- Per-worker test databases (parallel execution)
- Real PostgreSQL/TimescaleDB (no mocking)
- Factory pattern for test data

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fullon_strategies

# Run specific test
pytest tests/unit/test_feed_loader.py

# Run in parallel
pytest -n auto
```

## Development

See `CLAUDE.md` for comprehensive development guidelines.

## Status

Initial development phase - foundation being built.
