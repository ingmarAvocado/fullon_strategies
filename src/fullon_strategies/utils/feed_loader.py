"""
FeedLoader - Utility to load tick and OHLCV data for strategy feeds.

Responsibilities:
- Load tick data from fullon_cache.TickCache
- Load OHLCV data from fullon_ohlcv.CandleRepository
- Cache loaded feeds in memory
- Provide unified interface for strategies to access feed data
"""
from typing import Dict, Union, Optional
import pandas as pd

from fullon_orm.models import Strategy, Feed, Tick
from fullon_log import get_component_logger

logger = get_component_logger("fullon.strategies.feed_loader")


class FeedLoader:
    """
    Loads and manages feeds for a trading strategy.

    Usage:
        loader = FeedLoader(strategy_orm)
        await loader.load_feeds()

        # Access loaded data
        for feed in strategy_orm.feeds_list:
            data = loader.get_feed(feed.feed_id)
            if isinstance(data, Tick):
                print(f"Tick: {data.price}")
            else:
                print(f"OHLCV: {data.tail()}")
    """

    def __init__(self, strategy: Strategy):
        """
        Initialize FeedLoader.

        Args:
            strategy: Strategy ORM object with feeds_list relationship loaded
        """
        self.strategy = strategy
        self.feeds: Dict[int, Union[Tick, pd.DataFrame]] = {}

        logger.info(
            "FeedLoader initialized",
            str_id=strategy.str_id,
            num_feeds=len(strategy.feeds_list) if strategy.feeds_list else 0
        )

    async def load_feeds(self):
        """
        Load all feeds for the strategy.

        For each feed:
        - If period == "tick": Load from fullon_cache.TickCache
        - Otherwise: Load OHLCV DataFrame from fullon_ohlcv.CandleRepository
        """
        if not self.strategy.feeds_list:
            logger.warning("Strategy has no feeds", str_id=self.strategy.str_id)
            return

        for feed in self.strategy.feeds_list:
            try:
                if feed.period.lower() == "tick":
                    data = await self._load_tick(feed)
                else:
                    data = await self._load_ohlcv(feed)

                self.feeds[feed.feed_id] = data

                logger.info(
                    "Feed loaded",
                    feed_id=feed.feed_id,
                    symbol=feed.symbol.symbol if feed.symbol else None,
                    period=feed.period,
                    feed_type="tick" if feed.period.lower() == "tick" else "ohlcv"
                )

            except Exception as e:
                logger.error(
                    "Failed to load feed",
                    feed_id=feed.feed_id,
                    symbol=feed.symbol.symbol if feed.symbol else None,
                    period=feed.period,
                    error=str(e),
                    error_type=type(e).__name__
                )
                # Store None for failed feeds
                self.feeds[feed.feed_id] = None

    async def _load_tick(self, feed: Feed) -> Optional[Tick]:
        """
        Load tick data from fullon_cache.

        Args:
            feed: Feed ORM object with period == "tick"

        Returns:
            Tick object or None if not found
        """
        from fullon_cache import TickCache

        if not feed.symbol:
            logger.error("Feed has no symbol", feed_id=feed.feed_id)
            return None

        try:
            async with TickCache() as cache:
                tick = await cache.get_ticker(feed.symbol)

            if tick:
                logger.debug(
                    "Tick loaded from cache",
                    feed_id=feed.feed_id,
                    symbol=feed.symbol.symbol,
                    price=tick.price,
                    exchange=tick.exchange
                )
            else:
                logger.warning(
                    "Tick not found in cache",
                    feed_id=feed.feed_id,
                    symbol=feed.symbol.symbol
                )

            return tick

        except Exception as e:
            logger.error(
                "Failed to load tick from cache",
                feed_id=feed.feed_id,
                symbol=feed.symbol.symbol if feed.symbol else None,
                error=str(e)
            )
            return None

    async def _load_ohlcv(self, feed: Feed) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from fullon_ohlcv using TimeseriesRepository.

        TimeseriesRepository intelligently selects the fastest data source:
        - Continuous aggregates (50-90% faster when available)
        - Candles tables (medium speed)
        - Trades tables (fallback, always available)

        Args:
            feed: Feed ORM object with period like "1m", "5m", "1h"

        Returns:
            pandas DataFrame with OHLCV data or empty DataFrame if not found
        """
        from fullon_ohlcv.repositories.ohlcv import TimeseriesRepository
        import arrow

        if not feed.symbol or not feed.exchange:
            logger.error(
                "Feed missing symbol or exchange",
                feed_id=feed.feed_id
            )
            return pd.DataFrame()

        try:
            # Create TimeseriesRepository with context manager
            # NOTE: init_symbol() handled automatically by fullon_ohlcv_service
            async with TimeseriesRepository(
                exchange=feed.exchange.name,
                symbol=feed.symbol.symbol,
                test=False  # Use production database
            ) as repo:
                # Define time range (last 500 bars as default)
                end_time = arrow.utcnow()
                start_time = end_time.shift(hours=-500)  # Approximate

                # Fetch OHLCV data as DataFrame
                df = await repo.fetch_ohlcv_df(
                    compression=feed.compression or 1,
                    period=feed.period,
                    fromdate=start_time,
                    todate=end_time
                )

                logger.info(
                    "OHLCV data loaded",
                    feed_id=feed.feed_id,
                    symbol=feed.symbol.symbol,
                    exchange=feed.exchange.name,
                    rows=len(df),
                    primary_source=repo.primary_source,
                    last_used_source=repo.last_used_source
                )

                return df

        except Exception as e:
            logger.error(
                "Failed to load OHLCV from fullon_ohlcv",
                feed_id=feed.feed_id,
                symbol=feed.symbol.symbol if feed.symbol else None,
                exchange=feed.exchange.name if feed.exchange else None,
                period=feed.period,
                error=str(e),
                error_type=type(e).__name__
            )
            return pd.DataFrame()

    def get_feed(self, feed_id: int) -> Union[Tick, pd.DataFrame, None]:
        """
        Get loaded feed data by feed_id.

        Args:
            feed_id: Feed identifier

        Returns:
            Tick object (for tick feeds),
            pandas DataFrame (for OHLCV feeds),
            or None if feed not loaded
        """
        return self.feeds.get(feed_id)

    def get_all_feeds(self) -> Dict[int, Union[Tick, pd.DataFrame]]:
        """Get all loaded feeds."""
        return self.feeds.copy()

    def is_loaded(self, feed_id: int) -> bool:
        """Check if a feed is loaded."""
        return feed_id in self.feeds and self.feeds[feed_id] is not None
