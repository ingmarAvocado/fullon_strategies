"""
BaseStrategy - Abstract parent class for all trading strategies.

All custom strategies must inherit from this class and implement:
- on_tick(tick: Tick) - Called when new tick data arrives
- on_bar(df: pd.DataFrame) - Called when new OHLCV bar completes
"""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from fullon_orm.models import Strategy, Tick
from fullon_log import get_component_logger


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Provides:
    - Access to strategy configuration (from Strategy ORM object)
    - Feed management via FeedLoader
    - Logging infrastructure
    - Abstract methods that must be implemented by child strategies

    Usage:
        class MyStrategy(BaseStrategy):
            async def on_tick(self, tick: Tick):
                if tick.price > self.threshold:
                    await self.place_order("buy", 0.1)

            async def on_bar(self, df: pd.DataFrame):
                rsi = self.calculate_rsi(df['close'], 14)
                if rsi < 30:
                    await self.place_order("buy", 0.1)
    """

    def __init__(self, strategy_orm: Strategy):
        """
        Initialize BaseStrategy.

        Args:
            strategy_orm: Strategy ORM object from database with feeds, bot, etc.
        """
        self.strategy_orm = strategy_orm
        self.str_id = strategy_orm.str_id
        self.name = strategy_orm.name

        # Set up logging
        strategy_name = strategy_orm.cat_strategy.class_name if strategy_orm.cat_strategy else "unknown"
        self.logger = get_component_logger(f"fullon.strategies.{strategy_name}")

        # Feed loader will be set during init()
        self.feed_loader: Optional['FeedLoader'] = None  # noqa: F821

        self.logger.info(
            "Strategy initialized",
            str_id=self.str_id,
            name=self.name,
            class_name=strategy_name
        )

    async def init(self):
        """
        Initialize strategy - load feeds and prepare for execution.

        This is called after __init__ to perform async initialization.
        Override this in child classes if you need custom initialization,
        but remember to call super().init() first.
        """
        from .utils.feed_loader import FeedLoader

        self.logger.info("Initializing strategy", str_id=self.str_id)

        # Create feed loader and load all feeds
        self.feed_loader = FeedLoader(self.strategy_orm)
        await self.feed_loader.load_feeds()

        self.logger.info(
            "Strategy initialized successfully",
            str_id=self.str_id,
            feeds_loaded=len(self.feed_loader.feeds)
        )

    @abstractmethod
    async def on_tick(self, tick: Tick):
        """
        Called when new tick data arrives.

        Implement your tick-based trading logic here.

        Args:
            tick: Tick object with price, volume, exchange, symbol data
        """
        pass

    @abstractmethod
    async def on_bar(self, df: pd.DataFrame):
        """
        Called when new OHLCV bar completes.

        Implement your bar-based trading logic here.

        Args:
            df: pandas DataFrame with OHLCV data (timestamp, open, high, low, close, volume)
        """
        pass

    async def place_order(self, side: str, volume: float, price: Optional[float] = None):
        """
        Place an order (placeholder - to be implemented by fullon_bot).

        Args:
            side: "buy" or "sell"
            volume: Order size
            price: Optional limit price (None for market order)
        """
        self.logger.info(
            "Place order called",
            side=side,
            volume=volume,
            price=price,
            str_id=self.str_id
        )
        # TODO: Actual order placement will be implemented by fullon_bot
        pass

    async def stop(self):
        """
        Stop the strategy gracefully.

        Override this in child classes if you need custom cleanup.
        """
        self.logger.info("Strategy stopping", str_id=self.str_id)

    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(str_id={self.str_id}, name={self.name})"
