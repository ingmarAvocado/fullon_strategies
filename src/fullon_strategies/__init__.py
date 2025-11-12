"""
Fullon Strategies - Dynamic strategy system for trading bots.

This package provides base classes and utilities for building trading strategies.
"""

__version__ = "0.1.0"

from .base_strategy import BaseStrategy
from .strategy_loader import StrategyLoader

__all__ = [
    "__version__",
    "BaseStrategy",
    "StrategyLoader",
]
