"""
StrategyLoader - Dynamic strategy class loading by name.

Enables polymorphic execution - load different strategy classes at runtime
based on the class_name stored in the database.
"""
import importlib
from typing import Type, Optional

from .base_strategy import BaseStrategy
from fullon_log import get_component_logger

logger = get_component_logger("fullon.strategies.loader")


class StrategyLoader:
    """
    Dynamically loads strategy classes by name.

    Usage:
        # Load strategy class
        strategy_class = StrategyLoader.load("RSIStrategy")

        # Create instance
        strategy = strategy_class(strategy_orm_object)

        # Run
        await strategy.init()
        await strategy.run()
    """

    # Registry of known strategies (for now, empty - will be populated)
    _registry: dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseStrategy]):
        """
        Register a strategy class.

        Args:
            name: Strategy class name
            strategy_class: Strategy class (must inherit from BaseStrategy)
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class} must inherit from BaseStrategy")

        cls._registry[name] = strategy_class
        logger.info(f"Registered strategy: {name}")

    @classmethod
    def load(cls, class_name: str) -> Type[BaseStrategy]:
        """
        Load a strategy class by name.

        First checks the registry, then attempts to import from
        fullon_strategies.strategies module.

        Args:
            class_name: Name of strategy class (e.g., "RSIStrategy")

        Returns:
            Strategy class (NOT instance - you must instantiate it)

        Raises:
            ValueError: If strategy class not found or doesn't inherit from BaseStrategy
        """
        logger.info(f"Loading strategy class: {class_name}")

        # Check registry first
        if class_name in cls._registry:
            logger.debug(f"Found {class_name} in registry")
            return cls._registry[class_name]

        # Try to import from fullon_strategies.strategies module
        try:
            module = importlib.import_module(f"fullon_strategies.strategies.{class_name.lower()}")
            strategy_class = getattr(module, class_name)

            if not issubclass(strategy_class, BaseStrategy):
                raise ValueError(f"{class_name} must inherit from BaseStrategy")

            # Cache in registry
            cls._registry[class_name] = strategy_class
            logger.info(f"Loaded strategy class: {class_name}")

            return strategy_class

        except ImportError as e:
            logger.error(f"Failed to import strategy {class_name}: {e}")
            raise ValueError(f"Strategy class {class_name} not found") from e

        except AttributeError as e:
            logger.error(f"Strategy class {class_name} not found in module: {e}")
            raise ValueError(f"Strategy class {class_name} not found in module") from e

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return list(cls._registry.keys())

    @classmethod
    def clear_registry(cls):
        """Clear the strategy registry (for testing)."""
        cls._registry.clear()
        logger.debug("Strategy registry cleared")
