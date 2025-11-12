"""
Pytest configuration and fixtures for fullon_strategies tests.

Follows fullon_ohlcv testing patterns:
- Per-worker test databases
- Real PostgreSQL/TimescaleDB (no mocking)
- Factory pattern for test data
"""
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env configuration at module level (before any other imports)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env configuration from {env_path}")
else:
    # Fallback to .env.example for CI/CD
    example_path = Path(__file__).parent.parent / ".env.example"
    if example_path.exists():
        load_dotenv(example_path)
        print(f"Warning: Using .env.example - create .env for local development")
    else:
        print("Warning: No .env or .env.example found")

import asyncio
import pytest
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from fullon_orm import DatabaseContext
from fullon_orm.models import Strategy, Feed, Symbol
from fullon_log import get_component_logger

logger = get_component_logger("fullon.strategies.tests")

# Log that environment was loaded successfully
if env_path.exists():
    logger.info(f"Loaded .env configuration for tests from {env_path}")
elif example_path and example_path.exists():
    logger.warning("Using .env.example - create .env for local development")
else:
    logger.warning("No .env configuration found")


# Database configuration from environment
DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "name": os.getenv("DB_TEST_NAME", "fullon2_test"),
}


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


async def create_test_database(db_name: str, db_config: dict) -> bool:
    """Create a test database with TimescaleDB extension."""
    admin_url = (
        f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/postgres"
    )

    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")

    try:
        async with engine.begin() as conn:
            # Create database
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            logger.info(f"Created test database: {db_name}")

        # Connect to new database and create TimescaleDB extension
        test_db_url = (
            f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_name}"
        )
        test_engine = create_async_engine(test_db_url, isolation_level="AUTOCOMMIT")

        async with test_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            logger.info(f"Created TimescaleDB extension in: {db_name}")

        await test_engine.dispose()
        return True

    except Exception as e:
        logger.error(f"Failed to create test database {db_name}: {e}")
        return False
    finally:
        await engine.dispose()


async def drop_test_database(db_name: str, db_config: dict) -> bool:
    """Drop a test database."""
    admin_url = (
        f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/postgres"
    )

    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")

    try:
        async with engine.begin() as conn:
            # Terminate existing connections
            await conn.execute(text(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
            """))

            # Drop database
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
            logger.info(f"Dropped test database: {db_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to drop test database {db_name}: {e}")
        return False
    finally:
        await engine.dispose()


@pytest.fixture(scope="function")
async def test_db() -> AsyncGenerator[str, None]:
    """
    Create a unique test database for each test.

    Yields the database name and cleans up after the test.
    """
    # Create unique database name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
    unique_id = str(uuid.uuid4())[:8]
    db_name = f"test_strategies_{timestamp}_{worker_id}_{unique_id}"

    # Create database
    success = await create_test_database(db_name, DB_CONFIG)
    if not success:
        pytest.fail(f"Failed to create test database: {db_name}")

    # Update environment variable for DatabaseContext
    original_db_name = os.getenv("DB_NAME")
    os.environ["DB_NAME"] = db_name

    try:
        yield db_name
    finally:
        # Restore original database name
        if original_db_name:
            os.environ["DB_NAME"] = original_db_name
        else:
            os.environ.pop("DB_NAME", None)

        # Drop database
        await drop_test_database(db_name, DB_CONFIG)


@pytest.fixture
async def db_context(test_db: str) -> AsyncGenerator[DatabaseContext, None]:
    """
    Provide a DatabaseContext with a clean test database.

    Usage:
        async def test_something(db_context):
            strategy = await db_context.strategies.get_by_id(1)
    """
    async with DatabaseContext() as db:
        yield db


# Factory fixtures for creating test data
@pytest.fixture
async def strategy_factory(db_context):
    """Factory for creating Strategy ORM objects in the test database."""
    async def _create_strategy(**kwargs):
        defaults = {
            "bot_id": 1,
            "str_id": "test_strategy",
            "size": 100.0,
        }
        defaults.update(kwargs)
        strategy = Strategy(**defaults)

        # Use session.add() + flush() pattern
        db_context.session.add(strategy)
        await db_context.session.flush()
        await db_context.commit()

        return strategy
    return _create_strategy


@pytest.fixture
async def feed_factory(db_context):
    """Factory for creating Feed ORM objects in the test database."""
    async def _create_feed(**kwargs):
        defaults = {
            "str_id": 1,
            "symbol": "BTC/USDT",
            "period": "1m",
            "compression": 1,
            "order": 1,
        }
        defaults.update(kwargs)
        feed = Feed(**defaults)

        # Use session.add() + flush() pattern
        db_context.session.add(feed)
        await db_context.session.flush()
        await db_context.commit()

        return feed
    return _create_feed


@pytest.fixture
async def symbol_factory(db_context):
    """Factory for creating Symbol ORM objects in the test database."""
    async def _create_symbol(**kwargs):
        defaults = {
            "symbol": "BTC/USDT",
            "base": "BTC",
            "quote": "USDT",
            "cat_ex_id": 1,
        }
        defaults.update(kwargs)
        symbol = Symbol(**defaults)

        # Use session.add() + flush() pattern
        db_context.session.add(symbol)
        await db_context.session.flush()
        await db_context.commit()

        return symbol
    return _create_symbol
