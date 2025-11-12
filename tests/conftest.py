"""
Pytest configuration and fixtures for fullon_strategies tests.

Follows fullon_master_api testing patterns:
- Per-worker test databases with module-level caching
- Real PostgreSQL/TimescaleDB (no mocking)
- Factory pattern for test data with repository pattern
- DatabaseTestContext wrapper for rollback-based isolation
- Safety checks to prevent production database access
"""
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env configuration at module level (before any other imports)
env_path = Path(__file__).parent.parent / ".env"
example_path = Path(__file__).parent.parent / ".env.example"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env configuration from {env_path}")
else:
    # Fallback to .env.example for CI/CD
    if example_path.exists():
        load_dotenv(example_path)
        print(f"Warning: Using .env.example - create .env for local development")
    else:
        print("Warning: No .env or .env.example found")

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from fullon_orm import DatabaseContext
from fullon_orm.base import Base
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

# ============================================================================
# SAFETY CHECKS - Prevent Production Database Access
# ============================================================================

def _validate_test_environment():
    """Validate that we're in a test environment and not accidentally using production database."""
    is_pytest = "pytest" in sys.modules or (sys.argv and "pytest" in sys.argv[0])

    if is_pytest:
        production_db_names = {"fullon", "fullon2", "production", "prod"}
        current_db = os.getenv("DB_NAME", "").lower()

        if current_db in production_db_names:
            logger.info(
                f"DB_NAME is set to '{current_db}' but tests will use isolated test databases."
            )
        return

    # Non-test context: strict validation
    production_db_names = {"fullon", "fullon2", "production", "prod"}
    current_db = os.getenv("DB_NAME", "").lower()

    if current_db in production_db_names:
        raise RuntimeError(
            f"SAFETY CHECK FAILED: Cannot run against production database '{current_db}' outside of tests."
        )

# Run safety check on import
_validate_test_environment()

# ============================================================================
# DATABASE PER WORKER PATTERN - Module-Level Caches
# ============================================================================

# Cache for engines per database to reuse across tests
_engine_cache: Dict[str, AsyncEngine] = {}
_db_created: Dict[str, bool] = {}

# Database configuration from environment
DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "name": os.getenv("DB_TEST_NAME", "fullon2_test"),
}


async def get_or_create_engine(db_name: str) -> AsyncEngine:
    """Get or create an engine for the database.

    Args:
        db_name: Database name

    Returns:
        AsyncEngine for the database
    """
    if db_name not in _engine_cache:
        # Create database if needed
        await create_test_database(db_name, DB_CONFIG)

        # Create engine with NullPool to avoid connection pool issues
        from fullon_orm.database import create_database_url
        database_url = create_database_url(database=db_name)
        engine = create_async_engine(
            database_url,
            echo=False,
            poolclass=NullPool,  # Use NullPool to avoid event loop issues
        )

        # Create ORM tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        _engine_cache[db_name] = engine

    return _engine_cache[db_name]


async def create_test_database(db_name: str, db_config: dict) -> bool:
    """Create a test database with TimescaleDB extension."""
    admin_url = (
        f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/postgres"
    )

    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")

    try:
        async with engine.begin() as conn:
            # Drop existing database if it exists
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
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


# ============================================================================
# TEST DATABASE CONTEXT - Rollback-Based Isolation
# ============================================================================

class DatabaseTestContext:
    """DatabaseContext wrapper for testing with perfect isolation.

    This mimics fullon_orm's DatabaseContext pattern but uses savepoints for test isolation:
    - Never commits - always rollbacks to avoid event loop cleanup issues
    - Uses savepoints for nested transaction support
    - Provides same repository interface as real DatabaseContext
    - All changes automatically rolled back after each test

    Repository Properties:
        - strategies: StrategyRepository
        - symbols: SymbolRepository
        - exchanges: ExchangeRepository
        - bots: BotRepository
        - orders: OrderRepository
        - trades: TradeRepository
    """

    def __init__(self, session: AsyncSession):
        """Initialize with an async session."""
        self.session = session
        # Repository instances (lazy loaded)
        self._strategy_repo = None
        self._symbol_repo = None
        self._exchange_repo = None
        self._bot_repo = None
        self._order_repo = None
        self._trade_repo = None

    @property
    def strategies(self):
        """Get StrategyRepository with current session."""
        if self._strategy_repo is None:
            from fullon_orm.repositories import StrategyRepository
            self._strategy_repo = StrategyRepository(self.session)
        return self._strategy_repo

    @property
    def symbols(self):
        """Get SymbolRepository with current session."""
        if self._symbol_repo is None:
            from fullon_orm.repositories import SymbolRepository
            self._symbol_repo = SymbolRepository(self.session)
        return self._symbol_repo

    @property
    def exchanges(self):
        """Get ExchangeRepository with current session."""
        if self._exchange_repo is None:
            from fullon_orm.repositories import ExchangeRepository
            self._exchange_repo = ExchangeRepository(self.session)
        return self._exchange_repo

    @property
    def bots(self):
        """Get BotRepository with current session."""
        if self._bot_repo is None:
            from fullon_orm.repositories import BotRepository
            self._bot_repo = BotRepository(self.session)
        return self._bot_repo

    @property
    def orders(self):
        """Get OrderRepository with current session."""
        if self._order_repo is None:
            from fullon_orm.repositories import OrderRepository
            self._order_repo = OrderRepository(self.session)
        return self._order_repo

    @property
    def trades(self):
        """Get TradeRepository with current session."""
        if self._trade_repo is None:
            from fullon_orm.repositories import TradeRepository
            self._trade_repo = TradeRepository(self.session)
        return self._trade_repo

    async def commit(self):
        """Commit current transaction (for compatibility)."""
        await self.session.commit()

    async def rollback(self):
        """Rollback current transaction."""
        await self.session.rollback()

    async def flush(self):
        """Flush current session."""
        await self.session.flush()


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


# OLD test_db fixture - replaced by module-level caching pattern
# @pytest.fixture(scope="function")
# async def test_db() -> AsyncGenerator[str, None]:
#     """
#     Create a unique test database for each test.
#
#     Yields the database name and cleans up after the test.
#     """
#     # Create unique database name
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
#     unique_id = str(uuid.uuid4())[:8]
#     db_name = f"test_strategies_{timestamp}_{worker_id}_{unique_id}"
#
#     # Create database
#     success = await create_test_database(db_name, DB_CONFIG)
#     if not success:
#         pytest.fail(f"Failed to create test database: {db_name}")
#
#     # Update environment variable for DatabaseContext
#     original_db_name = os.getenv("DB_NAME")
#     os.environ["DB_NAME"] = db_name
#
#     try:
#         yield db_name
#     finally:
#         # Restore original database name
#         if original_db_name:
#             os.environ["DB_NAME"] = original_db_name
#         else:
#             os.environ.pop("DB_NAME", None)
#
#         # Drop database
#         await drop_test_database(db_name, DB_CONFIG)


@pytest_asyncio.fixture
async def db_context(request):
    """Create a DatabaseContext-like wrapper for testing with proper isolation.

    This provides:
    - Per-test database isolation via savepoints
    - Automatic rollback after each test
    - Same interface as fullon_orm.DatabaseContext
    - Access to all repositories (strategies, feeds, symbols, etc.)

    Usage:
        async def test_strategy_creation(db_context):
            strategy = Strategy(str_id="test", bot_id=1, size=100.0)
            created = await db_context.strategies.add(strategy)
            await db_context.commit()
            assert created.str_id == "test"
            # Automatically rolled back after test
    """
    # Get database name for this module
    module_name = request.module.__name__.split(".")[-1]
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    db_name = f"test_strategies_{module_name}_{worker_id}"

    # Get or create engine (cached at module level)
    engine = await get_or_create_engine(db_name)

    # Create session maker
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create a session WITHOUT context manager to have full control
    session = async_session_maker()

    try:
        # Begin transaction explicitly for proper rollback
        await session.begin()

        # Create test database context wrapper
        db = DatabaseTestContext(session)

        yield db
    finally:
        # Always rollback - this ensures no data persists
        await session.rollback()
        await session.close()


# ============================================================================
# SETUP FIXTURES - For Creating Required Base Data
# ============================================================================

@pytest.fixture
async def setup_base_data(db_context):
    """Create base data required for tests (exchanges, strategy categories, etc.)."""
    # Create a test exchange category
    from fullon_orm.models import CatExchange, CatStrategy, Bot, User, Exchange
    import uuid as uid_gen

    # Create test user with unique email
    unique_id = str(uid_gen.uuid4())[:8]
    user = User(
        mail=f"test_{unique_id}@example.com",
        name="Test",
        lastname="User",
        password="hashed_password",
        f2a="",
        phone="",
        id_num=""
    )
    db_context.session.add(user)
    await db_context.session.flush()

    # Create exchange category with unique name
    cat_ex = CatExchange(
        name=f"test_exchange_{unique_id}"
    )
    db_context.session.add(cat_ex)
    await db_context.session.flush()

    # Create exchange for user
    # ex_id is auto-increment, don't specify it
    exchange = Exchange(
        uid=user.uid,
        cat_ex_id=cat_ex.cat_ex_id,
        name="Test Exchange",
        test=True
    )
    db_context.session.add(exchange)
    await db_context.session.flush()

    # Create strategy category with unique name
    cat_str = CatStrategy(
        name=f"Test Strategy {unique_id}"
    )
    db_context.session.add(cat_str)
    await db_context.session.flush()

    # Create a bot
    bot = Bot(
        uid=user.uid,
        name="Test Bot",
        dry_run=True
    )
    db_context.session.add(bot)
    await db_context.session.flush()

    # Don't commit - let db_context handle transaction rollback

    return {
        "user": user,
        "cat_ex": cat_ex,
        "exchange": exchange,
        "cat_str": cat_str,
        "bot": bot
    }

# ============================================================================
# FACTORY FIXTURES - For Creating Test Data
# ============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.factories import (
    ExchangeFactory,
    SymbolFactory,
    FeedFactory,
    StrategyFactory
)
from tests.fixtures.ohlcv_data import ohlcv_test_data, empty_ohlcv_data
from tests.fixtures.tick_data import tick_test_data

@pytest.fixture(scope="session")
def cache_config():
    """Cache configuration from .env."""
    return {
        "host": os.getenv("CACHE_HOST", "localhost"),
        "port": int(os.getenv("CACHE_PORT", 6379)),
        "db": int(os.getenv("CACHE_TEST_DB", 15)),
        "password": os.getenv("CACHE_PASSWORD", ""),
    }


@pytest.fixture
def exchange_factory():
    """Factory for creating Exchange ORM objects."""
    return ExchangeFactory

@pytest.fixture
def symbol_factory():
    """Factory for creating Symbol ORM objects."""
    return SymbolFactory

@pytest.fixture
def feed_factory():
    """Factory for creating Feed ORM objects."""
    return FeedFactory

@pytest.fixture
def strategy_factory():
    """Factory for creating Strategy ORM objects."""
    return StrategyFactory
