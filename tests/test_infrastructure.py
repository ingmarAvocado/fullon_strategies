"""
Test the test infrastructure itself.

Factory tests have been moved to tests/test_factories.py for better organization.
"""
import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_database_creation(db_context):
    """Test that test database is created and accessible."""
    # Just check that we can access the database
    assert db_context is not None
    # Try a simple query
    result = await db_context.session.execute(text("SELECT 1"))
    assert result.scalar() == 1