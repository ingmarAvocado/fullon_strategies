from typing import Optional
from fullon_orm.models import Exchange
from tests.factories.base import BaseFactory


class ExchangeFactory(BaseFactory):
    """Factory for creating Exchange test objects."""

    @classmethod
    async def create(
        cls,
        db,  # DatabaseContext
        uid: int,
        name: Optional[str] = None,
        cat_ex_id: Optional[int] = None,
        **kwargs
    ) -> Exchange:
        """
        Create and persist Exchange to database.
        
        Args:
            db: DatabaseContext instance
            uid: User ID
            name: Exchange name (default: "test_exchange_N")
            cat_ex_id: Category exchange ID (default: 1)
            **kwargs: Additional Exchange fields
            
        Returns:
            Persisted Exchange object
        """
        counter = cls.get_next_id()
        
        exchange = Exchange(
            name=name or f"test_exchange_{counter}",
            cat_ex_id=cat_ex_id or 1,
            uid=uid,
            **kwargs
        )
        
        # Persist to database
        saved = await db.exchanges.add_user_exchange(exchange)
        await db.commit()
        
        return saved

    @classmethod
    def build(
        cls,
        name: Optional[str] = None,
        cat_ex_id: Optional[int] = None,
        **kwargs
    ) -> Exchange:
        """
        Build Exchange without persisting (for unit tests).
        
        Args:
            name: Exchange name
            **kwargs: Additional fields
            
        Returns:
            Exchange object (not persisted)
        """
        counter = cls.get_next_id()
        return Exchange(
            name=name or f"test_exchange_{counter}",
            cat_ex_id=cat_ex_id or 1,
            **kwargs
        )
