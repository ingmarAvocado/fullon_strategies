from typing import Optional
from fullon_orm.models import Feed
from tests.factories.base import BaseFactory


class FeedFactory(BaseFactory):
    """Factory for creating Feed test objects."""

    @classmethod
    async def create(
        cls,
        db,
        str_id: int,
        symbol_id: int,
        ex_id: int,
        period: Optional[str] = None,
        order: Optional[int] = None,
        **kwargs
    ) -> Feed:
        """Create and persist Feed to database."""
        counter = cls.get_next_id()
        
        feed = Feed(
            str_id=str_id,
            symbol_id=symbol_id,
            period=period or "1m",
            compression=kwargs.get("compression", 1),
            order=order or counter,
            **kwargs
        )
        feed.ex_id = ex_id

        # NOTE: Feed has no dedicated repository (no FeedRepository in fullon_orm).
        # Feeds are relationship entities managed through Strategy, so we use
        # direct session manipulation for test data creation.
        db.session.add(feed)
        await db.session.flush()
        await db.session.refresh(feed)
        return feed

    @classmethod
    def build(
        cls,
        str_id: int,
        symbol_id: int,
        ex_id: int,
        period: Optional[str] = None,
        order: Optional[int] = None,
        **kwargs
    ) -> Feed:
        """Build Feed without persisting."""
        counter = cls.get_next_id()
        
        feed = Feed(
            str_id=str_id,
            symbol_id=symbol_id,
            period=period or "1m",
            compression=kwargs.get("compression", 1),
            order=order or counter,
            **kwargs
        )
        feed.ex_id = ex_id
        return feed
