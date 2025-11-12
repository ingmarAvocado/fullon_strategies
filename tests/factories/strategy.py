from typing import Optional, List
from fullon_orm.models import Strategy, Feed
from tests.factories.base import BaseFactory


class StrategyFactory(BaseFactory):
    """Factory for creating Strategy test objects."""

    @classmethod
    async def create(
        cls,
        db,
        bot_id: Optional[int] = None,
        cat_str_id: Optional[int] = None,
        feeds: Optional[List[Feed]] = None,
        **kwargs
    ) -> Strategy:
        """Create and persist Strategy to database."""
        strategy = Strategy(
            bot_id=bot_id or 1,
            cat_str_id=cat_str_id or 1,
            **kwargs
        )
        
        saved = await db.strategies.add_bot_strategy(strategy)
        await db.commit()
        
        # Attach feeds if provided
        if feeds:
            # This assumes the ORM object can be updated like this.
            # Depending on the ORM, we might need a different approach.
            for feed in feeds:
                feed.str_id = saved.str_id
                db.session.add(feed)
            await db.commit()
            # Re-fetch to get the complete object with relationships
            saved = await db.strategies.get_by_id(saved.str_id)

        return saved

    @classmethod
    def build(
        cls,
        bot_id: Optional[int] = None,
        cat_str_id: Optional[int] = None,
        **kwargs
    ) -> Strategy:
        """Build Strategy without persisting."""
        return Strategy(
            bot_id=bot_id or 1,
            cat_str_id=cat_str_id or 1,
            **kwargs
        )
