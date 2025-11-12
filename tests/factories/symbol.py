from typing import Optional
from fullon_orm.models import Symbol
from tests.factories.base import BaseFactory


class SymbolFactory(BaseFactory):
    """Factory for creating Symbol test objects."""

    @classmethod
    async def create(
        cls,
        db,
        symbol: Optional[str] = None,
        symbol_base: Optional[str] = None,
        symbol_quote: Optional[str] = None,
        cat_ex_id: Optional[int] = None,
        **kwargs
    ) -> Symbol:
        """Create and persist Symbol to database."""
        if symbol and not (symbol_base and symbol_quote):
            parts = symbol.split('/')
            symbol_base = parts[0]
            symbol_quote = parts[1]

        sym = Symbol(
            symbol=symbol or f"BTC/USDT",
            base=symbol_base or "BTC",
            quote=symbol_quote or "USDT",
            cat_ex_id=cat_ex_id or 1,
            **kwargs
        )
        
        saved = await db.symbols.add_symbol(sym)
        await db.commit()
        return saved

    @classmethod
    def build(
        cls,
        symbol: Optional[str] = None,
        symbol_base: Optional[str] = None,
        symbol_quote: Optional[str] = None,
        cat_ex_id: Optional[int] = None,
        **kwargs
    ) -> Symbol:
        """Build Symbol without persisting."""
        if symbol and not (symbol_base and symbol_quote):
            parts = symbol.split('/')
            symbol_base = parts[0]
            symbol_quote = parts[1]

        return Symbol(
            symbol=symbol or "BTC/USDT",
            base=symbol_base or "BTC",
            quote=symbol_quote or "USDT",
            cat_ex_id=cat_ex_id or 1,
            **kwargs
        )
