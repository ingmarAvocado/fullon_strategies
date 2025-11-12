"""
Example Strategy - Demonstrates how to create a custom trading strategy.

This example shows the complete strategy lifecycle implementation:
- PRE-LOOP: Custom initialization
- IN-LOOP: prepare_indicators(), generate_signals(), on_signal(), on_position()
- Follows the STRATEGY_LIFECYCLE.md specification
"""
import asyncio
import pandas as pd

from fullon_strategies import BaseStrategy
from fullon_orm.models import Strategy, Tick
from fullon_orm import DatabaseContext


class ExampleStrategy(BaseStrategy):
    """
    Example RSI + EMA crossover trading strategy.

    This strategy demonstrates the complete lifecycle:
    - Custom initialization (__init__)
    - Indicator preparation (prepare_indicators)
    - Signal generation (generate_signals)
    - Entry logic (on_signal)
    - Position management (on_position)

    Strategy Logic:
    - Buy when RSI < 30 (oversold) AND fast EMA > slow EMA
    - Sell when RSI > 70 (overbought) AND fast EMA < slow EMA
    - Uses 2% stop loss and 6% take profit
    """

    def __init__(self, strategy_orm: Strategy):
        """
        PRE-LOOP Step 2: Child strategy initialization.

        Set up strategy-specific variables and configuration.
        This runs once before the main loop starts.
        """
        # Call parent init first
        super().__init__(strategy_orm)

        # Strategy-specific configuration
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.ema_fast_period = 10
        self.ema_slow_period = 20

        # Risk parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        self.trailing_stop_pct = 0.01  # 1% trailing stop

        # Custom state tracking
        self._dataframe = {}  # Store indicator-enriched DataFrames

        self.logger.info(
            "ExampleStrategy initialized",
            rsi_period=self.rsi_period,
            ema_fast=self.ema_fast_period,
            ema_slow=self.ema_slow_period
        )

    def prepare_indicators(self):
        """
        IN-LOOP Step 4: Prepare technical indicators.

        ONLY runs when bar_completed[feed_num] = True.
        Creates custom DataFrame with RSI and EMA indicators.
        """
        for feed_num in self.non_tick_feeds:
            # CRITICAL: Only calculate if new bar completed
            if not self.bar_completed[feed_num]:
                continue  # Skip - no new data

            self.logger.debug(f"Calculating indicators for feed {feed_num}")

            # Get base DataFrame (immutable)
            df = self.dataframe[feed_num].copy()

            # Add RSI indicator
            df.ta.rsi(length=self.rsi_period, append=True)

            # Add EMA indicators
            df.ta.ema(length=self.ema_fast_period, append=True)
            df.ta.ema(length=self.ema_slow_period, append=True)

            # Store enriched DataFrame
            self._dataframe[feed_num] = df

            # Reset flag
            self.bar_completed[feed_num] = False

            self.logger.debug(
                "Indicators calculated",
                feed_num=feed_num,
                df_rows=len(df)
            )

    def generate_signals(self):
        """
        IN-LOOP Step 5: Generate trading signals.

        Analyzes indicators and sets self.signal[feed_num].
        """
        for feed_num in self.non_tick_feeds:
            # Need indicators DataFrame
            if feed_num not in self._dataframe:
                continue

            df = self._dataframe[feed_num]

            # Need enough data for indicators
            if len(df) < max(self.rsi_period, self.ema_slow_period):
                continue

            # Get latest indicator values
            rsi = df[f'RSI_{self.rsi_period}'].iloc[-1]
            ema_fast = df[f'EMA_{self.ema_fast_period}'].iloc[-1]
            ema_slow = df[f'EMA_{self.ema_slow_period}'].iloc[-1]

            # Generate buy signal
            if rsi < self.rsi_oversold and ema_fast > ema_slow:
                self.signal[feed_num] = "buy"
                self.logger.info(
                    "BUY signal generated",
                    feed_num=feed_num,
                    rsi=rsi,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow
                )

            # Generate sell signal
            elif rsi > self.rsi_overbought and ema_fast < ema_slow:
                self.signal[feed_num] = "sell"
                self.logger.info(
                    "SELL signal generated",
                    feed_num=feed_num,
                    rsi=rsi,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow
                )

            # No signal
            else:
                self.signal[feed_num] = None

    async def on_signal(self, feed_num: int):
        """
        IN-LOOP Step 6a: Handle trading signal (when NO position).

        This is called by BaseStrategy when:
        - We have NO open position for this feed
        - AND self.signal[feed_num] is not None

        Decides whether to enter position based on signal.
        """
        signal = self.signal[feed_num]

        self.logger.info(
            "on_signal called",
            feed_num=feed_num,
            signal=signal
        )

        # Only handle buy signals in this example
        if signal != "buy":
            return

        # Calculate position size based on available funds
        available_funds = self.funds[feed_num]
        current_price = self.tick_buffer[feed_num][-1]

        # Use 25% of available funds
        position_size = (available_funds * 0.25) / current_price

        # Set risk parameters
        self.stop_loss[feed_num] = current_price * (1 - self.stop_loss_pct)
        self.take_profit[feed_num] = current_price * (1 + self.take_profit_pct)
        self.trailing_stop[feed_num] = current_price * self.trailing_stop_pct

        self.logger.info(
            "Placing BUY order",
            feed_num=feed_num,
            size=position_size,
            price=current_price,
            stop_loss=self.stop_loss[feed_num],
            take_profit=self.take_profit[feed_num]
        )

        # Submit order (BaseStrategy handles this)
        await self.place_order(feed_num, "buy", position_size)

    async def on_position(self, feed_num: int):
        """
        IN-LOOP Step 6b: Manage existing position.

        This is called by BaseStrategy when:
        - We HAVE an open position for this feed

        Monitors position health and can trigger manual exit.
        BaseStrategy risk_management() already handles stop_loss, take_profit, etc.
        """
        # Check if base risk management triggered exit
        if self.exit_signal[feed_num]:
            self.logger.info(
                "Closing position due to risk management",
                feed_num=feed_num,
                reason=self.exit_reason[feed_num]
            )
            await self.close_position(feed_num, reason=self.exit_reason[feed_num])
            return

        # Custom position management logic
        position = self.position[feed_num]
        current_price = self.tick_buffer[feed_num][-1]
        entry_price = position.entry_price

        # Calculate unrealized PnL percentage
        pnl_pct = (current_price - entry_price) / entry_price

        # Update trailing stop if in profit
        if pnl_pct > 0.03:  # 3% in profit
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > self.stop_loss[feed_num]:
                self.stop_loss[feed_num] = new_stop
                self.logger.info(
                    "Updated trailing stop",
                    feed_num=feed_num,
                    new_stop=new_stop,
                    pnl_pct=pnl_pct * 100
                )

        # Example: Manual exit if RSI becomes extreme while in position
        if feed_num in self._dataframe:
            df = self._dataframe[feed_num]
            rsi = df[f'RSI_{self.rsi_period}'].iloc[-1]

            # Close position if RSI becomes overbought
            if rsi > 80:
                self.logger.info(
                    "Closing position - RSI overbought",
                    feed_num=feed_num,
                    rsi=rsi
                )
                self.exit_signal[feed_num] = True
                self.exit_reason[feed_num] = "rsi_overbought"
                await self.close_position(feed_num, reason="rsi_overbought")


async def main():
    """
    Example of how to use the strategy.

    In real usage, fullon_bot would handle this.
    """
    print("=== Example Strategy Demo ===\n")

    # In real usage, you would load strategy from database
    print("NOTE: This is a demo - in production, fullon_bot loads strategies from database")
    print("\nExample usage:")
    print("  async with DatabaseContext() as db:")
    print("      strategy_orm = await db.strategies.get_by_id(str_id)")
    print("      strategy = ExampleStrategy(strategy_orm)")
    print("      await strategy.init()")
    print("      # Bot would then call on_tick() or on_bar() as data arrives")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
