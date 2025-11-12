My point of view a strategy must have life cycle.

So we have base_strategy.py -> child_strategy.py

Then we will end up in some sort of on_tick loop, but we need pre loop steps and loop steps, even and post loop nice dead step

So ill say the lifecycle of strategy oughta be:


##### PRE LOOP ####

Step 1:
    base_strategy: 
        __init__
            # we must check if fullon_ohlcv_service, fullon_ticker_service, fullon_account_service are up and running
            # we must make sure the candles are filled to the current time, like last bar is aiight
            # we load strategy parameters. so probably a bot activate the strategy, so it must pass a fullon_orm.bot,
            # from there we can like deduce a lot
            get feeds.

            set strategy variables:

            for each feed: check if feed is type tick or type ohlcv (lots of this is database fullon_orm data)
                set class ticks and ohlcv
                self.tick[feed_num].append(Numeric) = max last 100 items
                self.timestamp[feed_num].append(Arrow) = laste date
                self.take_profit(feed_num)  # for non tick feed
                self.trailing_stop(feed_num) for non tick feed
                self.stop_loss(feed_num)# for non tick feed
                self.maxtime_open(feed_num) # for non tick feed
                self.position(feed_num).position: fullon_orm.position # Vaidated vs exchange, for non tick feed
                self.open_trades: List  # list of buying for this trade
                self.pnl[feed_num] #validated vs exchange
                self.leverage[feed_num] # for non tick feed
                self.params # like RSI, SMA, EMA, should endup with self.params.RSI, self.paramas.RSI, etc
                self.dataframe[feed_num] = # for non tick feed
                self.funds[feed_num] # for non tick feed
                self.last_order[feed_num].append(Fullon_orm.Order) = max last 220 items
                self.is_new_bar[feed_num] # for non tick feed, only when new candle has been formed
                self.close_now[feed_num] # for non tick feed
                self.signal[feed_num] # for non tick feed
                self.dry_run # for non tick feed

step 2:
    child_strategy:



        __init__  [or any other name]

            # se can create some class variables that can be used internally for each strategy. such as X.id, tiktok, shit. 
            # other data sources, etc. Could be empty. nothing compulsory here.
            # Code we want to run only once, before main loop start


step 3:
    
    base_strategy:
        # we start main loop.  Lets call it main_loop.
        # main loop needs to well loop around new ticks, in an efficient manner tick by tick might not be it, perhaps by time.

        so this main loop should call like

            0) validate bars, ticks, etc are sync in time.
            1) update self.dataframe if it needs updating, like if we have a new bar formed
            2) update strategy variables (needs to check tick, etc)
            3) for each feed of not type tick we do risk management #if we have a position we set self.close_now
            4) call child.set_dataframe
            5) call child.set_signal
            hidden) call child.on_tick  # optional, child can skip it
            6.a) call child.on_signal # if we do not have open posiiton
            6.b) call child.on_position # if we have an open position
            7) fullon_cache.process.update_bot()  # or whowever it goes.double check.



##### IN LOOP ######  

step 0:

    base_strategy

        self.validate_sync()


step 1:

    base_strategy

        self.update_dataframe() if new candles have formed, well we need to add them to self.dataframe


step 2:
    
    base_strategy

        self.update_variables()

                some of this will need constant updating

                for each feed: check if feed is type tick or type ohlcv (lots of this is database fullon_orm data)
                    set class ticks and ohlcv
                    self.tick[feed_num].append(Numeric) = max last 100 items
                    self.timestamp[feed_num].append(Arrow) = laste date
                    self.take_profit(feed_num)  # for non tick feed
                    self.trailing_stop(feed_num) for non tick feed
                    self.stop_loss(feed_num)# for non tick feed
                    self.maxtime_open(feed_num) # for non tick feed
                    self.position(feed_num).position: fullon_orm.position # Vaidated vs exchange, for non tick feed
                    self.open_trades: List  # list of buying for this trade
                    self.pnl[feed_num] #validated vs exchange
                    self.leverage[feed_num] # for non tick feed
                    self.params # like RSI, SMA, EMA, should endup with self.params.RSI, self.paramas.RSI, etc
                    self.dataframe[feed_num] = # for non tick feed
                    self.funds[feed_num] # for non tick feed
                    self.last_order[feed_num].append(Fullon_orm.Order) = max last 220 items
                    self.is_new_bar[feed_num] # for non tick feed, only when new candle has been formed
                    self.close_now[feed_num] # for non tick feed
                    self.signal[feed_num] # for non tick feed
                    self.dry_run # for non tick feed


step 3

    base_strategy:


        self.risk_management(feed_num)
            #if we have an open position only.
            if we do, we must make sure stop_loss, take_profit, trailing_stop, maxtime_open triggers have not been triggered.
            if they have then we set self.close_now


step 4


    child_strategy:

        setup_dataframes():
            # so we have  self.dataframe[feed_num] we should consider this unmutable from the child
            # but we can make a nother dataframe
            self._dataframe[feed_num] = self.dataframe+whatever pandas_ta the users does.



step 5
    
    child_strategy

        set_signals()

            here we are supposed to set self.signal[feed_num]. perhaps calling llm, or juz cuz TA made a buy signal.
           

    


        
step 6.a

    child_strategy
        on_signal()

            if any feed has a signal we must evaluate and set self.signal[feed_num]

step 7) update bot status