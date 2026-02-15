"""Tests for candle aggregation engine."""

import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import pytest_asyncio

from src.candle_aggregator import (
    CandleAggregator,
    CandleState,
    OHLCV,
    Tick,
    handle_trade_message,
)


class TestTick:
    """Test Tick dataclass."""
    
    def test_creation(self):
        """Test Tick creation."""
        tick = Tick(
            market_ticker="KXBTC15M-26FEB150330-30",
            price=0.65,
            size=10,
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc)
        )
        
        assert tick.market_ticker == "KXBTC15M-26FEB150330-30"
        assert tick.price == 0.65
        assert tick.size == 10
    
    def test_adds_timezone_if_missing(self):
        """Test that timezone is added if missing."""
        tick = Tick(
            market_ticker="KXBTC15M-26FEB150330-30",
            price=0.65,
            size=10,
            timestamp=datetime(2024, 2, 26, 15, 30)  # No timezone
        )
        
        assert tick.timestamp.tzinfo is not None


class TestOHLCV:
    """Test OHLCV dataclass."""
    
    def test_creation(self):
        """Test OHLCV creation."""
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.65,
            volume=1000
        )
        
        assert candle.market_ticker == "KXBTC15M-26FEB150330-30"
        assert candle.open == 0.60
        assert candle.high == 0.70
        assert candle.low == 0.55
        assert candle.close == 0.65
        assert candle.volume == 1000
    
    def test_range_calculation(self):
        """Test range property."""
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.65,
            volume=1000
        )
        
        assert candle.range == 0.15  # 0.70 - 0.55
    
    def test_body_calculation(self):
        """Test body property."""
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.65,
            volume=1000
        )
        
        assert candle.body == 0.05  # 0.65 - 0.60
    
    def test_is_bullish(self):
        """Test bullish detection."""
        bullish = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.70,  # Close > open
            volume=1000
        )
        
        assert bullish.is_bullish is True
        assert bullish.is_bearish is False
    
    def test_is_bearish(self):
        """Test bearish detection."""
        bearish = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.70,
            high=0.75,
            low=0.55,
            close=0.60,  # Close < open
            volume=1000
        )
        
        assert bearish.is_bearish is True
        assert bearish.is_bullish is False
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.65,
            volume=1000
        )
        
        data = candle.to_dict()
        
        assert data['market_ticker'] == "KXBTC15M-26FEB150330-30"
        assert data['open'] == 0.60
        assert data['timestamp'] == int(datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc).timestamp())
    
    def test_from_dict(self):
        """Test dictionary deserialization."""
        data = {
            'market_ticker': "KXBTC15M-26FEB150330-30",
            'timestamp': int(datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc).timestamp()),
            'open': 0.60,
            'high': 0.70,
            'low': 0.55,
            'close': 0.65,
            'volume': 1000
        }
        
        candle = OHLCV.from_dict(data)
        
        assert candle.market_ticker == "KXBTC15M-26FEB150330-30"
        assert candle.open == 0.60


class TestCandleState:
    """Test CandleState."""
    
    def test_update_changes_high(self):
        """Test that update updates high."""
        state = CandleState(
            market_ticker="KXBTC15M-26FEB150330-30",
            candle_start=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.60,
            low=0.60,
            close=0.60,
            volume=0
        )
        
        state.update(0.70, 10, datetime(2024, 2, 26, 15, 31, tzinfo=timezone.utc))
        
        assert state.high == 0.70
        assert state.close == 0.70
        assert state.volume == 10
    
    def test_update_changes_low(self):
        """Test that update updates low."""
        state = CandleState(
            market_ticker="KXBTC15M-26FEB150330-30",
            candle_start=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.60,
            low=0.60,
            close=0.60,
            volume=0
        )
        
        state.update(0.50, 10, datetime(2024, 2, 26, 15, 31, tzinfo=timezone.utc))
        
        assert state.low == 0.50
        assert state.close == 0.50
    
    def test_update_accumulates_volume(self):
        """Test that volume accumulates."""
        state = CandleState(
            market_ticker="KXBTC15M-26FEB150330-30",
            candle_start=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.60,
            low=0.60,
            close=0.60,
            volume=100
        )
        
        state.update(0.65, 50, datetime(2024, 2, 26, 15, 31, tzinfo=timezone.utc))
        
        assert state.volume == 150


class TestCandleAggregatorInit:
    """Test CandleAggregator initialization."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_creates_database(self, temp_db):
        """Test that database is created."""
        aggregator = CandleAggregator(db_path=temp_db)
        
        assert os.path.exists(temp_db)
        
        # Verify table exists
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candles'"
            )
            assert cursor.fetchone() is not None
    
    def test_uses_default_db_path(self):
        """Test default database path."""
        aggregator = CandleAggregator()
        
        assert 'candles.db' in aggregator.db_path


@pytest_asyncio.fixture
async def aggregator():
    """Create aggregator with temp database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    agg = CandleAggregator(db_path=db_path)
    
    yield agg
    
    # Cleanup
    await agg.flush_all_candles()
    os.unlink(db_path)


class TestCandleAggregation:
    """Test candle aggregation."""
    
    @pytest.mark.asyncio
    async def test_aggregate_100_ticks_into_candle(self, aggregator):
        """Test aggregating 100 ticks into correct candle."""
        base_time = datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc)
        ticker = "KXBTC15M-26FEB150330-30"
        
        # Send 100 ticks
        prices = [0.60 + (i * 0.001) for i in range(100)]  # 0.60 to 0.699
        
        for i, price in enumerate(prices):
            tick_time = base_time + timedelta(seconds=i * 5)  # Every 5 seconds
            await aggregator.update_tick(ticker, price, 10, tick_time)
        
        # Get current candle
        candle = aggregator.get_current_candle(ticker)
        
        assert candle is not None
        assert candle.open == 0.60
        assert candle.high == 0.699  # Last price
        assert candle.low == 0.60  # First price
        assert candle.close == 0.699  # Last price
        assert candle.volume == 1000  # 100 ticks * 10 size
    
    @pytest.mark.asyncio
    async def test_boundary_detection_145959_vs_150000(self, aggregator):
        """Test boundary detection at 15-minute boundary."""
        ticker = "KXBTC15M-26FEB150330-30"
        
        # Tick at 14:59:59 (last tick of 14:45 candle)
        time_145959 = datetime(2024, 2, 26, 14, 59, 59, tzinfo=timezone.utc)
        completed = await aggregator.update_tick(ticker, 0.60, 10, time_145959)
        
        # Should not complete yet
        assert completed is None
        
        # Tick at 15:00:00 (first tick of new candle)
        time_150000 = datetime(2024, 2, 26, 15, 0, 0, tzinfo=timezone.utc)
        completed = await aggregator.update_tick(ticker, 0.65, 10, time_150000)
        
        # Should complete the 14:45 candle
        assert completed is not None
        assert completed.candle_start == datetime(2024, 2, 26, 14, 45, tzinfo=timezone.utc)
        assert completed.close == 0.60
    
    @pytest.mark.asyncio
    async def test_multiple_markets_simultaneously(self, aggregator):
        """Test handling multiple markets at once."""
        base_time = datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc)
        
        markets = [
            "KXBTC15M-26FEB150330-30",
            "KETH15M-26FEB150330-3500",
            "KSOL15M-26FEB150330-150"
        ]
        
        # Send ticks to all markets
        for i, ticker in enumerate(markets):
            await aggregator.update_tick(ticker, 0.50 + (i * 0.10), 100, base_time)
        
        # Verify all markets have candles
        for i, ticker in enumerate(markets):
            candle = aggregator.get_current_candle(ticker)
            assert candle is not None
            assert candle.open == 0.50 + (i * 0.10)
    
    @pytest.mark.asyncio
    async def test_candle_persistence(self, aggregator):
        """Test that candles are persisted to database."""
        ticker = "KXBTC15M-26FEB150330-30"
        
        # Create and complete a candle
        time1 = datetime(2024, 2, 26, 14, 45, tzinfo=timezone.utc)
        await aggregator.update_tick(ticker, 0.60, 10, time1)
        
        time2 = datetime(2024, 2, 26, 15, 0, 0, tzinfo=timezone.utc)
        completed = await aggregator.update_tick(ticker, 0.70, 10, time2)
        
        assert completed is not None
        
        # Wait a bit for async persistence
        await asyncio.sleep(0.1)
        
        # Load from database
        df = aggregator.get_historical_candles(ticker, periods=1)
        
        assert len(df) == 1
        assert df.iloc[0]['open'] == 0.60
        assert df.iloc[0]['close'] == 0.70


class TestCallbackSystem:
    """Test callback registration and triggering."""
    
    @pytest.mark.asyncio
    async def test_on_candle_complete_callback(self, aggregator):
        """Test callback is triggered on candle completion."""
        completed_candles = []
        
        @aggregator.on_candle_complete
        def on_complete(candle):
            completed_candles.append(candle)
        
        # Create and complete a candle
        time1 = datetime(2024, 2, 26, 14, 45, tzinfo=timezone.utc)
        await aggregator.update_tick("KXBTC15M-26FEB150330-30", 0.60, 10, time1)
        
        time2 = datetime(2024, 2, 26, 15, 0, 0, tzinfo=timezone.utc)
        await aggregator.update_tick("KXBTC15M-26FEB150330-30", 0.70, 10, time2)
        
        # Wait for callback
        await asyncio.sleep(0.1)
        
        assert len(completed_candles) == 1
        assert completed_candles[0].close == 0.60
    
    @pytest.mark.asyncio
    async def test_async_callback(self, aggregator):
        """Test async callback support."""
        completed_candles = []
        
        @aggregator.on_candle_complete
        async def on_complete_async(candle):
            await asyncio.sleep(0.01)  # Simulate async work
            completed_candles.append(candle)
        
        # Create and complete a candle
        time1 = datetime(2024, 2, 26, 14, 45, tzinfo=timezone.utc)
        await aggregator.update_tick("KXBTC15M-26FEB150330-30", 0.60, 10, time1)
        
        time2 = datetime(2024, 2, 26, 15, 0, 0, tzinfo=timezone.utc)
        await aggregator.update_tick("KXBTC15M-26FEB150330-30", 0.70, 10, time2)
        
        # Wait for callback
        await asyncio.sleep(0.1)
        
        assert len(completed_candles) == 1
    
    def test_remove_callback(self, aggregator):
        """Test callback removal."""
        def callback(candle):
            pass
        
        aggregator.on_candle_complete(callback)
        assert callback in aggregator._callbacks
        
        removed = aggregator.remove_callback(callback)
        
        assert removed is True
        assert callback not in aggregator._callbacks
    
    def test_remove_nonexistent_callback(self, aggregator):
        """Test removing callback that doesn't exist."""
        def callback(candle):
            pass
        
        removed = aggregator.remove_callback(callback)
        
        assert removed is False


class TestHistoricalCandles:
    """Test historical candle retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_historical_candles_returns_dataframe(self, aggregator):
        """Test that historical candles returns DataFrame."""
        ticker = "KXBTC15M-26FEB150330-30"
        
        # Create multiple candles
        for i in range(5):
            time1 = datetime(2024, 2, 26, 14, 45 + (i * 15), tzinfo=timezone.utc)
            await aggregator.update_tick(ticker, 0.60 + (i * 0.01), 100, time1)
            
            time2 = datetime(2024, 2, 26, 15, 0 + (i * 15), 0, tzinfo=timezone.utc)
            await aggregator.update_tick(ticker, 0.70 + (i * 0.01), 100, time2)
        
        await asyncio.sleep(0.1)
        
        df = aggregator.get_historical_candles(ticker, periods=5)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'timestamp' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    @pytest.mark.asyncio
    async def test_get_historical_candles_empty_result(self, aggregator):
        """Test empty result for unknown ticker."""
        df = aggregator.get_historical_candles("UNKNOWN-TICKER", periods=10)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @pytest.mark.asyncio
    async def test_get_historical_candles_limited_periods(self, aggregator):
        """Test period limiting."""
        ticker = "KXBTC15M-26FEB150330-30"
        
        # Create 10 candles
        for i in range(10):
            time1 = datetime(2024, 2, 26, 14, 45 + (i * 15), tzinfo=timezone.utc)
            await aggregator.update_tick(ticker, 0.60, 100, time1)
            
            time2 = datetime(2024, 2, 26, 15, 0 + (i * 15), 0, tzinfo=timezone.utc)
            await aggregator.update_tick(ticker, 0.70, 100, time2)
        
        await asyncio.sleep(0.1)
        
        df = aggregator.get_historical_candles(ticker, periods=5)
        
        assert len(df) == 5


class TestTimeSync:
    """Test time synchronization."""
    
    def test_time_offset_starts_at_zero(self, aggregator):
        """Test that time offset starts at zero."""
        assert aggregator._time_offset == 0.0
    
    def test_update_time_sync_sets_offset(self, aggregator):
        """Test time offset update."""
        # Exchange time 2 seconds ahead
        exchange_time = datetime.now(timezone.utc) + timedelta(seconds=2)
        
        aggregator.update_time_sync(exchange_time)
        
        # Offset should be approximately +2 seconds
        assert abs(aggregator._time_offset - 2.0) < 0.5
    
    def test_get_synced_time_applies_offset(self, aggregator):
        """Test that synced time applies offset."""
        aggregator._time_offset = 5.0  # 5 seconds ahead
        
        synced = aggregator.get_synced_time()
        now = datetime.now(timezone.utc)
        
        # Synced time should be ~5 seconds ahead
        diff = (synced - now).total_seconds()
        assert abs(diff - 5.0) < 0.1


class TestUtilityMethods:
    """Test utility methods."""
    
    @pytest.mark.asyncio
    async def test_force_candle_completion(self, aggregator):
        """Test forcing candle completion."""
        ticker = "KXBTC15M-26FEB150330-30"
        time1 = datetime(2024, 2, 26, 14, 45, tzinfo=timezone.utc)
        await aggregator.update_tick(ticker, 0.60, 100, time1)
        
        # Force completion
        completed = await aggregator.force_candle_completion(ticker)
        
        assert completed is not None
        assert completed.close == 0.60
        
        # Current candle should be gone
        assert aggregator.get_current_candle(ticker) is None
    
    @pytest.mark.asyncio
    async def test_flush_all_candles(self, aggregator):
        """Test flushing all active candles."""
        tickers = ["TICKER1", "TICKER2", "TICKER3"]
        time1 = datetime(2024, 2, 26, 14, 45, tzinfo=timezone.utc)
        
        for ticker in tickers:
            await aggregator.update_tick(ticker, 0.60, 100, time1)
        
        # Flush all
        completed = await aggregator.flush_all_candles()
        
        assert len(completed) == 3
        assert len(aggregator.get_active_markets()) == 0
    
    @pytest.mark.asyncio
    async def test_get_active_markets(self, aggregator):
        """Test getting active markets list."""
        tickers = ["TICKER1", "TICKER2"]
        time1 = datetime(2024, 2, 26, 14, 45, tzinfo=timezone.utc)
        
        for ticker in tickers:
            await aggregator.update_tick(ticker, 0.60, 100, time1)
        
        active = aggregator.get_active_markets()
        
        assert sorted(active) == sorted(tickers)
    
    def test_get_stats(self, aggregator):
        """Test stats retrieval."""
        stats = aggregator.get_stats()
        
        assert 'active_markets' in stats
        assert 'registered_callbacks' in stats
        assert 'time_offset_seconds' in stats
        assert 'last_exchange_time' in stats


class TestHandleTradeMessage:
    """Test WebSocket message handler."""
    
    @pytest.mark.asyncio
    async def test_handle_trade_message(self, aggregator):
        """Test handling trade message from WebSocket."""
        message = {
            'data': {
                'ticker': 'KXBTC15M-26FEB150330-30',
                'price': 0.65,
                'count': 50,
                'created_time': '2024-02-26T15:30:00Z'
            }
        }
        
        result = await handle_trade_message(aggregator, message)
        
        # Should have created candle (not completed yet)
        candle = aggregator.get_current_candle('KXBTC15M-26FEB150330-30')
        assert candle is not None
        assert candle.close == 0.65
    
    @pytest.mark.asyncio
    async def test_handle_trade_message_missing_data(self, aggregator):
        """Test handling message with missing data."""
        message = {'data': {}}  # Empty data
        
        result = await handle_trade_message(aggregator, message)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_handle_trade_message_invalid_price(self, aggregator):
        """Test handling message with invalid price."""
        message = {
            'data': {
                'ticker': 'KXBTC15M-26FEB150330-30',
                'price': 0,  # Invalid
                'count': 10
            }
        }
        
        result = await handle_trade_message(aggregator, message)
        
        assert result is None


class TestTimestampAlignment:
    """Test timestamp alignment to 15-minute boundaries."""
    
    def test_aligns_to_00(self, aggregator):
        """Test alignment to 00 minutes."""
        dt = datetime(2024, 2, 26, 15, 7, 30, tzinfo=timezone.utc)
        aligned = aggregator._get_aligned_timestamp(dt)
        
        assert aligned.minute == 0
        assert aligned.second == 0
    
    def test_aligns_to_15(self, aggregator):
        """Test alignment to 15 minutes."""
        dt = datetime(2024, 2, 26, 15, 22, 30, tzinfo=timezone.utc)
        aligned = aggregator._get_aligned_timestamp(dt)
        
        assert aligned.minute == 15
    
    def test_aligns_to_30(self, aggregator):
        """Test alignment to 30 minutes."""
        dt = datetime(2024, 2, 26, 15, 37, 30, tzinfo=timezone.utc)
        aligned = aggregator._get_aligned_timestamp(dt)
        
        assert aligned.minute == 30
    
    def test_aligns_to_45(self, aggregator):
        """Test alignment to 45 minutes."""
        dt = datetime(2024, 2, 26, 15, 52, 30, tzinfo=timezone.utc)
        aligned = aggregator._get_aligned_timestamp(dt)
        
        assert aligned.minute == 45
    
    def test_detects_new_candle_boundary(self, aggregator):
        """Test detection of new candle boundary."""
        candle_start = datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc)
        
        # Same period
        tick_time = datetime(2024, 2, 26, 15, 14, 59, tzinfo=timezone.utc)
        assert aggregator._is_new_candle_boundary(tick_time, candle_start) is False
        
        # New period
        tick_time = datetime(2024, 2, 26, 15, 15, 0, tzinfo=timezone.utc)
        assert aggregator._is_new_candle_boundary(tick_time, candle_start) is True
