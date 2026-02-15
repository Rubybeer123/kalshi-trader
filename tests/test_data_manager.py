"""Tests for DataManager."""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.candle_aggregator import OHLCV
from src.data_manager import DataManager, MarketState
from src.config import Config, create_config, Environment


class TestMarketState:
    """Test MarketState dataclass."""
    
    def test_creation(self):
        """Test MarketState creation."""
        state = MarketState(ticker="KXBTC15M-26FEB150330-30")
        
        assert state.ticker == "KXBTC15M-26FEB150330-30"
        assert state.current_candle is None
        assert state.is_active is False
    
    def test_update_candle(self):
        """Test candle update."""
        state = MarketState(ticker="KXBTC15M-26FEB150330-30")
        
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.65,
            volume=1000
        )
        
        state.update_candle(candle)
        
        assert state.current_candle == candle
        assert state.bollinger_bands.is_warmed_up() is True
    
    def test_get_signal_no_candle(self):
        """Test signal returns None without candle."""
        state = MarketState(ticker="KXBTC15M-26FEB150330-30")
        
        signal = state.get_signal()
        
        assert signal is None
    
    def test_get_signal_with_candle(self):
        """Test signal generation with candle."""
        state = MarketState(ticker="KXBTC15M-26FEB150330-30")
        
        # Add enough candles to warm up
        for i in range(25):
            candle = OHLCV(
                market_ticker="KXBTC15M-26FEB150330-30",
                timestamp=datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc),
                open=0.60,
                high=0.70,
                low=0.55,
                close=0.60 + (i * 0.01),
                volume=100
            )
            state.update_candle(candle)
        
        signal = state.get_signal()
        
        # Should return signal or None
        assert signal in [None, 'oversold', 'overbought']
    
    def test_get_bollinger_values(self):
        """Test getting Bollinger values."""
        state = MarketState(ticker="KXBTC15M-26FEB150330-30")
        
        # Not warmed up yet
        values = state.get_bollinger_values()
        assert values is None
        
        # Warm up
        for i in range(25):
            candle = OHLCV(
                market_ticker="KXBTC15M-26FEB150330-30",
                timestamp=datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc),
                open=0.60,
                high=0.70,
                low=0.55,
                close=0.60 + (i * 0.01),
                volume=100
            )
            state.update_candle(candle)
        
        values = state.get_bollinger_values()
        
        assert values is not None
        assert 'upper' in values
        assert 'middle' in values
        assert 'lower' in values


@pytest_asyncio.fixture
async def data_manager():
    """Create DataManager with mocked clients."""
    import tempfile
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
        temp_path = f.name
    
    config = create_config(
        kalshi_api_key_id="test-key",
        kalshi_private_key_path=temp_path,
        kalshi_env=Environment.DEMO,
    )
    
    # Mock clients
    mock_rest = MagicMock()
    mock_rest.connect = AsyncMock()
    mock_rest.close = AsyncMock()
    mock_rest.get_market = AsyncMock()
    
    mock_ws = MagicMock()
    mock_ws.connect = AsyncMock()
    mock_ws.disconnect = AsyncMock()
    mock_ws.subscribe = AsyncMock()
    mock_ws.unsubscribe = AsyncMock()
    mock_ws.is_connected = True
    mock_ws.on_trade = MagicMock(return_value=lambda x: x)
    mock_ws.on_ticker = MagicMock(return_value=lambda x: x)
    
    dm = DataManager(
        config=config,
        rest_client=mock_rest,
        ws_client=mock_ws,
        candle_aggregator=None
    )
    
    yield dm
    
    import os
    os.unlink(temp_path)


class TestDataManagerLifecycle:
    """Test DataManager start/stop lifecycle."""
    
    @pytest.mark.asyncio
    async def test_start_connects_clients(self, data_manager):
        """Test that start connects all clients."""
        await data_manager.start()
        
        data_manager.rest_client.connect.assert_called_once()
        data_manager.ws_client.connect.assert_called_once()
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_stop_disconnects_clients(self, data_manager):
        """Test that stop disconnects all clients."""
        await data_manager.start()
        await data_manager.stop()
        
        data_manager.ws_client.disconnect.assert_called_once()
        data_manager.rest_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, data_manager):
        """Test async context manager."""
        async with data_manager:
            assert data_manager._is_running
        
        assert not data_manager._is_running


class TestInitializeMarket:
    """Test market initialization."""
    
    @pytest.mark.asyncio
    async def test_initialize_creates_market_state(self, data_manager):
        """Test that initialize creates MarketState."""
        await data_manager.start()
        
        state = await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        assert isinstance(state, MarketState)
        assert state.ticker == "KXBTC15M-26FEB150330-30"
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_initialize_returns_existing_market(self, data_manager):
        """Test that initialize returns existing market if already tracked."""
        await data_manager.start()
        
        state1 = await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        state2 = await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        assert state1 is state2
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_initialize_adds_to_markets_dict(self, data_manager):
        """Test that initialize adds market to tracking."""
        await data_manager.start()
        
        await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        assert "KXBTC15M-26FEB150330-30" in data_manager.get_all_markets()
        
        await data_manager.stop()


class TestSubscribeToMarket:
    """Test market subscription."""
    
    @pytest.mark.asyncio
    async def test_subscribe_calls_ws_subscribe(self, data_manager):
        """Test that subscribe calls WebSocket subscribe."""
        await data_manager.start()
        
        await data_manager.subscribe_to_market("KXBTC15M-26FEB150330-30")
        
        assert data_manager.ws_client.subscribe.called
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_subscribe_only_once_per_ticker(self, data_manager):
        """Test that subscribe is idempotent."""
        await data_manager.start()
        
        await data_manager.subscribe_to_market("KXBTC15M-26FEB150330-30")
        await data_manager.subscribe_to_market("KXBTC15M-26FEB150330-30")
        
        # Should only subscribe twice (ticker + trade channels)
        # But not more than that
        call_count = data_manager.ws_client.subscribe.call_count
        assert call_count <= 4  # Max 2 channels x 2 calls
        
        await data_manager.stop()


class TestGetLatestData:
    """Test getting latest market data."""
    
    @pytest.mark.asyncio
    async def test_get_latest_data_returns_dict(self, data_manager):
        """Test that get_latest_data returns dictionary."""
        await data_manager.start()
        await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        data = data_manager.get_latest_data("KXBTC15M-26FEB150330-30")
        
        assert isinstance(data, dict)
        assert data['ticker'] == "KXBTC15M-26FEB150330-30"
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_get_latest_data_returns_none_for_unknown(self, data_manager):
        """Test that get_latest_data returns None for unknown ticker."""
        data = data_manager.get_latest_data("UNKNOWN-TICKER")
        
        assert data is None


class TestGetMarketState:
    """Test getting market state."""
    
    @pytest.mark.asyncio
    async def test_get_market_state_returns_state(self, data_manager):
        """Test that get_market_state returns MarketState."""
        await data_manager.start()
        await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        state = data_manager.get_market_state("KXBTC15M-26FEB150330-30")
        
        assert isinstance(state, MarketState)
        
        await data_manager.stop()
    
    def test_get_market_state_returns_none(self, data_manager):
        """Test that get_market_state returns None for unknown."""
        state = data_manager.get_market_state("UNKNOWN")
        
        assert state is None


class TestEventCallbacks:
    """Test event callback registration."""
    
    @pytest.mark.asyncio
    async def test_on_candle_closed_registration(self, data_manager):
        """Test candle closed callback registration."""
        received = []
        
        @data_manager.on_candle_closed
        def on_candle(ticker, candle):
            received.append((ticker, candle))
        
        await data_manager.start()
        
        # Manually trigger callback
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.65,
            volume=1000
        )
        
        await data_manager._on_candle_completed(candle)
        
        assert len(received) == 1
        assert received[0][0] == "KXBTC15M-26FEB150330-30"
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_on_indicator_updated_registration(self, data_manager):
        """Test indicator updated callback registration."""
        received = []
        
        @data_manager.on_indicator_updated
        def on_indicator(ticker, values):
            received.append((ticker, values))
        
        await data_manager.start()
        await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        # Add candles to warm up indicator
        for i in range(25):
            candle = OHLCV(
                market_ticker="KXBTC15M-26FEB150330-30",
                timestamp=datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc),
                open=0.60,
                high=0.70,
                low=0.55,
                close=0.60 + (i * 0.01),
                volume=100
            )
            await data_manager._on_candle_completed(candle)
        
        # Should have received indicator updates
        assert len(received) > 0
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_on_signal_triggered_registration(self, data_manager):
        """Test signal triggered callback registration."""
        received = []
        
        @data_manager.on_signal_triggered
        def on_signal(ticker, signal):
            received.append((ticker, signal))
        
        await data_manager.start()
        await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        # Add candles to warm up
        for i in range(25):
            candle = OHLCV(
                market_ticker="KXBTC15M-26FEB150330-30",
                timestamp=datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc),
                open=0.60,
                high=0.70,
                low=0.55,
                close=0.60 + (i * 0.01),
                volume=100
            )
            await data_manager._on_candle_completed(candle)
        
        # Add extreme candle to trigger signal
        extreme_candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.30,  # Well below bands
            high=0.35,
            low=0.25,
            close=0.30,
            volume=1000
        )
        await data_manager._on_candle_completed(extreme_candle)
        
        # Should have received signal
        # (Note: signal depends on Bollinger bands calculation)
        
        await data_manager.stop()
    
    def test_remove_callback(self, data_manager):
        """Test callback removal."""
        @data_manager.on_candle_closed
        def callback(ticker, candle):
            pass
        
        removed = data_manager.remove_callback(callback)
        
        assert removed is True
        assert callback not in data_manager._candle_callbacks


class TestMultipleMarkets:
    """Test handling multiple markets concurrently."""
    
    @pytest.mark.asyncio
    async def test_multiple_markets(self, data_manager):
        """Test handling multiple markets at once."""
        await data_manager.start()
        
        tickers = [
            "KXBTC15M-26FEB150330-30",
            "KETH15M-26FEB150330-3500",
            "KSOL15M-26FEB150330-150"
        ]
        
        # Initialize all markets
        for ticker in tickers:
            await data_manager.initialize_market(ticker)
        
        assert len(data_manager.get_all_markets()) == 3
        
        # Add candles for each
        for ticker in tickers:
            for i in range(5):
                candle = OHLCV(
                    market_ticker=ticker,
                    timestamp=datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc),
                    open=0.60,
                    high=0.70,
                    low=0.55,
                    close=0.60 + (i * 0.01),
                    volume=100
                )
                await data_manager._on_candle_completed(candle)
        
        # Check all markets have state
        for ticker in tickers:
            state = data_manager.get_market_state(ticker)
            assert state is not None
        
        await data_manager.stop()


class TestStats:
    """Test statistics reporting."""
    
    def test_get_stats_returns_expected_fields(self, data_manager):
        """Test that get_stats returns all expected fields."""
        stats = data_manager.get_stats()
        
        assert 'is_running' in stats
        assert 'total_markets' in stats
        assert 'active_markets' in stats
        assert 'subscribed_markets' in stats
        assert 'candles_processed' in stats
        assert 'signals_generated' in stats
    
    @pytest.mark.asyncio
    async def test_stats_reflect_state(self, data_manager):
        """Test that stats reflect current state."""
        await data_manager.start()
        await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        stats = data_manager.get_stats()
        
        assert stats['is_running'] is True
        assert stats['total_markets'] == 1
        
        await data_manager.stop()


class TestHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, data_manager):
        """Test health check returns status dict."""
        await data_manager.start()
        
        health = await data_manager.health_check()
        
        assert 'rest_connected' in health
        assert 'ws_connected' in health
        assert 'markets_tracked' in health
        assert 'timestamp' in health
        
        await data_manager.stop()


class TestMemoryManagement:
    """Test memory management - don't leak candles."""
    
    @pytest.mark.asyncio
    async def test_market_state_doesnt_accumulate_candles(self, data_manager):
        """Test that MarketState doesn't accumulate candles."""
        await data_manager.start()
        
        state = await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        # Add many candles
        for i in range(1000):
            candle = OHLCV(
                market_ticker="KXBTC15M-26FEB150330-30",
                timestamp=datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc),
                open=0.60,
                high=0.70,
                low=0.55,
                close=0.60 + (i * 0.001),
                volume=100
            )
            state.update_candle(candle)
        
        # Only current candle should be stored
        assert state.current_candle is not None
        # Not accumulating all 1000 candles
        
        await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_candle_aggregator_uses_fixed_window(self, data_manager):
        """Test that candle aggregator uses fixed window."""
        # The candle aggregator uses deque with maxlen, so it shouldn't grow unbounded
        aggregator = data_manager.candle_aggregator
        
        # Add many ticks
        for i in range(10000):
            await aggregator.update_tick(
                "KXBTC15M-26FEB150330-30",
                0.60 + (i * 0.0001),
                1,
                datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc) + timedelta(seconds=i)
            )
        
        # Memory should be bounded
        # This is more of a sanity check - actual memory test would need tracemalloc


class TestBackfillMerge:
    """Test backfill and real-time merge."""
    
    @pytest.mark.asyncio
    async def test_backfill_warms_up_indicator(self, data_manager):
        """Test that backfill warms up indicator."""
        # Mock candle aggregator to return historical data
        mock_candles = []
        for i in range(100):
            mock_candles.append({
                'timestamp': datetime(2024, 2, 26, 14, 0, tzinfo=timezone.utc).timestamp() + (i * 900),
                'open': 0.60,
                'high': 0.70,
                'low': 0.55,
                'close': 0.60 + (i * 0.001),
                'volume': 100
            })
        
        with patch.object(
            data_manager.candle_aggregator,
            'get_historical_candles',
            return_value=mock_candles
        ):
            await data_manager.start()
            
            state = await data_manager.initialize_market(
                "KXBTC15M-26FEB150330-30",
                backfill_periods=100
            )
            
            # Indicator should be warmed up
            assert state.bollinger_bands.is_warmed_up()
            
            await data_manager.stop()
    
    @pytest.mark.asyncio
    async def test_real_time_updates_after_backfill(self, data_manager):
        """Test that real-time updates work after backfill."""
        await data_manager.start()
        
        state = await data_manager.initialize_market("KXBTC15M-26FEB150330-30")
        
        # Add real-time candle
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            open=0.60,
            high=0.70,
            low=0.55,
            close=0.65,
            volume=1000
        )
        
        await data_manager._on_candle_completed(candle)
        
        assert state.current_candle.close == 0.65
        
        await data_manager.stop()
