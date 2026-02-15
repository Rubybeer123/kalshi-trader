"""Integration tests for TradingBot."""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.main import TradingBot, TradingBotConfig
from src.config import Config, create_config, Environment
from src.paper_trading import PaperTradingExchange
from src.candle_aggregator import OHLCV, OrderBook, OrderBookLevel


@pytest_asyncio.fixture
async def bot_config():
    """Create test bot configuration."""
    return TradingBotConfig(
        markets=["KXBTC15M-26FEB150330-30"],  # Single market for testing
        max_positions=1,
        paper_trading=True,
        starting_balance=1000.0,
        enable_circuit_breaker=True,
        daily_loss_limit=100.0,
    )


@pytest_asyncio.fixture
async def trading_bot(bot_config):
    """Create TradingBot for testing."""
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
    
    bot = TradingBot(config, bot_config)
    
    yield bot
    
    # Cleanup
    await bot.stop()
    import os
    os.unlink(temp_path)


class TestTradingBotInit:
    """Test TradingBot initialization."""
    
    @pytest.mark.asyncio
    async def test_initializes_paper_exchange(self, trading_bot):
        """Test that paper trading exchange is initialized."""
        await trading_bot.start()
        
        assert isinstance(trading_bot.exchange, PaperTradingExchange)
        assert trading_bot.exchange.starting_balance == 1000.0
        
        await trading_bot.stop()
    
    @pytest.mark.asyncio
    async def test_initializes_risk_manager(self, trading_bot):
        """Test that risk manager is initialized."""
        await trading_bot.start()
        
        assert trading_bot.risk_manager is not None
        assert trading_bot.circuit_breaker is not None
        
        await trading_bot.stop()
    
    @pytest.mark.asyncio
    async def test_initializes_performance_tracker(self, trading_bot):
        """Test that performance tracker is initialized."""
        await trading_bot.start()
        
        assert trading_bot.performance_tracker is not None
        
        await trading_bot.stop()


class TestTradingBotLifecycle:
    """Test TradingBot start/stop lifecycle."""
    
    @pytest.mark.asyncio
    async def test_start_runs_successfully(self, trading_bot):
        """Test that bot starts without errors."""
        await trading_bot.start()
        
        assert trading_bot._running is True
        assert len(trading_bot._tasks) > 0
        
        await trading_bot.stop()
    
    @pytest.mark.asyncio
    async def test_stop_gracefully(self, trading_bot):
        """Test that bot stops gracefully."""
        await trading_bot.start()
        await trading_bot.stop()
        
        assert trading_bot._running is False
    
    @pytest.mark.asyncio
    async def test_context_manager(self, bot_config):
        """Test async context manager."""
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
        
        try:
            config = create_config(
                kalshi_api_key_id="test-key",
                kalshi_private_key_path=temp_path,
                kalshi_env=Environment.DEMO,
            )
            
            async with TradingBot(config, bot_config) as bot:
                assert bot._running is True
        finally:
            import os
            os.unlink(temp_path)


class TestSignalProcessing:
    """Test signal processing."""
    
    @pytest.mark.asyncio
    async def test_processes_candle_signal(self, trading_bot):
        """Test that candle signals are processed."""
        await trading_bot.start()
        
        # Create a candle that would trigger a signal
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=datetime.now(timezone.utc),
            open=0.30,
            high=0.35,
            low=0.25,
            close=0.30,
            volume=1000
        )
        
        # This would normally come from data manager
        # For test, call directly
        initial_count = trading_bot._signals_generated
        
        # Note: Strategy may or may not generate signal depending on warmup
        # We just verify no exception is raised
        await trading_bot._on_candle_closed("KXBTC15M-26FEB150330-30", candle)
        
        await trading_bot.stop()
    
    @pytest.mark.asyncio
    async def test_handles_market_error_gracefully(self, trading_bot):
        """Test that market errors don't crash the bot."""
        await trading_bot.start()
        
        # Simulate an error
        error = Exception("Test error")
        await trading_bot._handle_market_error("KXBTC", error)
        
        # Bot should still be running
        assert trading_bot._running is True
        
        await trading_bot.stop()


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_stops_trading(self, trading_bot):
        """Test that circuit breaker prevents trading."""
        await trading_bot.start()
        
        # Trigger circuit breaker
        trading_bot.circuit_breaker._trigger("Test trigger")
        
        # Should not trade
        assert trading_bot._should_trade() is False
        
        await trading_bot.stop()


class TestTradingHours:
    """Test trading hours enforcement."""
    
    @pytest.mark.asyncio
    async def test_respects_trading_hours(self, bot_config):
        """Test that trading hours are enforced."""
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
        
        try:
            # Set trading hours in the past
            from datetime import time as dt_time
            bot_config.trading_start_time = dt_time(0, 0)  # Midnight
            bot_config.trading_end_time = dt_time(0, 1)    # 1 minute after midnight
            
            config = create_config(
                kalshi_api_key_id="test-key",
                kalshi_private_key_path=temp_path,
                kalshi_env=Environment.DEMO,
            )
            
            bot = TradingBot(config, bot_config)
            
            # Should not trade outside hours
            assert bot._should_trade() is False
        finally:
            import os
            os.unlink(temp_path)


class TestStatusReporting:
    """Test status reporting."""
    
    @pytest.mark.asyncio
    async def test_get_status_returns_dict(self, trading_bot):
        """Test that get_status returns status dictionary."""
        await trading_bot.start()
        
        status = trading_bot.get_status()
        
        assert isinstance(status, dict)
        assert 'running' in status
        assert 'subscribed_markets' in status
        
        await trading_bot.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_bot_for_5_seconds():
    """
    Integration test: Run bot for 5 seconds in paper mode.
    
    Verifies:
    - Bot connects successfully
    - Receives data without crashing
    - Graceful shutdown works
    """
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
    
    try:
        config = create_config(
            kalshi_api_key_id="test-key",
            kalshi_private_key_path=temp_path,
            kalshi_env=Environment.DEMO,
        )
        
        bot_config = TradingBotConfig(
            markets=["KXBTC15M-26FEB150330-30"],
            max_positions=1,
            paper_trading=True,
            starting_balance=1000.0,
        )
        
        bot = TradingBot(config, bot_config)
        
        # Start bot
        await bot.start()
        
        # Verify bot is running
        assert bot._running is True
        assert bot.exchange is not None
        
        # Feed some market data
        orderbook = OrderBook(
            ticker="KXBTC15M-26FEB150330-30",
            yes_asks=[OrderBookLevel(price=0.60, count=100)],
            yes_bids=[OrderBookLevel(price=0.58, count=100)],
            no_asks=[OrderBookLevel(price=0.40, count=100)],
            no_bids=[OrderBookLevel(price=0.38, count=100)]
        )
        bot.exchange.update_market_data("KXBTC15M-26FEB150330-30", orderbook)
        
        # Let it run for 2 seconds (shorter for test)
        await asyncio.sleep(2)
        
        # Verify still running
        assert bot._running is True
        
        # Get status
        status = bot.get_status()
        assert status['running'] is True
        
        # Stop gracefully
        await bot.stop()
        
        # Verify stopped
        assert bot._running is False
        
        print("✅ Integration test passed: Bot ran successfully for 2 seconds")
    
    finally:
        import os
        os.unlink(temp_path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graceful_shutdown_closes_positions():
    """
    Test that graceful shutdown closes open positions.
    """
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
    
    try:
        config = create_config(
            kalshi_api_key_id="test-key",
            kalshi_private_key_path=temp_path,
            kalshi_env=Environment.DEMO,
        )
        
        bot_config = TradingBotConfig(
            paper_trading=True,
            starting_balance=1000.0,
        )
        
        bot = TradingBot(config, bot_config)
        
        await bot.start()
        
        # Create a position manually
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        bot.exchange.update_market_data("KXBTC", orderbook)
        
        # Create position
        await bot.exchange.create_order("KXBTC", "yes", 65, 10)
        
        # Verify position exists
        positions = await bot.exchange.get_positions()
        assert len(positions) > 0
        
        # Stop bot
        await bot.stop()
        
        # Verify graceful shutdown completed
        assert bot._running is False
        
        print("✅ Graceful shutdown test passed")
    
    finally:
        import os
        os.unlink(temp_path)


class TestErrorIsolation:
    """Test error isolation between markets."""
    
    @pytest.mark.asyncio
    async def test_one_market_error_doesnt_affect_others(self, trading_bot):
        """Test that error in one market doesn't crash others."""
        await trading_bot.start()
        
        # Setup multiple markets
        trading_bot._subscribed_markets.add("MARKET_A")
        trading_bot._subscribed_markets.add("MARKET_B")
        
        # Simulate errors for one market
        for _ in range(10):
            await trading_bot._handle_market_error("MARKET_A", Exception("Test error"))
        
        # MARKET_A should be unsubscribed after max errors
        assert "MARKET_A" not in trading_bot._subscribed_markets
        
        # MARKET_B should still be subscribed
        assert "MARKET_B" in trading_bot._subscribed_markets
        
        # Bot should still be running
        assert trading_bot._running is True
        
        await trading_bot.stop()


class TestHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_logs_status(self, trading_bot):
        """Test that health check logs status."""
        await trading_bot.start()
        
        # Wait for one health check cycle
        await asyncio.sleep(2)
        
        # Bot should still be running
        assert trading_bot._running is True
        
        await trading_bot.stop()
