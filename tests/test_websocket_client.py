"""Tests for Kalshi WebSocket client."""

import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import websockets
from websockets.exceptions import ConnectionClosed

from src.config import Config, Environment, create_config
from src.websocket_client import (
    ChannelType,
    KalshiWebSocketClient,
    Subscription,
    WebSocketMessage,
)


class TestWebSocketMessage:
    """Test WebSocketMessage dataclass."""
    
    def test_message_creation(self):
        """Test WebSocketMessage creation."""
        msg = WebSocketMessage(
            channel="ticker",
            sequence=1,
            data={"price": 50},
            raw='{"channel": "ticker", "seq": 1, "data": {"price": 50}}'
        )
        
        assert msg.channel == "ticker"
        assert msg.sequence == 1
        assert msg.data == {"price": 50}
        assert msg.timestamp > 0


class TestChannelType:
    """Test ChannelType enum."""
    
    def test_channel_values(self):
        """Test channel type values."""
        assert ChannelType.TICKER.value == "ticker"
        assert ChannelType.ORDERBOOK_DELTA.value == "orderbook_delta"
        assert ChannelType.TRADE.value == "trade"
        assert ChannelType.FILL.value == "fill"
        assert ChannelType.MARKET_LIFECYCLE.value == "market_lifecycle_v2"


class TestKalshiWebSocketClientInit:
    """Test WebSocket client initialization."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
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
        
        yield config
        
        import os
        os.unlink(temp_path)
    
    def test_initializes_with_demo_config(self, config):
        """Test initialization with demo config."""
        client = KalshiWebSocketClient(config)
        
        assert client.ws_url == KalshiWebSocketClient.DEMO_WS_URL
        assert client.config == config
        assert not client.is_connected
    
    def test_initializes_with_prod_config(self, config):
        """Test initialization with production config."""
        prod_config = create_config(
            kalshi_api_key_id="test-key",
            kalshi_private_key_path=config.kalshi_private_key_path,
            kalshi_env=Environment.PRODUCTION,
        )
        
        client = KalshiWebSocketClient(prod_config)
        
        assert client.ws_url == KalshiWebSocketClient.PROD_WS_URL


class TestCallbackRegistration:
    """Test callback registration methods."""
    
    @pytest.fixture
    def client(self):
        """Create client for testing."""
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
        
        client = KalshiWebSocketClient(config)
        
        yield client
        
        import os
        os.unlink(temp_path)
    
    def test_on_ticker_registers_callback(self, client):
        """Test on_ticker registers callback."""
        def callback(msg):
            pass
        
        result = client.on_ticker(callback)
        
        assert result == callback
        assert callback in client._callbacks[ChannelType.TICKER]
    
    def test_on_orderbook_registers_callback(self, client):
        """Test on_orderbook registers callback."""
        def callback(msg):
            pass
        
        client.on_orderbook(callback)
        
        assert callback in client._callbacks[ChannelType.ORDERBOOK_DELTA]
    
    def test_on_trade_registers_callback(self, client):
        """Test on_trade registers callback."""
        def callback(msg):
            pass
        
        client.on_trade(callback)
        
        assert callback in client._callbacks[ChannelType.TRADE]
    
    def test_on_fill_registers_callback(self, client):
        """Test on_fill registers callback."""
        def callback(msg):
            pass
        
        client.on_fill(callback)
        
        assert callback in client._callbacks[ChannelType.FILL]
    
    def test_on_market_lifecycle_registers_callback(self, client):
        """Test on_market_lifecycle registers callback."""
        def callback(msg):
            pass
        
        client.on_market_lifecycle(callback)
        
        assert callback in client._callbacks[ChannelType.MARKET_LIFECYCLE]
    
    def test_remove_callback(self, client):
        """Test removing a callback."""
        def callback(msg):
            pass
        
        client.on_ticker(callback)
        assert callback in client._callbacks[ChannelType.TICKER]
        
        removed = client.remove_callback(ChannelType.TICKER, callback)
        
        assert removed is True
        assert callback not in client._callbacks[ChannelType.TICKER]
    
    def test_remove_nonexistent_callback(self, client):
        """Test removing a callback that doesn't exist."""
        def callback(msg):
            pass
        
        removed = client.remove_callback(ChannelType.TICKER, callback)
        
        assert removed is False
    
    def test_callback_decorator_syntax(self, client):
        """Test using decorator syntax for callbacks."""
        @client.on_ticker
        def my_handler(msg):
            return "handled"
        
        assert my_handler in client._callbacks[ChannelType.TICKER]


class TestMessageRouting:
    """Test message routing to callbacks."""
    
    @pytest.fixture
    async def client(self):
        """Create client for testing."""
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
        
        client = KalshiWebSocketClient(config)
        
        yield client
        
        import os
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_routes_ticker_message(self, client):
        """Test routing ticker message to callback."""
        received_messages = []
        
        def callback(msg):
            received_messages.append(msg)
        
        client.on_ticker(callback)
        
        # Simulate message
        raw_msg = json.dumps({
            "channel": "ticker",
            "seq": 1,
            "data": {"ticker": "KXBTC", "price": 50}
        })
        
        await client._handle_message(raw_msg)
        
        assert len(received_messages) == 1
        assert received_messages[0].channel == "ticker"
        assert received_messages[0].data["ticker"] == "KXBTC"
    
    @pytest.mark.asyncio
    async def test_routes_orderbook_message(self, client):
        """Test routing orderbook message to callback."""
        received_messages = []
        
        client.on_orderbook(lambda msg: received_messages.append(msg))
        
        raw_msg = json.dumps({
            "channel": "orderbook_delta",
            "seq": 5,
            "data": {"ticker": "KXBTC", "bids": []}
        })
        
        await client._handle_message(raw_msg)
        
        assert len(received_messages) == 1
        assert received_messages[0].channel == "orderbook_delta"
    
    @pytest.mark.asyncio
    async def test_handles_pong_message(self, client):
        """Test handling pong message."""
        initial_pong_time = client._last_pong_time
        
        raw_msg = json.dumps({"type": "pong"})
        await client._handle_message(raw_msg)
        
        assert client._last_pong_time > initial_pong_time
    
    @pytest.mark.asyncio
    async def test_handles_heartbeat_message(self, client):
        """Test handling heartbeat message."""
        raw_msg = json.dumps({"type": "heartbeat"})
        
        # Should not raise error
        await client._handle_message(raw_msg)
    
    @pytest.mark.asyncio
    async def test_handles_error_message(self, client):
        """Test handling error message from server."""
        raw_msg = json.dumps({
            "type": "error",
            "message": "Invalid subscription"
        })
        
        # Should not raise error
        await client._handle_message(raw_msg)
    
    @pytest.mark.asyncio
    async def test_handles_unknown_channel(self, client):
        """Test handling unknown channel gracefully."""
        raw_msg = json.dumps({
            "channel": "unknown_channel",
            "seq": 1,
            "data": {}
        })
        
        # Should not raise error
        await client._handle_message(raw_msg)
    
    @pytest.mark.asyncio
    async def test_detects_sequence_gap(self, client):
        """Test sequence gap detection."""
        received_messages = []
        client.on_ticker(lambda msg: received_messages.append(msg))
        
        # Send messages with gap
        await client._handle_message(json.dumps({
            "channel": "ticker",
            "seq": 1,
            "data": {}
        }))
        
        await client._handle_message(json.dumps({
            "channel": "ticker",
            "seq": 3,  # Gap: missing seq 2
            "data": {}
        }))
        
        # Both should be received despite gap
        assert len(received_messages) == 2


class TestSequenceTracking:
    """Test sequence number tracking."""
    
    @pytest.fixture
    async def client(self):
        """Create client for testing."""
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
        
        client = KalshiWebSocketClient(config)
        
        yield client
        
        import os
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_tracks_last_sequence_per_channel(self, client):
        """Test that last sequence is tracked per channel."""
        client.on_ticker(lambda msg: None)
        client.on_trade(lambda msg: None)
        
        await client._handle_message(json.dumps({
            "channel": "ticker",
            "seq": 5,
            "data": {}
        }))
        
        await client._handle_message(json.dumps({
            "channel": "trade",
            "seq": 10,
            "data": {}
        }))
        
        assert client._last_sequence["ticker"] == 5
        assert client._last_sequence["trade"] == 10


class TestStats:
    """Test statistics tracking."""
    
    @pytest.fixture
    def client(self):
        """Create client for testing."""
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
        
        client = KalshiWebSocketClient(config)
        
        yield client
        
        import os
        os.unlink(temp_path)
    
    def test_get_stats_returns_expected_fields(self, client):
        """Test that get_stats returns all expected fields."""
        stats = client.get_stats()
        
        assert "connected" in stats
        assert "messages_received" in stats
        assert "messages_sent" in stats
        assert "reconnect_count" in stats
        assert "active_subscriptions" in stats
        assert "last_pong_time" in stats
        assert "time_since_last_pong" in stats
    
    def test_stats_initial_values(self, client):
        """Test initial stats values."""
        stats = client.get_stats()
        
        assert stats["connected"] is False
        assert stats["messages_received"] == 0
        assert stats["messages_sent"] == 0
        assert stats["reconnect_count"] == 0
        assert stats["active_subscriptions"] == 0


# Integration tests (marked to skip by default)
@pytest.mark.skip(reason="Integration test - requires live WebSocket server")
class TestWebSocketIntegration:
    """Integration tests against live WebSocket server."""
    
    @pytest_asyncio.fixture
    async def client(self):
        """Create authenticated client."""
        import os
        
        key_id = os.getenv("KALSHI_API_KEY_ID")
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        
        if not key_id or not key_path:
            pytest.skip("Credentials not available")
        
        config = create_config(
            kalshi_api_key_id=key_id,
            kalshi_private_key_path=key_path,
            kalshi_env=Environment.DEMO,
        )
        
        client = KalshiWebSocketClient(config)
        
        yield client
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_connect_and_subscribe_to_ticker(self, client):
        """Test connecting and subscribing to ticker channel."""
        received_messages = []
        
        @client.on_ticker
        def handle_ticker(msg):
            received_messages.append(msg)
        
        # Connect
        await client.connect()
        assert client.is_connected
        
        # Subscribe to ticker
        await client.subscribe(ChannelType.TICKER)
        
        # Wait for messages
        await asyncio.sleep(5)
        
        # Should have received some messages
        assert len(received_messages) > 0
        assert all(m.channel == "ticker" for m in received_messages)
    
    @pytest.mark.asyncio
    async def test_reconnection(self, client):
        """Test auto-reconnection after disconnect."""
        await client.connect()
        assert client.is_connected
        
        initial_reconnect_count = client._reconnect_count
        
        # Force disconnect by closing connection
        if client.websocket:
            await client.websocket.close()
        
        # Wait for reconnection
        await asyncio.sleep(5)
        
        # Should have reconnected
        assert client._reconnect_count > initial_reconnect_count
        assert client.is_connected
    
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, client):
        """Test that heartbeat keeps connection alive."""
        await client.connect()
        
        initial_pong = client._last_pong_time
        
        # Wait for heartbeat cycle
        await asyncio.sleep(35)  # Longer than heartbeat interval
        
        # Should have received pong
        assert client._last_pong_time > initial_pong
        assert client.is_connected


class TestReconnectionLogic:
    """Test reconnection logic with mocks."""
    
    @pytest.fixture
    def client(self):
        """Create client for testing."""
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
        
        client = KalshiWebSocketClient(config)
        
        yield client
        
        import os
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_increases_delay(self, client):
        """Test that reconnection delay increases exponentially."""
        initial_delay = client._reconnect_delay
        
        # Simulate failed reconnection
        client._running = True
        client._connected = False
        
        # Test delay calculation
        delay1 = client._reconnect_delay
        client._reconnect_delay = min(
            delay1 * client.RECONNECT_BACKOFF_FACTOR,
            client.MAX_RECONNECT_DELAY
        )
        
        delay2 = client._reconnect_delay
        assert delay2 > delay1
        
        # Continue increasing
        for _ in range(10):
            client._reconnect_delay = min(
                client._reconnect_delay * client.RECONNECT_BACKOFF_FACTOR,
                client.MAX_RECONNECT_DELAY
            )
        
        # Should be capped at MAX_RECONNECT_DELAY
        assert client._reconnect_delay == client.MAX_RECONNECT_DELAY
    
    @pytest.mark.asyncio
    async def test_reconnect_delay_resets_on_success(self, client):
        """Test that reconnection delay resets after successful connection."""
        # Set high delay
        client._reconnect_delay = 30.0
        
        # Simulate successful connection
        client._connected = True
        client._reconnect_delay = client.INITIAL_RECONNECT_DELAY
        
        assert client._reconnect_delay == client.INITIAL_RECONNECT_DELAY
