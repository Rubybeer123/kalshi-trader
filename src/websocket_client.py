"""Kalshi WebSocket client with auto-reconnection and message routing."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from src.auth import KalshiAuthenticator
from src.config import Config

logger = structlog.get_logger(__name__)


class ChannelType(str, Enum):
    """WebSocket channel types."""
    TICKER = "ticker"
    ORDERBOOK_DELTA = "orderbook_delta"
    TRADE = "trade"
    FILL = "fill"
    MARKET_LIFECYCLE = "market_lifecycle_v2"


@dataclass
class Subscription:
    """Active subscription tracking."""
    channel: ChannelType
    market_tickers: List[str] = field(default_factory=list)
    subscribed_at: float = field(default_factory=time.time)


@dataclass
class WebSocketMessage:
    """Parsed WebSocket message."""
    channel: str
    sequence: int
    data: Dict[str, Any]
    raw: str
    timestamp: float = field(default_factory=time.time)


class KalshiWebSocketClient:
    """Async WebSocket client for Kalshi real-time data."""
    
    # WebSocket URLs
    DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    PROD_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    # Reconnection settings
    INITIAL_RECONNECT_DELAY = 1.0  # seconds
    MAX_RECONNECT_DELAY = 60.0     # seconds
    RECONNECT_BACKOFF_FACTOR = 2.0
    
    # Heartbeat settings
    HEARTBEAT_INTERVAL = 30.0      # seconds
    HEARTBEAT_TIMEOUT = 10.0       # seconds to wait for pong
    
    def __init__(self, config: Config):
        """
        Initialize WebSocket client.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.authenticator = KalshiAuthenticator(
            config.kalshi_api_key_id,
            config.kalshi_private_key_path
        )
        
        # Select appropriate URL
        self.ws_url = self.DEMO_WS_URL if config.is_demo else self.PROD_WS_URL
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Heartbeat
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_pong_time: float = 0
        
        # Subscriptions
        self._subscriptions: Dict[ChannelType, Subscription] = {}
        self._subscriptions_lock = asyncio.Lock()
        
        # Message sequence tracking for gap detection
        self._last_sequence: Dict[str, int] = {}
        
        # Callbacks
        self._callbacks: Dict[ChannelType, List[Callable]] = {
            ChannelType.TICKER: [],
            ChannelType.ORDERBOOK_DELTA: [],
            ChannelType.TRADE: [],
            ChannelType.FILL: [],
            ChannelType.MARKET_LIFECYCLE: [],
        }
        
        # Running state
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        
        # Stats
        self._messages_received = 0
        self._messages_sent = 0
        self._reconnect_count = 0
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self.websocket is not None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """
        Connect to WebSocket with authentication.
        """
        async with self._connection_lock:
            if self._connected:
                return
            
            try:
                # Generate authentication headers
                # WebSocket auth uses the same signing as REST
                path = "/trade-api/ws/v2"
                auth_headers = self.authenticator.generate_headers("GET", path)
                
                # Convert to WebSocket header format
                ws_headers = [
                    (k, v) for k, v in auth_headers.items()
                ]
                
                logger.info("Connecting to WebSocket", url=self.ws_url)
                
                self.websocket = await websockets.connect(
                    self.ws_url,
                    additional_headers=ws_headers,
                    ping_interval=None,  # We'll handle heartbeats manually
                    ping_timeout=None,
                )
                
                self._connected = True
                self._running = True
                self._last_pong_time = time.time()
                self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
                
                logger.info("WebSocket connected")
                
                # Start tasks
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Resubscribe to previous channels
                await self._resubscribe_all()
                
            except InvalidStatusCode as e:
                logger.error(
                    "WebSocket connection failed",
                    status_code=e.status_code,
                    error=str(e)
                )
                raise
            except Exception as e:
                logger.error("WebSocket connection error", error=str(e))
                raise
    
    async def disconnect(self) -> None:
        """
        Disconnect from WebSocket and cleanup.
        """
        async with self._connection_lock:
            self._running = False
            self._connected = False
            
            # Cancel tasks
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                self._receive_task = None
            
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None
            
            if self._reconnect_task:
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass
                self._reconnect_task = None
            
            # Close websocket
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception:
                    pass
                self.websocket = None
            
            logger.info("WebSocket disconnected")
    
    async def _receive_loop(self) -> None:
        """
        Main receive loop for WebSocket messages.
        """
        while self._running:
            try:
                if not self.websocket:
                    await asyncio.sleep(0.1)
                    continue
                
                message = await self.websocket.recv()
                self._messages_received += 1
                
                # Handle string messages (JSON)
                if isinstance(message, str):
                    await self._handle_message(message)
                # Handle binary messages
                else:
                    logger.warning("Received binary message", length=len(message))
                    
            except ConnectionClosed as e:
                logger.warning(
                    "WebSocket connection closed",
                    code=e.code,
                    reason=e.reason
                )
                self._connected = False
                
                # Schedule reconnection if still running
                if self._running:
                    self._schedule_reconnect()
                break
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error("Error in receive loop", error=str(e))
                await asyncio.sleep(0.1)
    
    async def _handle_message(self, raw_message: str) -> None:
        """
        Parse and route incoming message.
        
        Args:
            raw_message: Raw JSON message string
        """
        try:
            data = json.loads(raw_message)
            
            # Handle control messages
            msg_type = data.get("type", "")
            
            if msg_type == "pong":
                self._last_pong_time = time.time()
                logger.debug("Received pong")
                return
            
            if msg_type == "heartbeat":
                logger.debug("Received server heartbeat")
                return
            
            if msg_type == "error":
                logger.error("Server error message", error=data.get("message"))
                return
            
            # Handle data messages
            channel = data.get("channel", "")
            sequence = data.get("seq", 0)
            
            # Check for sequence gaps
            if channel in self._last_sequence:
                expected = self._last_sequence[channel] + 1
                if sequence != expected:
                    logger.warning(
                        "Sequence gap detected",
                        channel=channel,
                        expected=expected,
                        received=sequence,
                        gap=sequence - expected
                    )
            
            self._last_sequence[channel] = sequence
            
            # Create message object
            message = WebSocketMessage(
                channel=channel,
                sequence=sequence,
                data=data.get("data", {}),
                raw=raw_message
            )
            
            # Route to appropriate callback
            await self._route_message(message)
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse message", error=str(e), message=raw_message[:200])
        except Exception as e:
            logger.error("Error handling message", error=str(e))
    
    async def _route_message(self, message: WebSocketMessage) -> None:
        """
        Route message to appropriate callbacks.
        
        Args:
            message: Parsed WebSocket message
        """
        # Map channel string to ChannelType
        channel_map = {
            "ticker": ChannelType.TICKER,
            "orderbook_delta": ChannelType.ORDERBOOK_DELTA,
            "trade": ChannelType.TRADE,
            "fill": ChannelType.FILL,
            "market_lifecycle_v2": ChannelType.MARKET_LIFECYCLE,
        }
        
        channel_type = channel_map.get(message.channel)
        
        if not channel_type:
            logger.warning("Unknown message channel", channel=message.channel)
            return
        
        # Call registered callbacks
        callbacks = self._callbacks.get(channel_type, [])
        for callback in callbacks:
            try:
                # Support both sync and async callbacks
                result = callback(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "Callback error",
                    channel=message.channel,
                    error=str(e)
                )
    
    async def _heartbeat_loop(self) -> None:
        """
        Send periodic heartbeat pings.
        """
        while self._running:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                
                if not self.websocket or not self._connected:
                    continue
                
                # Send ping
                ping_msg = json.dumps({"type": "ping"})
                await self.websocket.send(ping_msg)
                self._messages_sent += 1
                
                logger.debug("Sent heartbeat ping")
                
                # Check for pong timeout
                time_since_pong = time.time() - self._last_pong_time
                if time_since_pong > self.HEARTBEAT_INTERVAL + self.HEARTBEAT_TIMEOUT:
                    logger.warning(
                        "Heartbeat timeout - no pong received",
                        seconds_since_pong=time_since_pong
                    )
                    # Force reconnection
                    await self._trigger_reconnect()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error", error=str(e))
                await asyncio.sleep(1)
    
    def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if self._reconnect_task and not self._reconnect_task.done():
            return
        
        self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())
    
    async def _reconnect_with_backoff(self) -> None:
        """
        Attempt reconnection with exponential backoff.
        """
        while self._running and not self._connected:
            try:
                logger.info(
                    "Attempting reconnection",
                    delay=self._reconnect_delay,
                    attempt=self._reconnect_count + 1
                )
                
                await asyncio.sleep(self._reconnect_delay)
                
                # Attempt connection
                await self.connect()
                
                # Success!
                self._reconnect_count += 1
                logger.info("Reconnection successful")
                return
                
            except Exception as e:
                logger.error("Reconnection failed", error=str(e))
                
                # Increase delay with exponential backoff
                self._reconnect_delay = min(
                    self._reconnect_delay * self.RECONNECT_BACKOFF_FACTOR,
                    self.MAX_RECONNECT_DELAY
                )
    
    async def _trigger_reconnect(self) -> None:
        """Force reconnection."""
        self._connected = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        
        self._schedule_reconnect()
    
    async def _resubscribe_all(self) -> None:
        """Resubscribe to all active subscriptions after reconnection."""
        async with self._subscriptions_lock:
            for channel, subscription in self._subscriptions.items():
                try:
                    await self.subscribe(channel, subscription.market_tickers)
                    logger.info(
                        "Resubscribed to channel",
                        channel=channel.value,
                        markets=subscription.market_tickers
                    )
                except Exception as e:
                    logger.error(
                        "Failed to resubscribe",
                        channel=channel.value,
                        error=str(e)
                    )
    
    # === Public API ===
    
    async def subscribe(
        self,
        channel: ChannelType,
        market_tickers: Optional[List[str]] = None
    ) -> None:
        """
        Subscribe to a WebSocket channel.
        
        Args:
            channel: Channel type to subscribe to
            market_tickers: Optional list of specific market tickers
        """
        if not self._connected:
            raise ConnectionError("WebSocket not connected")
        
        # Build subscription message
        subscribe_msg = {
            "type": "subscribe",
            "channel": channel.value,
        }
        
        if market_tickers:
            subscribe_msg["market_tickers"] = market_tickers
        
        # Send subscription
        await self.websocket.send(json.dumps(subscribe_msg))
        self._messages_sent += 1
        
        # Track subscription
        async with self._subscriptions_lock:
            self._subscriptions[channel] = Subscription(
                channel=channel,
                market_tickers=market_tickers or []
            )
        
        logger.info(
            "Subscribed to channel",
            channel=channel.value,
            markets=market_tickers
        )
    
    async def unsubscribe(
        self,
        channel: ChannelType,
        market_tickers: Optional[List[str]] = None
    ) -> None:
        """
        Unsubscribe from a WebSocket channel.
        
        Args:
            channel: Channel type to unsubscribe from
            market_tickers: Optional specific markets to unsubscribe
        """
        if not self._connected:
            return
        
        # Build unsubscribe message
        unsubscribe_msg = {
            "type": "unsubscribe",
            "channel": channel.value,
        }
        
        if market_tickers:
            unsubscribe_msg["market_tickers"] = market_tickers
        
        # Send unsubscription
        await self.websocket.send(json.dumps(unsubscribe_msg))
        self._messages_sent += 1
        
        # Remove from tracking
        async with self._subscriptions_lock:
            if channel in self._subscriptions:
                del self._subscriptions[channel]
        
        logger.info(
            "Unsubscribed from channel",
            channel=channel.value,
            markets=market_tickers
        )
    
    def on_ticker(self, callback: Callable[[WebSocketMessage], Any]) -> Callable:
        """
        Register callback for ticker messages.
        
        Args:
            callback: Function to call on ticker messages
            
        Returns:
            The callback (for use as decorator)
        """
        self._callbacks[ChannelType.TICKER].append(callback)
        return callback
    
    def on_orderbook(self, callback: Callable[[WebSocketMessage], Any]) -> Callable:
        """
        Register callback for orderbook delta messages.
        
        Args:
            callback: Function to call on orderbook messages
            
        Returns:
            The callback (for use as decorator)
        """
        self._callbacks[ChannelType.ORDERBOOK_DELTA].append(callback)
        return callback
    
    def on_trade(self, callback: Callable[[WebSocketMessage], Any]) -> Callable:
        """
        Register callback for trade messages.
        
        Args:
            callback: Function to call on trade messages
            
        Returns:
            The callback (for use as decorator)
        """
        self._callbacks[ChannelType.TRADE].append(callback)
        return callback
    
    def on_fill(self, callback: Callable[[WebSocketMessage], Any]) -> Callable:
        """
        Register callback for fill messages.
        
        Args:
            callback: Function to call on fill messages
            
        Returns:
            The callback (for use as decorator)
        """
        self._callbacks[ChannelType.FILL].append(callback)
        return callback
    
    def on_market_lifecycle(self, callback: Callable[[WebSocketMessage], Any]) -> Callable:
        """
        Register callback for market lifecycle messages.
        
        Args:
            callback: Function to call on lifecycle messages
            
        Returns:
            The callback (for use as decorator)
        """
        self._callbacks[ChannelType.MARKET_LIFECYCLE].append(callback)
        return callback
    
    def remove_callback(
        self,
        channel: ChannelType,
        callback: Callable
    ) -> bool:
        """
        Remove a registered callback.
        
        Args:
            channel: Channel type
            callback: Callback to remove
            
        Returns:
            True if callback was found and removed
        """
        callbacks = self._callbacks.get(channel, [])
        if callback in callbacks:
            callbacks.remove(callback)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected": self._connected,
            "messages_received": self._messages_received,
            "messages_sent": self._messages_sent,
            "reconnect_count": self._reconnect_count,
            "active_subscriptions": len(self._subscriptions),
            "last_pong_time": self._last_pong_time,
            "time_since_last_pong": time.time() - self._last_pong_time,
        }
