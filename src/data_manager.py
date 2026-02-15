"""DataManager coordinates all data sources for the trading bot."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd
import structlog

from src.bollinger_bands import BollingerBands
from src.candle_aggregator import CandleAggregator, OHLCV, handle_trade_message
from src.client import KalshiRestClient
from src.config import Config
from src.data.models import Market
from src.market_discovery import CryptoMarket
from src.websocket_client import ChannelType, KalshiWebSocketClient, WebSocketMessage

logger = structlog.get_logger(__name__)


@dataclass
class MarketState:
    """Complete state for a single market."""
    ticker: str
    market_info: Optional[CryptoMarket] = None
    current_candle: Optional[OHLCV] = None
    bollinger_bands: BollingerBands = field(default_factory=lambda: BollingerBands(period=25))
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = False
    
    def update_candle(self, candle: OHLCV) -> None:
        """Update current candle and indicators."""
        self.current_candle = candle
        self.bollinger_bands.update(candle.close)
        self.last_update = datetime.now(timezone.utc)
    
    def get_signal(self) -> Optional[str]:
        """Generate trading signal from current state."""
        if self.current_candle is None:
            return None
        
        return self.bollinger_bands.get_signal(
            self.current_candle.open,
            self.current_candle.close
        )
    
    def get_bollinger_values(self) -> Optional[Dict[str, float]]:
        """Get current Bollinger Bands values."""
        return self.bollinger_bands.get_last_values()


class DataManager:
    """
    Coordinates all data sources: REST, WebSocket, indicators, and storage.
    
    Provides event-driven architecture for real-time trading.
    """
    
    def __init__(
        self,
        config: Config,
        rest_client: Optional[KalshiRestClient] = None,
        ws_client: Optional[KalshiWebSocketClient] = None,
        candle_aggregator: Optional[CandleAggregator] = None
    ):
        """
        Initialize DataManager.
        
        Args:
            config: Application configuration
            rest_client: Optional REST client (created if not provided)
            ws_client: Optional WebSocket client (created if not provided)
            candle_aggregator: Optional candle aggregator (created if not provided)
        """
        self.config = config
        
        # Clients
        self.rest_client = rest_client or KalshiRestClient(config)
        self.ws_client = ws_client or KalshiWebSocketClient(config)
        self.candle_aggregator = candle_aggregator or CandleAggregator()
        
        # Market state
        self._markets: Dict[str, MarketState] = {}
        self._markets_lock = asyncio.Lock()
        
        # Event callbacks
        self._candle_callbacks: List[Callable[[str, OHLCV], Any]] = []
        self._indicator_callbacks: List[Callable[[str, Dict], Any]] = []
        self._signal_callbacks: List[Callable[[str, str], Any]] = []
        
        # Tracking
        self._subscribed_tickers: Set[str] = set()
        self._is_running = False
        self._tasks: List[asyncio.Task] = []
        
        # Stats
        self._candles_processed = 0
        self._signals_generated = 0
        
        logger.info("DataManager initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start all data connections."""
        if self._is_running:
            return
        
        logger.info("Starting DataManager")
        
        # Connect REST client
        await self.rest_client.connect()
        
        # Connect WebSocket and setup handlers
        await self.ws_client.connect()
        self._setup_websocket_handlers()
        
        # Setup candle aggregator callback
        self.candle_aggregator.on_candle_complete(self._on_candle_completed)
        
        self._is_running = True
        
        logger.info("DataManager started")
    
    async def stop(self) -> None:
        """Stop all data connections."""
        if not self._is_running:
            return
        
        logger.info("Stopping DataManager")
        
        self._is_running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        
        # Flush candles
        await self.candle_aggregator.flush_all_candles()
        
        # Disconnect clients
        await self.ws_client.disconnect()
        await self.rest_client.close()
        
        logger.info("DataManager stopped")
    
    def _setup_websocket_handlers(self) -> None:
        """Setup WebSocket message handlers."""
        
        @self.ws_client.on_trade
        async def handle_trade(msg: WebSocketMessage):
            """Handle trade messages for candle aggregation."""
            await handle_trade_message(self.candle_aggregator, msg.data)
        
        @self.ws_client.on_ticker
        async def handle_ticker(msg: WebSocketMessage):
            """Handle ticker updates."""
            data = msg.data
            ticker = data.get('ticker')
            
            if ticker and ticker in self._markets:
                async with self._markets_lock:
                    market = self._markets[ticker]
                    market.last_update = datetime.now(timezone.utc)
    
    async def _on_candle_completed(self, candle: OHLCV) -> None:
        """
        Handle completed candle from aggregator.
        
        Args:
            candle: Completed OHLCV candle
        """
        ticker = candle.market_ticker
        
        async with self._markets_lock:
            if ticker not in self._markets:
                return
            
            market = self._markets[ticker]
            market.update_candle(candle)
            self._candles_processed += 1
        
        # Emit candle event
        for callback in self._candle_callbacks:
            try:
                result = callback(ticker, candle)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Candle callback error", ticker=ticker, error=str(e))
        
        # Check for signals
        await self._check_signal(ticker)
        
        # Emit indicator event
        await self._emit_indicator_update(ticker)
    
    async def _check_signal(self, ticker: str) -> None:
        """Check for trading signals."""
        async with self._markets_lock:
            if ticker not in self._markets:
                return
            
            market = self._markets[ticker]
            signal = market.get_signal()
        
        if signal:
            self._signals_generated += 1
            
            # Emit signal event
            for callback in self._signal_callbacks:
                try:
                    result = callback(ticker, signal)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("Signal callback error", ticker=ticker, error=str(e))
            
            logger.info(
                "Signal generated",
                ticker=ticker,
                signal=signal,
                total_signals=self._signals_generated
            )
    
    async def _emit_indicator_update(self, ticker: str) -> None:
        """Emit indicator update event."""
        async with self._markets_lock:
            if ticker not in self._markets:
                return
            
            market = self._markets[ticker]
            values = market.get_bollinger_values()
        
        if values:
            for callback in self._indicator_callbacks:
                try:
                    result = callback(ticker, values)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("Indicator callback error", ticker=ticker, error=str(e))
    
    async def initialize_market(
        self,
        ticker: str,
        backfill_periods: int = 100
    ) -> MarketState:
        """
        Initialize market with historical backfill.
        
        Args:
            ticker: Market ticker symbol
            backfill_periods: Number of historical candles to load
            
        Returns:
            MarketState for the initialized market
        """
        logger.info("Initializing market", ticker=ticker, backfill_periods=backfill_periods)
        
        async with self._markets_lock:
            if ticker in self._markets:
                return self._markets[ticker]
            
            # Create new market state
            market = MarketState(ticker=ticker)
            self._markets[ticker] = market
        
        try:
            # Load historical candles
            df = self.candle_aggregator.get_historical_candles(ticker, periods=backfill_periods)
            
            if not df.empty:
                logger.info(
                    "Loaded historical candles",
                    ticker=ticker,
                    count=len(df)
                )
                
                # Backfill indicators
                for _, row in df.iterrows():
                    market.bollinger_bands.update(row['close'])
            
            # Get current market info
            try:
                market_data = await self.rest_client.get_market(ticker)
                # Convert to CryptoMarket if needed
                from src.market_discovery import MarketDiscovery
                discovery = MarketDiscovery()
                crypto_market = discovery.market_to_crypto_market(market_data)
                if crypto_market:
                    market.market_info = crypto_market
            except Exception as e:
                logger.warning("Failed to get market info", ticker=ticker, error=str(e))
            
            market.is_active = True
            
            logger.info("Market initialized", ticker=ticker, warmed_up=market.bollinger_bands.is_warmed_up())
            
            return market
            
        except Exception as e:
            logger.error("Failed to initialize market", ticker=ticker, error=str(e))
            async with self._markets_lock:
                if ticker in self._markets:
                    del self._markets[ticker]
            raise
    
    async def subscribe_to_market(self, ticker: str) -> None:
        """
        Subscribe to real-time data for a market.
        
        Args:
            ticker: Market ticker symbol
        """
        if ticker in self._subscribed_tickers:
            return
        
        logger.info("Subscribing to market", ticker=ticker)
        
        try:
            # Subscribe to WebSocket channels
            await self.ws_client.subscribe(ChannelType.TICKER, [ticker])
            await self.ws_client.subscribe(ChannelType.TRADE, [ticker])
            
            self._subscribed_tickers.add(ticker)
            
            logger.info("Subscribed to market", ticker=ticker)
            
        except Exception as e:
            logger.error("Failed to subscribe", ticker=ticker, error=str(e))
            raise
    
    async def unsubscribe_from_market(self, ticker: str) -> None:
        """
        Unsubscribe from real-time data.
        
        Args:
            ticker: Market ticker symbol
        """
        if ticker not in self._subscribed_tickers:
            return
        
        logger.info("Unsubscribing from market", ticker=ticker)
        
        try:
            await self.ws_client.unsubscribe(ChannelType.TICKER, [ticker])
            await self.ws_client.unsubscribe(ChannelType.TRADE, [ticker])
            
            self._subscribed_tickers.discard(ticker)
            
            logger.info("Unsubscribed from market", ticker=ticker)
            
        except Exception as e:
            logger.error("Failed to unsubscribe", ticker=ticker, error=str(e))
    
    def get_latest_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get latest data for a market.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            Dictionary with current candle and indicators
        """
        if ticker not in self._markets:
            return None
        
        market = self._markets[ticker]
        
        return {
            'ticker': ticker,
            'current_candle': market.current_candle.to_dict() if market.current_candle else None,
            'bollinger_bands': market.get_bollinger_values(),
            'signal': market.get_signal(),
            'last_update': market.last_update.isoformat(),
        }
    
    def get_market_state(self, ticker: str) -> Optional[MarketState]:
        """
        Get full market state snapshot.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            MarketState object or None
        """
        return self._markets.get(ticker)
    
    def get_all_markets(self) -> List[str]:
        """Get list of all tracked market tickers."""
        return list(self._markets.keys())
    
    def get_active_markets(self) -> List[str]:
        """Get list of active market tickers."""
        return [
            ticker for ticker, market in self._markets.items()
            if market.is_active
        ]
    
    # === Event Registration ===
    
    def on_candle_closed(self, callback: Callable[[str, OHLCV], Any]) -> Callable:
        """
        Register callback for candle closed events.
        
        Args:
            callback: Function(ticker, candle) called when candle closes
            
        Returns:
            The callback (for use as decorator)
        """
        self._candle_callbacks.append(callback)
        return callback
    
    def on_indicator_updated(self, callback: Callable[[str, Dict], Any]) -> Callable:
        """
        Register callback for indicator update events.
        
        Args:
            callback: Function(ticker, indicator_values) called on updates
            
        Returns:
            The callback (for use as decorator)
        """
        self._indicator_callbacks.append(callback)
        return callback
    
    def on_signal_triggered(self, callback: Callable[[str, str], Any]) -> Callable:
        """
        Register callback for signal triggered events.
        
        Args:
            callback: Function(ticker, signal) called when signal generated
            
        Returns:
            The callback (for use as decorator)
        """
        self._signal_callbacks.append(callback)
        return callback
    
    def remove_callback(self, callback: Callable) -> bool:
        """
        Remove a registered callback.
        
        Args:
            callback: Callback to remove
            
        Returns:
            True if callback was found and removed
        """
        removed = False
        
        if callback in self._candle_callbacks:
            self._candle_callbacks.remove(callback)
            removed = True
        
        if callback in self._indicator_callbacks:
            self._indicator_callbacks.remove(callback)
            removed = True
        
        if callback in self._signal_callbacks:
            self._signal_callbacks.remove(callback)
            removed = True
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DataManager statistics."""
        return {
            'is_running': self._is_running,
            'total_markets': len(self._markets),
            'active_markets': len(self.get_active_markets()),
            'subscribed_markets': len(self._subscribed_tickers),
            'candles_processed': self._candles_processed,
            'signals_generated': self._signals_generated,
            'candle_callbacks': len(self._candle_callbacks),
            'indicator_callbacks': len(self._indicator_callbacks),
            'signal_callbacks': len(self._signal_callbacks),
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        return {
            'rest_connected': self.rest_client.session is not None,
            'ws_connected': self.ws_client.is_connected,
            'markets_tracked': len(self._markets),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
