"""Enhanced main trading bot with robust async event loop."""

import asyncio
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, time as dt_time, timezone
from typing import Dict, List, Optional, Set, Any

import structlog

from src.config import Config, get_config
from src.client import KalshiRestClient
from src.data_manager import DataManager
from src.execution_engine import ExecutionEngine
from src.paper_trading import PaperTradingExchange
from src.performance_tracker import PerformanceTracker, Trade
from src.risk_manager import RiskConfig, RiskManager, CircuitBreaker
from src.strategies.base import Signal, Strategy
from src.strategies.bollinger_scalper import BollingerScalper
from src.market_discovery import MarketDiscovery, CryptoMarket
from src.metrics_server import create_metrics_server, get_metrics_server

logger = structlog.get_logger(__name__)


class TradingBotConfig:
    """Configuration for TradingBot."""
    
    def __init__(
        self,
        markets: Optional[List[str]] = None,  # None = auto-discover
        max_positions: int = 2,
        trading_start_time: Optional[dt_time] = None,  # None = 24/7
        trading_end_time: Optional[dt_time] = None,
        enable_circuit_breaker: bool = True,
        daily_loss_limit: float = 50.0,
        paper_trading: bool = True,
        starting_balance: float = 10000.0,
    ):
        self.markets = markets
        self.max_positions = max_positions
        self.trading_start_time = trading_start_time
        self.trading_end_time = trading_end_time
        self.enable_circuit_breaker = enable_circuit_breaker
        self.daily_loss_limit = daily_loss_limit
        self.paper_trading = paper_trading
        self.starting_balance = starting_balance


class TradingBot:
    """
    Main trading bot with robust async event loop.
    
    Features:
    - Multi-market support with exception isolation
    - Circuit breaker for daily loss limits
    - Graceful shutdown handling
    - Trading hours enforcement
    - Performance tracking
    """
    
    def __init__(
        self,
        config: Config,
        bot_config: Optional[TradingBotConfig] = None,
        strategy: Optional[Strategy] = None,
    ):
        """
        Initialize TradingBot.
        
        Args:
            config: Application configuration
            bot_config: Trading bot specific configuration
            strategy: Trading strategy (default: BollingerScalper)
        """
        self.config = config
        self.bot_config = bot_config or TradingBotConfig()
        self.strategy = strategy or BollingerScalper()
        
        # Components (initialized in start())
        self.exchange: Optional[Any] = None
        self.rest_client: Optional[KalshiRestClient] = None
        self.data_manager: Optional[DataManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.metrics_server: Optional[Any] = None
        
        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._subscribed_markets: Set[str] = set()
        self._error_counts: Dict[str, int] = {}
        self._max_errors_per_market = 5
        
        # Stats
        self._start_time: Optional[datetime] = None
        self._signals_generated = 0
        self._signals_executed = 0
        self._signals_rejected = 0
        
        logger.info("TradingBot initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Initialize and start all components."""
        if self._running:
            return
        
        logger.info("Starting TradingBot...")
        self._start_time = datetime.now(timezone.utc)
        
        # Initialize exchange (paper or real)
        if self.bot_config.paper_trading:
            logger.info("Using paper trading exchange")
            self.exchange = PaperTradingExchange(
                starting_balance=self.bot_config.starting_balance
            )
            # Paper trading doesn't need REST client
            self.rest_client = None
        else:
            logger.info("Using live exchange")
            self.rest_client = KalshiRestClient(self.config)
            await self.rest_client.connect()
            self.exchange = self.rest_client
        
        # Initialize risk manager
        risk_config = RiskConfig(
            max_positions=self.bot_config.max_positions,
            max_daily_loss_pct=self.bot_config.daily_loss_limit / self.bot_config.starting_balance,
        )
        self.risk_manager = RiskManager(risk_config)
        
        # Initialize circuit breaker
        if self.bot_config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                daily_loss_limit_pct=risk_config.max_daily_loss_pct
            )
        
        # Initialize data manager (if using real data)
        if self.rest_client:
            self.data_manager = DataManager(self.config, self.rest_client)
            await self.data_manager.start()
            
            # Setup candle callback
            self.data_manager.on_candle_closed(self._on_candle_closed)
        
        # Initialize execution engine (if using real exchange)
        if self.rest_client:
            self.execution_engine = ExecutionEngine(
                self.rest_client,
                self.risk_manager
            )
            await self.execution_engine.start()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Initialize metrics server
        metrics_port = int(os.environ.get('METRICS_PORT', '8080'))
        self.metrics_server = create_metrics_server(port=metrics_port, trading_bot=self)
        await self.metrics_server.start()
        
        # Discover and setup markets
        await self._setup_markets()
        
        self._running = True
        
        # Start main loop
        self._tasks.append(asyncio.create_task(self._main_loop()))
        self._tasks.append(asyncio.create_task(self._health_check_loop()))
        
        logger.info("TradingBot started successfully")
    
    async def stop(self) -> None:
        """Graceful shutdown."""
        if not self._running:
            return
        
        logger.info("Stopping TradingBot...")
        self._running = False
        self._shutdown_event.set()
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close positions gracefully
        await self._close_all_positions()
        
        # Stop components
        if self.metrics_server:
            await self.metrics_server.stop()
        
        if self.execution_engine:
            await self.execution_engine.stop()
        
        if self.data_manager:
            await self.data_manager.stop()
        
        if self.rest_client:
            await self.rest_client.close()
        
        # Generate final report
        if self.performance_tracker:
            report = self.performance_tracker.generate_report()
            logger.info("Final Performance Report:\n" + report)
        
        logger.info("TradingBot stopped")
    
    async def _setup_markets(self) -> None:
        """Discover and setup markets to trade."""
        if self.bot_config.markets:
            # Use configured markets
            markets_to_trade = self.bot_config.markets
            logger.info(f"Using configured markets: {markets_to_trade}")
        else:
            # Auto-discover markets
            if not self.rest_client:
                logger.warning("Cannot auto-discover markets without REST client")
                return
            
            logger.info("Auto-discovering crypto markets...")
            discovery = MarketDiscovery()
            
            markets_response = await self.rest_client.get_markets(status="open")
            crypto_markets = discovery.filter_crypto_markets(
                markets_response.markets,
                assets=["BTC", "ETH", "SOL"],
                only_15min=True
            )
            
            markets_to_trade = [m.ticker for m in crypto_markets]
            logger.info(f"Discovered {len(markets_to_trade)} markets")
        
        # Subscribe to markets
        for ticker in markets_to_trade[:self.bot_config.max_positions]:
            try:
                await self._subscribe_to_market(ticker)
            except Exception as e:
                logger.error(f"Failed to setup market {ticker}", error=str(e))
    
    async def _subscribe_to_market(self, ticker: str) -> None:
        """Subscribe to a market."""
        if ticker in self._subscribed_markets:
            return
        
        logger.info(f"Subscribing to market: {ticker}")
        
        if self.data_manager:
            # Initialize with backfill
            await self.data_manager.initialize_market(ticker, backfill_periods=100)
            
            # Subscribe to real-time feeds
            await self.data_manager.subscribe_to_market(ticker)
        
        self._subscribed_markets.add(ticker)
        logger.info(f"Subscribed to {ticker}")
    
    async def _on_candle_closed(self, ticker: str, candle) -> None:
        """
        Handle candle close event.
        
        Args:
            ticker: Market ticker
            candle: Completed OHLCV candle
        """
        try:
            # Check if we should trade
            if not self._should_trade():
                return
            
            # Run strategy
            signal = self.strategy.on_candle(ticker, candle)
            
            if signal:
                self._signals_generated += 1
                
                # Record metric
                if self.metrics_server:
                    self.metrics_server.record_signal(ticker)
                
                await self._process_signal(signal)
        
        except Exception as e:
            logger.error(f"Error processing candle for {ticker}", error=str(e))
            
            # Record error metric
            if self.metrics_server:
                self.metrics_server.record_error(type(e).__name__, ticker)
            
            await self._handle_market_error(ticker, e)
    
    async def _process_signal(self, signal: Signal) -> None:
        """
        Process trading signal.
        
        Args:
            signal: Trading signal
        """
        ticker = signal.market_ticker
        
        logger.info(
            "Processing signal",
            ticker=ticker,
            type=signal.type.value,
            entry=signal.entry_price,
            stop=signal.stop_loss,
            target=signal.take_profit
        )
        
        try:
            # Validate with risk manager
            if self.execution_engine:
                positions = self.execution_engine.get_all_positions()
            else:
                positions = {}
            
            account_balance = self.bot_config.starting_balance
            
            is_valid = self.risk_manager.validate_signal(
                signal, positions, account_balance
            )
            
            if not is_valid:
                self._signals_rejected += 1
                logger.info(f"Signal rejected by risk manager: {ticker}")
                return
            
            # Check circuit breaker
            if self.circuit_breaker and self.circuit_breaker.is_triggered:
                logger.warning(
                    "Circuit breaker active, rejecting signal",
                    reason=self.circuit_breaker.trigger_reason
                )
                return
            
            # Execute signal
            if self.execution_engine:
                result = await self.execution_engine.execute_signal(signal)
            elif hasattr(self.exchange, 'create_order'):
                # Direct paper trading execution
                result = await self._execute_paper_order(signal)
            else:
                logger.error("No execution method available")
                return
            
            if result.success:
                self._signals_executed += 1
                logger.info(f"Signal executed: {ticker}")
                
                # Record trade metric
                if self.metrics_server:
                    side = "long" if signal.type.value == "long" else "short"
                    self.metrics_server.record_trade(ticker, side, "success")
                
                # Track in performance tracker
                # (Will update when position closes)
            else:
                logger.error(f"Signal execution failed: {result.error_message}")
                
                # Record failed trade
                if self.metrics_server:
                    side = "long" if signal.type.value == "long" else "short"
                    self.metrics_server.record_trade(ticker, side, "failed")
        
        except Exception as e:
            logger.error(f"Error processing signal for {ticker}", error=str(e))
            await self._handle_market_error(ticker, e)
    
    async def _execute_paper_order(self, signal: Signal):
        """Execute order on paper trading exchange."""
        # Convert signal to order params
        side = "yes" if signal.type.value == "long" else "no"
        price_cents = int(signal.entry_price * 100)
        
        # Calculate position size
        count = self.risk_manager.calculate_position_size(
            self.bot_config.starting_balance,
            0.02,  # 2% risk
            signal.stop_distance
        )
        
        order = await self.exchange.create_order(
            market_ticker=signal.market_ticker,
            side=side,
            price=price_cents,
            count=count
        )
        
        # Create simple result object
        class SimpleResult:
            def __init__(self, success, order_id=None, error=None):
                self.success = success
                self.order_id = order_id
                self.error_message = error
        
        if order:
            return SimpleResult(True, order.order_id)
        else:
            return SimpleResult(False, error="Order creation failed")
    
    def _should_trade(self) -> bool:
        """Check if trading should be active."""
        # Check trading hours
        if self.bot_config.trading_start_time or self.bot_config.trading_end_time:
            now = datetime.now(timezone.utc).time()
            
            if self.bot_config.trading_start_time and now < self.bot_config.trading_start_time:
                return False
            
            if self.bot_config.trading_end_time and now > self.bot_config.trading_end_time:
                return False
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_triggered:
            return False
        
        return True
    
    async def _handle_market_error(self, ticker: str, error: Exception) -> None:
        """Handle errors for specific markets."""
        self._error_counts[ticker] = self._error_counts.get(ticker, 0) + 1
        
        if self._error_counts[ticker] >= self._max_errors_per_market:
            logger.error(
                f"Max errors reached for {ticker}, unsubscribing",
                error_count=self._error_counts[ticker]
            )
            
            # Unsubscribe from problematic market
            if ticker in self._subscribed_markets:
                self._subscribed_markets.discard(ticker)
    
    async def _close_all_positions(self) -> None:
        """Close all open positions gracefully."""
        logger.info("Closing all positions...")
        
        if not self.execution_engine:
            return
        
        positions = self.execution_engine.get_all_positions()
        
        for ticker in list(positions.keys()):
            try:
                logger.info(f"Closing position: {ticker}")
                await self.execution_engine.close_position(ticker)
            except Exception as e:
                logger.error(f"Error closing position {ticker}", error=str(e))
    
    async def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Main loop started")
        
        try:
            while self._running:
                # Check circuit breaker daily limits
                if self.circuit_breaker:
                    daily_pnl = self.risk_manager._daily_pnl if self.risk_manager else 0
                    account_balance = self.bot_config.starting_balance
                    
                    self.circuit_breaker.check_daily_loss(daily_pnl, account_balance)
                
                # Sleep to prevent tight loop
                await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        except Exception as e:
            logger.error("Main loop error", error=str(e))
            raise
    
    async def _health_check_loop(self) -> None:
        """Periodic health check and logging."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                if not self._running:
                    break
                
                # Log status
                uptime = datetime.now(timezone.utc) - self._start_time if self._start_time else timedelta(0)
                
                logger.info(
                    "Health check",
                    uptime_seconds=uptime.total_seconds(),
                    signals_generated=self._signals_generated,
                    signals_executed=self._signals_executed,
                    signals_rejected=self._signals_rejected,
                    subscribed_markets=len(self._subscribed_markets),
                    circuit_breaker=self.circuit_breaker.is_triggered if self.circuit_breaker else False
                )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        uptime = datetime.now(timezone.utc) - self._start_time if self._start_time else timedelta(0)
        
        return {
            'running': self._running,
            'uptime_seconds': uptime.total_seconds(),
            'subscribed_markets': list(self._subscribed_markets),
            'signals_generated': self._signals_generated,
            'signals_executed': self._signals_executed,
            'signals_rejected': self._signals_rejected,
            'circuit_breaker_triggered': self.circuit_breaker.is_triggered if self.circuit_breaker else False,
            'positions': len(self.execution_engine.get_all_positions()) if self.execution_engine else 0,
        }


async def main():
    """Main entry point."""
    from src.utils.logger import configure_logging
    configure_logging(level="INFO", format="json")
    
    # Create configurations
    config = get_config()
    
    bot_config = TradingBotConfig(
        markets=None,  # Auto-discover
        max_positions=2,
        enable_circuit_breaker=True,
        daily_loss_limit=50.0,
        paper_trading=True,  # Start with paper trading
        starting_balance=10000.0,
    )
    
    # Create and run bot
    bot = TradingBot(config, bot_config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        asyncio.create_task(bot.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        async with bot:
            # Wait for shutdown signal
            while bot._running:
                await asyncio.sleep(1)
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
