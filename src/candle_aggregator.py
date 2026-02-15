"""15-minute OHLCV candle aggregation engine."""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import pandas as pd
import structlog

from src.data.storage import DatabaseManager

logger = structlog.get_logger(__name__)


@dataclass
class Tick:
    """Represents a single trade tick."""
    market_ticker: str
    price: float
    size: int
    timestamp: datetime
    side: Optional[str] = None  # 'yes' or 'no'
    
    def __post_init__(self):
        """Ensure timestamp has timezone info."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class OHLCV:
    """Represents a single OHLCV candle."""
    market_ticker: str
    timestamp: datetime  # Candle start time
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    def __post_init__(self):
        """Ensure timestamp has timezone info."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
    
    @property
    def range(self) -> float:
        """Calculate price range (high - low)."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Calculate candle body (close - open)."""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'market_ticker': self.market_ticker,
            'timestamp': int(self.timestamp.timestamp()),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCV':
        """Create from dictionary."""
        return cls(
            market_ticker=data['market_ticker'],
            timestamp=datetime.fromtimestamp(data['timestamp'], tz=timezone.utc),
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
        )


@dataclass
class CandleState:
    """Current state of a building candle."""
    market_ticker: str
    candle_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    tick_count: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update(self, price: float, size: int, timestamp: datetime) -> None:
        """Update candle with new tick."""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.tick_count += 1
        self.last_update = timestamp
    
    def to_ohlcv(self) -> OHLCV:
        """Convert to OHLCV object."""
        return OHLCV(
            market_ticker=self.market_ticker,
            timestamp=self.candle_start,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


class CandleAggregator:
    """Aggregates trade ticks into 15-minute OHLCV candles."""
    
    CANDLE_INTERVAL = 15  # minutes
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize candle aggregator.
        
        Args:
            db_path: Path to SQLite database (default: data/candles.db)
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "candles.db")
        
        self.db_path = db_path
        self._current_candles: Dict[str, CandleState] = {}
        self._completed_candles: List[OHLCV] = []
        self._callbacks: List[Callable[[OHLCV], Any]] = []
        self._lock = asyncio.Lock()
        
        # Time synchronization
        self._time_offset: float = 0.0  # Offset from system time in seconds
        self._last_exchange_time: Optional[datetime] = None
        
        # Initialize database
        self._init_database()
        
        logger.info("CandleAggregator initialized", db_path=self.db_path)
    
    def _init_database(self) -> None:
        """Initialize SQLite database with candles table."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    market_ticker TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (market_ticker, timestamp)
                )
            """)
            
            # Index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_ticker_time 
                ON candles (market_ticker, timestamp)
            """)
            
            conn.commit()
        
        logger.debug("Database initialized", db_path=self.db_path)
    
    def _get_aligned_timestamp(self, dt: datetime) -> datetime:
        """
        Align datetime to 15-minute boundary.
        
        Args:
            dt: Input datetime
            
        Returns:
            Datetime aligned to 00, 15, 30, or 45 minute boundary
        """
        # Round down to nearest 15 minutes
        minute_bucket = (dt.minute // self.CANDLE_INTERVAL) * self.CANDLE_INTERVAL
        
        return dt.replace(
            minute=minute_bucket,
            second=0,
            microsecond=0
        )
    
    def _get_next_candle_start(self, current_start: datetime) -> datetime:
        """Get the start time of the next candle."""
        return current_start + timedelta(minutes=self.CANDLE_INTERVAL)
    
    def _is_new_candle_boundary(self, tick_time: datetime, candle_start: datetime) -> bool:
        """
        Check if tick is in a new candle period.
        
        Args:
            tick_time: Timestamp of the tick
            candle_start: Start time of current candle
            
        Returns:
            True if tick belongs to a new candle period
        """
        tick_aligned = self._get_aligned_timestamp(tick_time)
        candle_aligned = self._get_aligned_timestamp(candle_start)
        
        return tick_aligned > candle_aligned
    
    def update_time_sync(self, exchange_timestamp: datetime) -> None:
        """
        Update time synchronization from exchange timestamp.
        
        Args:
            exchange_timestamp: Timestamp from exchange
        """
        now = datetime.now(timezone.utc)
        offset = (exchange_timestamp - now).total_seconds()
        
        # Use exponential moving average for smooth adjustment
        alpha = 0.1  # Smoothing factor
        self._time_offset = alpha * offset + (1 - alpha) * self._time_offset
        
        self._last_exchange_time = exchange_timestamp
        
        # Log if significant drift detected
        if abs(self._time_offset) > 1.0:
            logger.warning(
                "Clock drift detected",
                offset_seconds=self._time_offset,
                exchange_time=exchange_timestamp.isoformat(),
                local_time=now.isoformat()
            )
    
    def get_synced_time(self) -> datetime:
        """Get current time adjusted for exchange synchronization."""
        return datetime.now(timezone.utc) + timedelta(seconds=self._time_offset)
    
    async def update_tick(
        self,
        market_ticker: str,
        price: float,
        size: int,
        timestamp: Optional[datetime] = None
    ) -> Optional[OHLCV]:
        """
        Update aggregator with a new tick.
        
        Args:
            market_ticker: Market identifier
            price: Trade price
            size: Trade size/volume
            timestamp: Tick timestamp (defaults to synced current time)
            
        Returns:
            Completed OHLCV candle if boundary crossed, None otherwise
        """
        if timestamp is None:
            timestamp = self.get_synced_time()
        
        async with self._lock:
            return await self._update_tick_internal(market_ticker, price, size, timestamp)
    
    async def _update_tick_internal(
        self,
        market_ticker: str,
        price: float,
        size: int,
        timestamp: datetime
    ) -> Optional[OHLCV]:
        """Internal tick update with lock held."""
        completed_candle = None
        
        # Check if we need to start a new candle
        if market_ticker in self._current_candles:
            current = self._current_candles[market_ticker]
            
            if self._is_new_candle_boundary(timestamp, current.candle_start):
                # Complete the current candle
                completed_candle = current.to_ohlcv()
                await self._complete_candle(completed_candle)
                
                # Start new candle
                candle_start = self._get_aligned_timestamp(timestamp)
                self._current_candles[market_ticker] = CandleState(
                    market_ticker=market_ticker,
                    candle_start=candle_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                    tick_count=1,
                    last_update=timestamp
                )
            else:
                # Update existing candle
                current.update(price, size, timestamp)
        else:
            # Start new candle for this market
            candle_start = self._get_aligned_timestamp(timestamp)
            self._current_candles[market_ticker] = CandleState(
                market_ticker=market_ticker,
                candle_start=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size,
                tick_count=1,
                last_update=timestamp
            )
        
        return completed_candle
    
    async def _complete_candle(self, candle: OHLCV) -> None:
        """Handle completed candle."""
        # Persist to database
        await self._persist_candle(candle)
        
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                result = callback(candle)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Callback error", candle=candle.ticker, error=str(e))
        
        logger.debug(
            "Candle completed",
            ticker=candle.market_ticker,
            timestamp=candle.timestamp.isoformat(),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume
        )
    
    async def _persist_candle(self, candle: OHLCV) -> None:
        """Persist candle to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO candles 
                    (market_ticker, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candle.market_ticker,
                        int(candle.timestamp.timestamp()),
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to persist candle", candle=candle.ticker, error=str(e))
            raise
    
    def get_current_candle(self, market_ticker: str) -> Optional[OHLCV]:
        """
        Get the current building candle for a market.
        
        Args:
            market_ticker: Market identifier
            
        Returns:
            Current OHLCV or None if no active candle
        """
        if market_ticker not in self._current_candles:
            return None
        
        return self._current_candles[market_ticker].to_ohlcv()
    
    def get_historical_candles(
        self,
        market_ticker: str,
        periods: int = 25
    ) -> pd.DataFrame:
        """
        Get historical candles from database.
        
        Args:
            market_ticker: Market identifier
            periods: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM candles
                    WHERE market_ticker = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                
                df = pd.read_sql_query(query, conn, params=(market_ticker, periods))
                
                if df.empty:
                    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                
                # Sort by timestamp ascending
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                return df
                
        except Exception as e:
            logger.error("Failed to load historical candles", ticker=market_ticker, error=str(e))
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def on_candle_complete(self, callback: Callable[[OHLCV], Any]) -> Callable:
        """
        Register callback for completed candles.
        
        Args:
            callback: Function to call when candle completes
            
        Returns:
            The callback (for use as decorator)
        """
        self._callbacks.append(callback)
        return callback
    
    def remove_callback(self, callback: Callable[[OHLCV], Any]) -> bool:
        """
        Remove a registered callback.
        
        Args:
            callback: Callback to remove
            
        Returns:
            True if callback was found and removed
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False
    
    async def force_candle_completion(self, market_ticker: str) -> Optional[OHLCV]:
        """
        Force completion of current candle (useful for testing).
        
        Args:
            market_ticker: Market to complete
            
        Returns:
            Completed candle or None
        """
        async with self._lock:
            if market_ticker not in self._current_candles:
                return None
            
            candle = self._current_candles[market_ticker].to_ohlcv()
            await self._complete_candle(candle)
            del self._current_candles[market_ticker]
            
            return candle
    
    async def flush_all_candles(self) -> List[OHLCV]:
        """
        Complete all active candles (useful for shutdown).
        
        Returns:
            List of completed candles
        """
        async with self._lock:
            completed = []
            
            for ticker in list(self._current_candles.keys()):
                candle = self._current_candles[ticker].to_ohlcv()
                await self._complete_candle(candle)
                completed.append(candle)
            
            self._current_candles.clear()
            
            return completed
    
    def get_active_markets(self) -> List[str]:
        """Get list of markets with active candles."""
        return list(self._current_candles.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            'active_markets': len(self._current_candles),
            'registered_callbacks': len(self._callbacks),
            'time_offset_seconds': self._time_offset,
            'last_exchange_time': self._last_exchange_time.isoformat() if self._last_exchange_time else None,
        }


# Convenience function for WebSocket integration
async def handle_trade_message(
    aggregator: CandleAggregator,
    message: Dict[str, Any]
) -> Optional[OHLCV]:
    """
    Handle trade message from WebSocket.
    
    Args:
        aggregator: CandleAggregator instance
        message: Trade message from WebSocket
        
    Returns:
        Completed candle if boundary crossed
    """
    try:
        data = message.get('data', {})
        
        ticker = data.get('ticker')
        price = float(data.get('price', 0))
        size = int(data.get('count', 0))
        
        # Parse timestamp if available
        timestamp_str = data.get('created_time')
        if timestamp_str:
            # Assume ISO format
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = None
        
        if ticker and price > 0 and size > 0:
            return await aggregator.update_tick(ticker, price, size, timestamp)
        
    except Exception as e:
        logger.error("Error handling trade message", error=str(e), message=message)
    
    return None