"""Backtesting engine for strategy validation and parameter optimization."""

import asyncio
import itertools
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import structlog

from src.candle_aggregator import OHLCV
from src.data.models import OrderBook, OrderBookLevel, OrderStatus, Side
from src.performance_tracker import PerformanceMetrics, PerformanceTracker, Trade
from src.paper_trading import PaperTradingExchange, SimulatedPosition
from src.strategies.base import Signal, SignalType, Strategy

logger = structlog.get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_balance: float = 10000.0
    fee_rate: float = 0.07  # Kalshi fee formula component
    enable_slippage: bool = True
    enable_market_impact: bool = True
    slippage_volatility_factor: float = 1.5  # Slippage scales with volatility
    market_impact_threshold: int = 100  # Contracts before impact kicks in
    market_impact_per_lot: float = 0.001  # Price impact per 100 contracts
    partial_fill_probability: float = 0.05
    random_seed: Optional[int] = None


@dataclass
class BacktestTrade:
    """Trade record from backtest."""
    entry_time: datetime
    exit_time: datetime
    market_ticker: str
    side: str  # 'yes' or 'no'
    entry_price: float
    exit_price: float
    contracts: int
    pnl: float
    fees: float
    exit_reason: str
    slippage: float = 0.0
    market_impact: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Configuration
    config: BacktestConfig
    strategy_name: str
    param_set: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Time analysis
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'param_set': self.param_set,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_trade': self.avg_trade,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
        }


class VariableSlippageModel:
    """
    Variable slippage model based on market volatility.
    
    Slippage increases with:
    - Higher volatility (ATR-based)
    - Larger order size
    - Stressed market conditions
    """
    
    def __init__(self, volatility_factor: float = 1.5):
        """
        Initialize slippage model.
        
        Args:
            volatility_factor: Multiplier for volatility-based slippage
        """
        self.volatility_factor = volatility_factor
        self._recent_ranges: List[float] = []
        self._max_history = 20
    
    def update_volatility(self, candle: OHLCV) -> None:
        """Update volatility estimate with new candle."""
        price_range = candle.high - candle.low
        self._recent_ranges.append(price_range)
        
        if len(self._recent_ranges) > self._max_history:
            self._recent_ranges.pop(0)
    
    def get_atr(self) -> float:
        """Calculate average true range from recent candles."""
        if not self._recent_ranges:
            return 0.01  # Default 1 cent
        return np.mean(self._recent_ranges)
    
    def calculate_slippage(
        self,
        intended_price: float,
        side: str,
        is_buy: bool,
        order_size: int,
        stressed: bool = False
    ) -> float:
        """
        Calculate slippage amount.
        
        Args:
            intended_price: Target price
            side: 'yes' or 'no'
            is_buy: True if buying
            order_size: Number of contracts
            stressed: Whether market is in stressed condition
            
        Returns:
            Slippage amount in price units
        """
        atr = self.get_atr()
        
        # Base slippage from volatility
        base_slippage = atr * self.volatility_factor * 0.1
        
        # Size-based slippage (larger orders = more slippage)
        size_factor = 1.0 + (order_size / 500)  # Increases with size
        
        # Stressed condition multiplier
        stress_multiplier = 2.5 if stressed else 1.0
        
        slippage = base_slippage * size_factor * stress_multiplier
        
        # Ensure minimum slippage
        slippage = max(0.01, slippage)
        
        return round(slippage, 2)
    
    def apply_slippage(
        self,
        intended_price: float,
        side: str,
        is_buy: bool,
        order_size: int,
        stressed: bool = False
    ) -> Tuple[float, float]:
        """
        Apply slippage to price.
        
        Returns:
            Tuple of (filled_price, slippage_amount)
        """
        slippage = self.calculate_slippage(
            intended_price, side, is_buy, order_size, stressed
        )
        
        if is_buy:
            filled_price = intended_price + slippage
        else:
            filled_price = intended_price - slippage
        
        # Clamp to valid range
        filled_price = max(0.01, min(0.99, filled_price))
        
        return round(filled_price, 2), slippage


class MarketImpactModel:
    """
    Models market impact from large orders.
    
    Large orders can move the market price against the trader.
    Impact increases with order size relative to typical volume.
    """
    
    def __init__(
        self,
        threshold: int = 100,
        impact_per_lot: float = 0.001
    ):
        """
        Initialize market impact model.
        
        Args:
            threshold: Minimum contracts before impact applies
            impact_per_lot: Price impact per lot (100 contracts)
        """
        self.threshold = threshold
        self.impact_per_lot = impact_per_lot
        self._recent_volumes: List[int] = []
        self._max_history = 20
    
    def update_volume(self, volume: int) -> None:
        """Update volume estimate with new candle."""
        self._recent_volumes.append(volume)
        
        if len(self._recent_volumes) > self._max_history:
            self._recent_volumes.pop(0)
    
    def get_avg_volume(self) -> int:
        """Get average volume from recent candles."""
        if not self._recent_volumes:
            return 100
        return int(np.mean(self._recent_volumes))
    
    def calculate_impact(
        self,
        order_size: int,
        is_buy: bool
    ) -> float:
        """
        Calculate market impact.
        
        Args:
            order_size: Number of contracts
            is_buy: True if buying
            
        Returns:
            Price impact amount
        """
        if order_size < self.threshold:
            return 0.0
        
        avg_volume = self.get_avg_volume()
        
        # Impact scales with order size relative to average volume
        relative_size = order_size / max(avg_volume, 1)
        lots = order_size / 100
        
        impact = lots * self.impact_per_lot * relative_size
        
        return round(impact, 3)
    
    def apply_impact(
        self,
        price: float,
        order_size: int,
        is_buy: bool
    ) -> Tuple[float, float]:
        """
        Apply market impact to price.
        
        Returns:
            Tuple of (impacted_price, impact_amount)
        """
        impact = self.calculate_impact(order_size, is_buy)
        
        if is_buy:
            # Buying pushes price up
            impacted_price = price + impact
        else:
            # Selling pushes price down
            impacted_price = price - impact
        
        # Clamp to valid range
        impacted_price = max(0.01, min(0.99, impacted_price))
        
        return round(impacted_price, 2), impact


class BacktestExchange:
    """
    Simulated exchange for backtesting.
    
    Implements similar interface to PaperTradingExchange but optimized
    for historical data replay.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest exchange.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self._balance = config.initial_balance
        self._available_balance = config.initial_balance
        
        # Position tracking
        self._positions: Dict[str, SimulatedPosition] = {}
        self._orders: List[Dict[str, Any]] = []
        
        # Models
        self.slippage_model = VariableSlippageModel(
            config.slippage_volatility_factor
        )
        self.impact_model = MarketImpactModel(
            config.market_impact_threshold,
            config.market_impact_per_lot
        )
        
        # State
        self._current_candle: Optional[OHLCV] = None
        self._current_time: Optional[datetime] = None
        
        # Stats
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_market_impact = 0.0
        
        if config.random_seed:
            np.random.seed(config.random_seed)
    
    def update_market_state(self, candle: OHLCV) -> None:
        """Update market state with new candle."""
        self._current_candle = candle
        self._current_time = candle.timestamp
        
        # Update models
        self.slippage_model.update_volatility(candle)
        self.impact_model.update_volume(candle.volume)
        
        # Update position P&L
        for pos in self._positions.values():
            current_price = candle.close
            if pos.side == 'yes':
                pos.unrealized_pnl = (current_price - pos.avg_entry_price) * pos.count
            else:
                pos.unrealized_pnl = (pos.avg_entry_price - current_price) * pos.count
    
    def calculate_fees(self, contracts: int, price: float) -> float:
        """Calculate Kalshi trading fees."""
        return self.config.fee_rate * contracts * price * (1 - price)
    
    def execute_signal(
        self,
        signal: Signal,
        contracts: int
    ) -> Optional[BacktestTrade]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal
            contracts: Position size
            
        Returns:
            BacktestTrade if executed, None otherwise
        """
        if self._current_candle is None:
            return None
        
        ticker = signal.market_ticker
        
        # Determine side and entry price
        if signal.type == SignalType.LONG:
            side = 'yes'
            intended_entry = signal.entry_price
            is_buy = True
        elif signal.type == SignalType.SHORT:
            side = 'no'
            intended_entry = signal.entry_price
            is_buy = True
        else:
            return None
        
        # Apply slippage
        slippage = 0.0
        if self.config.enable_slippage:
            # Stressed condition if candle range is > 2x ATR
            atr = self.slippage_model.get_atr()
            candle_range = self._current_candle.high - self._current_candle.low
            stressed = candle_range > (atr * 2)
            
            intended_entry, slippage = self.slippage_model.apply_slippage(
                intended_entry, side, is_buy, contracts, stressed
            )
            self._total_slippage += slippage * contracts
        
        # Apply market impact
        impact = 0.0
        if self.config.enable_market_impact:
            intended_entry, impact = self.impact_model.apply_impact(
                intended_entry, contracts, is_buy
            )
            self._total_market_impact += impact * contracts
        
        # Check balance
        entry_cost = contracts * intended_entry
        fees = self.calculate_fees(contracts, intended_entry)
        total_cost = entry_cost + fees
        
        if total_cost > self._available_balance:
            logger.warning(
                "Insufficient balance for trade",
                ticker=ticker,
                required=total_cost,
                available=self._available_balance
            )
            return None
        
        # Update balance
        self._available_balance -= total_cost
        self._total_fees += fees
        
        # Create/update position
        if ticker in self._positions:
            pos = self._positions[ticker]
            total_cost = (pos.avg_entry_price * pos.count) + (intended_entry * contracts)
            pos.count += contracts
            pos.avg_entry_price = total_cost / pos.count
        else:
            self._positions[ticker] = SimulatedPosition(
                ticker=ticker,
                side=side,
                count=contracts,
                avg_entry_price=intended_entry
            )
        
        logger.debug(
            "Position opened",
            ticker=ticker,
            side=side,
            entry=intended_entry,
            contracts=contracts,
            slippage=slippage,
            impact=impact
        )
        
        # For backtest, we immediately simulate the exit based on stop/target
        # In reality, this would happen on future candles
        return self._simulate_exit(signal, contracts, intended_entry, fees, slippage, impact)
    
    def _simulate_exit(
        self,
        signal: Signal,
        contracts: int,
        entry_price: float,
        entry_fees: float,
        entry_slippage: float,
        entry_impact: float
    ) -> BacktestTrade:
        """
        Simulate position exit.
        
        In backtesting, we assume the exit happens based on the signal's
        stop loss or take profit levels.
        """
        ticker = signal.market_ticker
        
        # Determine exit price based on which level is hit first
        # For simplicity, we use the signal's levels directly
        # In a more sophisticated model, we'd walk forward through candles
        
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        target_distance = abs(signal.take_profit - signal.entry_price)
        
        # Simulate price path - in reality, we'd check future candles
        # For now, assume target is hit if R:R is favorable
        # This is a simplification for the backtest
        
        # Random path simulation based on volatility
        atr = self.slippage_model.get_atr()
        
        # Probability of hitting target vs stop based on distance
        # Closer target = more likely to hit
        if signal.type == SignalType.LONG:
            # For long: check if low hits stop before high hits target
            stop_hit = self._current_candle.low <= signal.stop_loss
            target_hit = self._current_candle.high >= signal.take_profit
        else:
            # For short: check if high hits stop before low hits target
            stop_hit = self._current_candle.high >= signal.stop_loss
            target_hit = self._current_candle.low <= signal.take_profit
        
        # Determine exit
        if target_hit and not stop_hit:
            exit_price = signal.take_profit
            exit_reason = 'target'
        elif stop_hit:
            exit_price = signal.stop_loss
            exit_reason = 'stop'
        else:
            # Neither hit - exit at close
            exit_price = self._current_candle.close
            # Determine if it's closer to target or stop
            if signal.type == SignalType.LONG:
                exit_reason = 'target' if exit_price > signal.entry_price else 'stop'
            else:
                exit_reason = 'target' if exit_price < signal.entry_price else 'stop'
        
        # Apply exit slippage
        exit_slippage = 0.0
        if self.config.enable_slippage:
            exit_slippage = self.slippage_model.calculate_slippage(
                exit_price, 'yes' if signal.type == SignalType.LONG else 'no',
                False, contracts
            )
            if signal.type == SignalType.LONG:
                exit_price -= exit_slippage
            else:
                exit_price += exit_slippage
        
        # Calculate P&L
        if signal.type == SignalType.LONG:
            gross_pnl = (exit_price - entry_price) * contracts
        else:
            gross_pnl = (entry_price - exit_price) * contracts
        
        # Calculate exit fees
        exit_fees = self.calculate_fees(contracts, exit_price)
        self._total_fees += exit_fees
        
        # Net P&L
        net_pnl = gross_pnl - entry_fees - exit_fees
        
        # Update balance with proceeds
        exit_proceeds = contracts * exit_price
        self._available_balance += exit_proceeds - exit_fees
        
        # Remove position
        if ticker in self._positions:
            del self._positions[ticker]
        
        trade = BacktestTrade(
            entry_time=self._current_time,
            exit_time=self._current_time,  # Same candle for simplicity
            market_ticker=ticker,
            side='yes' if signal.type == SignalType.LONG else 'no',
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=contracts,
            pnl=net_pnl,
            fees=entry_fees + exit_fees,
            exit_reason=exit_reason,
            slippage=entry_slippage + exit_slippage,
            market_impact=entry_impact
        )
        
        logger.debug(
            "Trade completed",
            ticker=ticker,
            entry=entry_price,
            exit=exit_price,
            pnl=net_pnl,
            reason=exit_reason
        )
        
        return trade
    
    def get_balance(self) -> float:
        """Get current balance."""
        return self._available_balance
    
    def get_total_equity(self) -> float:
        """Get total equity including positions."""
        position_value = sum(
            pos.count * self._current_candle.close if self._current_candle else pos.avg_entry_price
            for pos in self._positions.values()
        )
        return self._available_balance + position_value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get exchange statistics."""
        return {
            'balance': self._balance,
            'available_balance': self._available_balance,
            'total_fees': self._total_fees,
            'total_slippage': self._total_slippage,
            'total_market_impact': self._total_market_impact,
            'open_positions': len(self._positions),
        }


class Backtester:
    """
    Backtesting engine for trading strategies.
    
    Features:
    - Load historical candles from SQLite or CSV
    - Replay market data chronologically
    - Simulate execution with realistic slippage and fees
    - Calculate performance metrics
    - Parameter optimization via grid search
    """
    
    def __init__(
        self,
        strategy_factory: Callable[..., Strategy],
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.
        
        Args:
            strategy_factory: Factory function that creates strategy instances
            config: Backtest configuration
        """
        self.strategy_factory = strategy_factory
        self.config = config or BacktestConfig()
        
        # Data storage
        self._candles: pd.DataFrame = pd.DataFrame()
        self._market_ticker: Optional[str] = None
        
        # Results
        self._last_result: Optional[BacktestResult] = None
        
        logger.info("Backtester initialized")
    
    def load_data_sqlite(
        self,
        db_path: str,
        market_ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'Backtester':
        """
        Load historical candles from SQLite database.
        
        Args:
            db_path: Path to SQLite database
            market_ticker: Market ticker to load
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Self for chaining
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE market_ticker = ?
        """
        params = [market_ticker]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(int(start_date.timestamp()))
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(int(end_date.timestamp()))
        
        query += " ORDER BY timestamp"
        
        try:
            with sqlite3.connect(db_path) as conn:
                self._candles = pd.read_sql_query(query, conn, params=params)
                
                if self._candles.empty:
                    logger.warning("No candles found", ticker=market_ticker)
                    return self
                
                # Convert timestamp
                self._candles['timestamp'] = pd.to_datetime(
                    self._candles['timestamp'], unit='s', utc=True
                )
                
                self._market_ticker = market_ticker
                
                logger.info(
                    "Loaded candles from SQLite",
                    ticker=market_ticker,
                    count=len(self._candles),
                    start=self._candles['timestamp'].iloc[0],
                    end=self._candles['timestamp'].iloc[-1]
                )
                
        except Exception as e:
            logger.error("Failed to load data from SQLite", error=str(e))
            raise
        
        return self
    
    def load_data_csv(
        self,
        csv_path: str,
        market_ticker: Optional[str] = None,
        timestamp_col: str = 'timestamp',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'Backtester':
        """
        Load historical candles from CSV file.
        
        Args:
            csv_path: Path to CSV file
            market_ticker: Market ticker (if not in CSV)
            timestamp_col: Name of timestamp column
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Self for chaining
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Parse timestamp
            if timestamp_col in df.columns:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], utc=True)
            
            # Filter by date if specified
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            self._candles = df
            self._market_ticker = market_ticker or df.get('market_ticker', ['unknown']).iloc[0]
            
            logger.info(
                "Loaded candles from CSV",
                ticker=self._market_ticker,
                count=len(self._candles),
                start=self._candles['timestamp'].iloc[0],
                end=self._candles['timestamp'].iloc[-1]
            )
            
        except Exception as e:
            logger.error("Failed to load data from CSV", error=str(e))
            raise
        
        return self
    
    def run(
        self,
        strategy_params: Optional[Dict[str, Any]] = None,
        position_sizer: Optional[Callable[[Signal, float], int]] = None
    ) -> BacktestResult:
        """
        Run backtest with loaded data.
        
        Args:
            strategy_params: Optional parameters to pass to strategy factory
            position_sizer: Optional function(signal, balance) -> contracts
            
        Returns:
            BacktestResult with performance metrics
        """
        if self._candles.empty:
            raise ValueError("No data loaded. Call load_data_sqlite() or load_data_csv() first.")
        
        # Create strategy instance
        params = strategy_params or {}
        strategy = self.strategy_factory(**params)
        
        # Create exchange
        exchange = BacktestExchange(self.config)
        
        # Initialize result
        result = BacktestResult(
            config=self.config,
            strategy_name=strategy.name,
            param_set=params,
            start_time=self._candles['timestamp'].iloc[0],
            end_time=self._candles['timestamp'].iloc[-1]
        )
        
        # Warmup period
        warmup_periods = strategy.get_required_warmup()
        
        logger.info(
            "Starting backtest",
            strategy=strategy.name,
            candles=len(self._candles),
            warmup=warmup_periods
        )
        
        # Replay candles
        equity_curve = []
        trades = []
        
        for idx, row in self._candles.iterrows():
            # Create OHLCV candle
            candle = OHLCV(
                market_ticker=self._market_ticker or 'unknown',
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume'])
            )
            
            # Update exchange state
            exchange.update_market_state(candle)
            
            # Get signal from strategy
            signal = strategy.on_candle(self._market_ticker or 'unknown', candle)
            
            if signal:
                # Calculate position size
                if position_sizer:
                    contracts = position_sizer(signal, exchange.get_balance())
                else:
                    # Default: risk 2% per trade
                    contracts = self._default_position_sizer(signal, exchange.get_balance())
                
                if contracts > 0:
                    trade = exchange.execute_signal(signal, contracts)
                    if trade:
                        trades.append(trade)
            
            # Record equity
            equity_curve.append((candle.timestamp, exchange.get_total_equity()))
        
        # Populate result
        result.trades = trades
        result.equity_curve = equity_curve
        
        # Calculate metrics
        self._calculate_metrics(result)
        
        self._last_result = result
        
        logger.info(
            "Backtest complete",
            trades=len(trades),
            return_pct=result.total_return_pct,
            sharpe=result.sharpe_ratio,
            max_dd=result.max_drawdown_pct
        )
        
        return result
    
    def _default_position_sizer(self, signal: Signal, balance: float) -> int:
        """Default position sizing: risk 2% per trade."""
        risk_pct = 0.02
        risk_amount = balance * risk_pct
        
        stop_distance = signal.stop_distance
        if stop_distance == 0:
            return 0
        
        # Number of contracts = risk amount / stop distance
        contracts = int(risk_amount / stop_distance)
        
        # Limit max position
        max_contracts = int(balance * 0.5 / signal.entry_price)
        return min(contracts, max_contracts)
    
    def _calculate_metrics(self, result: BacktestResult) -> None:
        """Calculate all performance metrics."""
        trades = result.trades
        
        if not trades:
            return
        
        # Basic counts
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in trades if t.pnl < 0)
        
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades
        
        # P&L statistics
        pnls = [t.pnl for t in trades]
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        result.total_return = sum(pnls)
        result.avg_trade = np.mean(pnls) if pnls else 0
        
        if wins:
            result.avg_win = np.mean(wins)
            result.largest_win = max(wins)
        
        if losses:
            result.avg_loss = np.mean(losses)
            result.largest_loss = min(losses)
        
        # Return percentage
        result.total_return_pct = result.total_return / self.config.initial_balance
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            result.profit_factor = float('inf')
        
        # Expectancy
        loss_rate = 1 - result.win_rate
        result.expectancy = (result.win_rate * result.avg_win) - (loss_rate * abs(result.avg_loss))
        
        # Consecutive trades
        max_wins = 0
        max_losses = 0
        current_streak = 0
        current_type = None
        
        for trade in trades:
            if trade.pnl > 0:
                if current_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'win'
                max_wins = max(max_wins, current_streak)
            else:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'loss'
                max_losses = max(max_losses, current_streak)
        
        result.max_consecutive_wins = max_wins
        result.max_consecutive_losses = max_losses
        
        # Calculate from equity curve
        if result.equity_curve:
            equity_values = [e[1] for e in result.equity_curve]
            result.sharpe_ratio = self._calculate_sharpe(equity_values)
            result.max_drawdown, result.max_drawdown_pct = self._calculate_drawdown(equity_values)
        
        # Duration
        if result.start_time and result.end_time:
            result.total_duration = result.end_time - result.start_time
        else:
            result.total_duration = timedelta(0)
    
    def _calculate_sharpe(self, equity: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(equity) < 2:
            return 0.0
        
        returns = np.diff(equity) / equity[:-1]
        
        if returns.std() == 0:
            return 0.0
        
        # Annualize (assuming 15-min candles, 96 per day, ~250 trading days)
        periods_per_year = 96 * 250
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        return sharpe
    
    def _calculate_drawdown(self, equity: List[float]) -> Tuple[float, float]:
        """Calculate maximum drawdown."""
        if not equity:
            return 0.0, 0.0
        
        equity_array = np.array(equity)
        peak = np.maximum.accumulate(equity_array)
        drawdown = equity_array - peak
        drawdown_pct = drawdown / peak
        
        max_dd = drawdown.min()
        max_dd_pct = abs(drawdown_pct.min())
        
        return max_dd, max_dd_pct
    
    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        position_sizer: Optional[Callable[[Signal, float], int]] = None
    ) -> Tuple[BacktestResult, Dict[str, Any]]:
        """
        Run grid search optimization over parameter space.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values
            metric: Metric to optimize ('sharpe_ratio', 'total_return_pct', 'profit_factor', etc.')
            position_sizer: Optional position sizing function
            
        Returns:
            Tuple of (best_result, best_params)
        """
        if self._candles.empty:
            raise ValueError("No data loaded. Call load_data first.")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = np.prod([len(v) for v in param_values])
        
        logger.info(
            "Starting parameter optimization",
            combinations=total_combinations,
            metric=metric
        )
        
        results = []
        
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                result = self.run(params, position_sizer)
                results.append((result, params))
                
                logger.debug(
                    "Tested parameters",
                    params=params,
                    metric_value=getattr(result, metric, 0)
                )
                
            except Exception as e:
                logger.warning("Parameter set failed", params=params, error=str(e))
                continue
        
        if not results:
            raise ValueError("No valid results from optimization")
        
        # Find best result
        # Handle cases where metric might be negative (lower is better for drawdown)
        if metric in ['max_drawdown', 'max_drawdown_pct']:
            best_result, best_params = min(results, key=lambda x: getattr(x[0], metric, float('inf')))
        else:
            best_result, best_params = max(results, key=lambda x: getattr(x[0], metric, float('-inf')))
        
        logger.info(
            "Optimization complete",
            best_metric=getattr(best_result, metric),
            best_params=best_params,
            total_tested=len(results)
        )
        
        return best_result, best_params
    
    def get_last_result(self) -> Optional[BacktestResult]:
        """Get the result from the last backtest run."""
        return self._last_result
    
    def generate_report(self, result: Optional[BacktestResult] = None) -> str:
        """Generate text report of backtest results."""
        result = result or self._last_result
        
        if not result:
            return "No backtest results available."
        
        lines = [
            "=" * 70,
            "                    BACKTEST REPORT",
            "=" * 70,
            "",
            f"Strategy: {result.strategy_name}",
            f"Parameters: {result.param_set}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 40,
            f"Total Return:        ${result.total_return:,.2f} ({result.total_return_pct:.1%})",
            f"Sharpe Ratio:        {result.sharpe_ratio:.2f}",
            f"Max Drawdown:        ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1%})",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades:        {result.total_trades}",
            f"Win Rate:            {result.win_rate:.1%}",
            f"Profit Factor:       {result.profit_factor:.2f}",
            f"Expectancy:          ${result.expectancy:.2f}",
            "",
            "P&L BREAKDOWN",
            "-" * 40,
            f"Average Trade:       ${result.avg_trade:.2f}",
            f"Average Win:         ${result.avg_win:.2f}",
            f"Average Loss:        ${result.avg_loss:.2f}",
            f"Largest Win:         ${result.largest_win:.2f}",
            f"Largest Loss:        ${result.largest_loss:.2f}",
            "",
            "CONSECUTIVE TRADES",
            "-" * 40,
            f"Max Consecutive Wins:   {result.max_consecutive_wins}",
            f"Max Consecutive Losses: {result.max_consecutive_losses}",
            "",
            "=" * 70,
        ]
        
        return "\n".join(lines)


def create_synthetic_candles(
    n_periods: int = 1000,
    trend: float = 0.0,
    volatility: float = 0.02,
    start_price: float = 0.5,
    interval_minutes: int = 15
) -> pd.DataFrame:
    """
    Create synthetic candle data for testing.
    
    Args:
        n_periods: Number of candles to generate
        trend: Daily drift (0.0 = random walk)
        volatility: Volatility parameter
        start_price: Starting price
        interval_minutes: Candle interval in minutes
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(42)
    
    # Generate price path
    returns = np.random.normal(trend / (24 * 4), volatility / np.sqrt(24 * 4), n_periods)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Clamp to valid range
    prices = np.clip(prices, 0.01, 0.99)
    
    # Generate OHLC from close prices
    data = []
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close
        noise = volatility * np.random.random()
        high = min(close * (1 + noise), 0.99)
        low = max(close * (1 - noise), 0.01)
        open_price = low + np.random.random() * (high - low)
        
        timestamp = base_time + timedelta(minutes=i * interval_minutes)
        volume = int(np.random.randint(100, 10000))
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)
