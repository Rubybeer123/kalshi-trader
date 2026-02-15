"""Performance tracking and reporting for trading strategies."""

import json
import math
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

import pandas as pd
import numpy as np
import structlog

from src.data.storage import DatabaseManager

logger = structlog.get_logger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    market_ticker: str = ""
    side: str = ""  # 'yes' or 'no'
    entry_price: float = 0.0
    exit_price: float = 0.0
    contracts: int = 0
    pnl: float = 0.0
    exit_reason: str = ""  # 'target', 'stop', 'manual', 'expiry'
    
    @property
    def is_win(self) -> bool:
        """True if trade was profitable."""
        return self.pnl > 0
    
    @property
    def is_loss(self) -> bool:
        """True if trade was a loss."""
        return self.pnl < 0
    
    @property
    
    def duration_seconds(self) -> Optional[float]:
        """Trade duration if timestamps available."""
        return None  # Would need entry/exit timestamps


@dataclass
class EquitySnapshot:
    """Equity curve data point."""
    timestamp: datetime
    balance: float
    unrealized_pnl: float = 0.0
    
    @property
    def total_equity(self) -> float:
        """Total equity including unrealized."""
        return self.balance + self.unrealized_pnl


@dataclass
class PerformanceMetrics:
    """Calculated performance metrics."""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L statistics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Ratios
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Consecutive
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_consecutive: int = 0
    
    # Time
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Handle datetime serialization
        if self.first_trade_time:
            result['first_trade_time'] = self.first_trade_time.isoformat()
        if self.last_trade_time:
            result['last_trade_time'] = self.last_trade_time.isoformat()
        return result


class PerformanceTracker:
    """
    Tracks trading performance and calculates metrics.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize performance tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "performance.db")
        
        self.db_path = db_path
        self._db = DatabaseManager(db_path)
        self._equity_curve: List[EquitySnapshot] = []
        self._starting_balance: Optional[float] = None
        
        # Initialize database
        self._init_database()
        
        logger.info("PerformanceTracker initialized", db_path=db_path)
    
    def _init_database(self) -> None:
        """Initialize database tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    market_ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    contracts INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    exit_reason TEXT
                )
            """)
            
            # Equity curve table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    balance REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0.0
                )
            """)
            
            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_ticker 
                ON trades (market_ticker)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_time 
                ON trades (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_equity_time 
                ON equity_curve (timestamp)
            """)
            
            conn.commit()
        
        logger.debug("Performance database initialized")
    
    def record_trade(self, trade: Trade) -> int:
        """
        Record a completed trade.
        
        Args:
            trade: Trade to record
            
        Returns:
            Trade ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades 
                (timestamp, market_ticker, side, entry_price, exit_price, contracts, pnl, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(trade.timestamp.timestamp()),
                    trade.market_ticker,
                    trade.side,
                    trade.entry_price,
                    trade.exit_price,
                    trade.contracts,
                    trade.pnl,
                    trade.exit_reason
                )
            )
            conn.commit()
            trade.id = cursor.lastrowid
        
        logger.info(
            "Trade recorded",
            trade_id=trade.id,
            ticker=trade.market_ticker,
            pnl=trade.pnl
        )
        
        return trade.id
    
    def record_equity_snapshot(
        self,
        balance: float,
        unrealized_pnl: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record equity curve snapshot.
        
        Args:
            balance: Account balance
            unrealized_pnl: Unrealized P&L
            timestamp: Snapshot time (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Store in memory for fast access
        snapshot = EquitySnapshot(timestamp, balance, unrealized_pnl)
        self._equity_curve.append(snapshot)
        
        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO equity_curve (timestamp, balance, unrealized_pnl)
                VALUES (?, ?, ?)
                """,
                (int(timestamp.timestamp()), balance, unrealized_pnl)
            )
            conn.commit()
    
    def get_trades(
        self,
        ticker: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Trade]:
        """
        Get trades with optional filtering.
        
        Args:
            ticker: Filter by market ticker
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of trades
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND market_ticker = ?"
            params.append(ticker)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp()))
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp()))
        
        query += " ORDER BY timestamp"
        
        rows = self._db.fetch_all(query, tuple(params))
        
        return [
            Trade(
                id=row['id'],
                timestamp=datetime.fromtimestamp(row['timestamp'], tz=timezone.utc),
                market_ticker=row['market_ticker'],
                side=row['side'],
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                contracts=row['contracts'],
                pnl=row['pnl'],
                exit_reason=row['exit_reason']
            )
            for row in rows
        ]
    
    def get_equity_curve(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            DataFrame with equity curve
        """
        query = "SELECT * FROM equity_curve WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp()))
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp()))
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, sqlite3.connect(self.db_path), params=params)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['total_equity'] = df['balance'] + df['unrealized_pnl']
        
        return df
    
    def calculate_metrics(self, trades: Optional[List[Trade]] = None) -> PerformanceMetrics:
        """
        Calculate performance metrics from trades.
        
        Args:
            trades: List of trades (loads from DB if None)
            
        Returns:
            PerformanceMetrics object
        """
        if trades is None:
            trades = self.get_trades()
        
        if not trades:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # Basic counts
        metrics.total_trades = len(trades)
        metrics.winning_trades = sum(1 for t in trades if t.is_win)
        metrics.losing_trades = sum(1 for t in trades if t.is_loss)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # P&L statistics
        wins = [t.pnl for t in trades if t.is_win]
        losses = [t.pnl for t in trades if t.is_loss]
        
        metrics.gross_profit = sum(wins)
        metrics.gross_loss = abs(sum(losses))
        metrics.net_profit = metrics.gross_profit - metrics.gross_loss
        
        if wins:
            metrics.avg_win = sum(wins) / len(wins)
            metrics.largest_win = max(wins)
        
        if losses:
            metrics.avg_loss = sum(losses) / len(losses)
            metrics.largest_loss = min(losses)
        
        # Profit factor
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        elif metrics.gross_profit > 0:
            metrics.profit_factor = float('inf')
        
        # Expectancy
        loss_rate = 1 - metrics.win_rate
        metrics.expectancy = (metrics.win_rate * metrics.avg_win) - (loss_rate * abs(metrics.avg_loss))
        
        # Consecutive trades
        max_consec_wins = 0
        max_consec_losses = 0
        current_consec = 0
        current_type = None
        
        for trade in trades:
            if trade.is_win:
                if current_type == 'win':
                    current_consec += 1
                else:
                    current_consec = 1
                    current_type = 'win'
                max_consec_wins = max(max_consec_wins, current_consec)
            elif trade.is_loss:
                if current_type == 'loss':
                    current_consec += 1
                else:
                    current_consec = 1
                    current_type = 'loss'
                max_consec_losses = max(max_consec_losses, current_consec)
        
        metrics.max_consecutive_wins = max_consec_wins
        metrics.max_consecutive_losses = max_consec_losses
        metrics.current_consecutive = current_consec if current_consec > 0 else 0
        
        # Time range
        metrics.first_trade_time = min(t.timestamp for t in trades)
        metrics.last_trade_time = max(t.timestamp for t in trades)
        
        return metrics
    
    def calculate_sharpe_ratio(
        self,
        risk_free_rate: float = 0.0,
        period: str = 'daily'
    ) -> float:
        """
        Calculate Sharpe ratio from equity curve.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0)
            period: 'daily' or 'hourly'
            
        Returns:
            Sharpe ratio
        """
        df = self.get_equity_curve()
        
        if df.empty or len(df) < 2:
            return 0.0
        
        # Calculate returns
        df['returns'] = df['total_equity'].pct_change().dropna()
        
        if df['returns'].empty or df['returns'].std() == 0:
            return 0.0
        
        # Annualize based on period
        if period == 'daily':
            periods_per_year = 365
        elif period == 'hourly':
            periods_per_year = 365 * 24
        else:
            periods_per_year = 365
        
        # Sharpe = (mean return - risk free) / std dev * sqrt(periods)
        excess_returns = df['returns'] - (risk_free_rate / periods_per_year)
        sharpe = excess_returns.mean() / excess_returns.std() * math.sqrt(periods_per_year)
        
        return sharpe
    
    def calculate_max_drawdown(self) -> Tuple[float, float]:
        """
        Calculate maximum drawdown.
        
        Returns:
            Tuple of (max_drawdown_amount, max_drawdown_percentage)
        """
        df = self.get_equity_curve()
        
        if df.empty:
            return 0.0, 0.0
        
        # Calculate running maximum
        df['peak'] = df['total_equity'].cummax()
        df['drawdown'] = df['total_equity'] - df['peak']
        df['drawdown_pct'] = df['drawdown'] / df['peak']
        
        max_dd = df['drawdown'].min()
        max_dd_pct = df['drawdown_pct'].min()
        
        return max_dd, abs(max_dd_pct)
    
    def update_metrics(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Update metrics with calculated values."""
        # Calculate Sharpe and drawdown
        sharpe = self.calculate_sharpe_ratio()
        max_dd, max_dd_pct = self.calculate_max_drawdown()
        
        metrics.sharpe_ratio = sharpe
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_pct = max_dd_pct
        
        return metrics
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get complete performance metrics."""
        trades = self.get_trades()
        metrics = self.calculate_metrics(trades)
        return self.update_metrics(metrics)
    
    def generate_report(self, format: str = 'text') -> str:
        """
        Generate performance report.
        
        Args:
            format: 'text', 'json', or 'html'
            
        Returns:
            Report string
        """
        metrics = self.get_metrics()
        
        if format == 'json':
            return json.dumps(metrics.to_dict(), indent=2)
        
        elif format == 'html':
            return self._generate_html_report(metrics)
        
        else:  # text
            return self._generate_text_report(metrics)
    
    def _generate_text_report(self, metrics: PerformanceMetrics) -> str:
        """Generate text format report."""
        lines = [
            "=" * 60,
            "           PERFORMANCE REPORT",
            "=" * 60,
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades:        {metrics.total_trades}",
            f"Winning Trades:      {metrics.winning_trades}",
            f"Losing Trades:       {metrics.losing_trades}",
            f"Win Rate:            {metrics.win_rate:.1%}",
            "",
            "P&L STATISTICS",
            "-" * 40,
            f"Net Profit:          ${metrics.net_profit:,.2f}",
            f"Gross Profit:        ${metrics.gross_profit:,.2f}",
            f"Gross Loss:          ${metrics.gross_loss:,.2f}",
            f"Average Win:         ${metrics.avg_win:,.2f}",
            f"Average Loss:        ${metrics.avg_loss:,.2f}",
            f"Largest Win:         ${metrics.largest_win:,.2f}",
            f"Largest Loss:        ${metrics.largest_loss:,.2f}",
            "",
            "RATIOS",
            "-" * 40,
            f"Profit Factor:       {metrics.profit_factor:.2f}",
            f"Expectancy:          ${metrics.expectancy:,.2f}",
            f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f}",
            f"Max Drawdown:        ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1%})",
            "",
            "CONSECUTIVE TRADES",
            "-" * 40,
            f"Max Consecutive Wins:   {metrics.max_consecutive_wins}",
            f"Max Consecutive Losses: {metrics.max_consecutive_losses}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def _generate_html_report(self, metrics: PerformanceMetrics) -> str:
        """Generate HTML format report."""
        html = f"""
        <html>
        <head>
            <title>Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 500px; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
            
            <h2>Trade Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{metrics.total_trades}</td></tr>
                <tr><td>Win Rate</td><td class="{'positive' if metrics.win_rate > 0.5 else 'negative'}">{metrics.win_rate:.1%}</td></tr>
                <tr><td>Net Profit</td><td class="{'positive' if metrics.net_profit > 0 else 'negative'}">${metrics.net_profit:,.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{metrics.profit_factor:.2f}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{metrics.sharpe_ratio:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">{metrics.max_drawdown_pct:.1%}</td></tr>
            </table>
        </body>
        </html>
        """
        return html
    
    def export_trades_to_csv(self, filepath: str) -> None:
        """Export trades to CSV file."""
        trades = self.get_trades()
        
        if not trades:
            logger.warning("No trades to export")
            return
        
        df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'ticker': t.market_ticker,
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'contracts': t.contracts,
                'pnl': t.pnl,
                'exit_reason': t.exit_reason
            }
            for t in trades
        ])
        
        df.to_csv(filepath, index=False)
        logger.info("Trades exported to CSV", filepath=filepath)
    
    def reset(self) -> None:
        """Clear all performance data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM equity_curve")
            conn.commit()
        
        self._equity_curve.clear()
        
        logger.info("Performance data reset")
    
    def get_daily_stats(self) -> pd.DataFrame:
        """Get daily performance statistics."""
        trades = self.get_trades()
        
        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'date': t.timestamp.date(),
                'pnl': t.pnl,
                'is_win': t.is_win
            }
            for t in trades
        ])
        
        daily = df.groupby('date').agg({
            'pnl': 'sum',
            'is_win': ['count', 'sum']
        }).reset_index()
        
        daily.columns = ['date', 'total_pnl', 'total_trades', 'winning_trades']
        daily['win_rate'] = daily['winning_trades'] / daily['total_trades']
        
        return daily
