"""Tests for performance tracking and reporting."""

import os
import tempfile
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone

import numpy as np

from src.performance_tracker import (
    PerformanceTracker,
    Trade,
    EquitySnapshot,
    PerformanceMetrics,
)


@pytest_asyncio.fixture
async def tracker():
    """Create performance tracker with temp database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    tracker = PerformanceTracker(db_path=db_path)
    
    yield tracker
    
    # Cleanup
    os.unlink(db_path)


class TestTrade:
    """Test Trade dataclass."""
    
    def test_win_detection(self):
        """Test win detection."""
        win_trade = Trade(pnl=100.0)
        loss_trade = Trade(pnl=-50.0)
        flat_trade = Trade(pnl=0.0)
        
        assert win_trade.is_win is True
        assert win_trade.is_loss is False
        
        assert loss_trade.is_win is False
        assert loss_trade.is_loss is True
        
        assert flat_trade.is_win is False
        assert flat_trade.is_loss is False


class TestEquitySnapshot:
    """Test EquitySnapshot dataclass."""
    
    def test_total_equity(self):
        """Test total equity calculation."""
        snapshot = EquitySnapshot(
            timestamp=datetime.now(timezone.utc),
            balance=1000.0,
            unrealized_pnl=150.0
        )
        
        assert snapshot.total_equity == 1150.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            total_trades=10,
            win_rate=0.6,
            net_profit=500.0
        )
        
        d = metrics.to_dict()
        
        assert d['total_trades'] == 10
        assert d['win_rate'] == 0.6
        assert d['net_profit'] == 500.0


class TestRecordTrade:
    """Test trade recording."""
    
    @pytest.mark.asyncio
    async def test_record_single_trade(self, tracker):
        """Test recording a single trade."""
        trade = Trade(
            timestamp=datetime.now(timezone.utc),
            market_ticker="KXBTC",
            side="yes",
            entry_price=0.60,
            exit_price=0.70,
            contracts=10,
            pnl=1.0,
            exit_reason="target"
        )
        
        trade_id = tracker.record_trade(trade)
        
        assert trade_id is not None
        assert trade_id > 0
    
    @pytest.mark.asyncio
    async def test_record_multiple_trades(self, tracker):
        """Test recording multiple trades."""
        for i in range(5):
            trade = Trade(
                timestamp=datetime.now(timezone.utc),
                market_ticker="KXBTC",
                side="yes",
                entry_price=0.60,
                exit_price=0.70,
                contracts=10,
                pnl=1.0
            )
            tracker.record_trade(trade)
        
        trades = tracker.get_trades()
        
        assert len(trades) == 5


class TestCalculateMetrics:
    """Test metric calculations."""
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        
        return [
            # 6 wins of $100 each = $600
            Trade(timestamp=base_time, pnl=100.0),
            Trade(timestamp=base_time + timedelta(hours=1), pnl=100.0),
            Trade(timestamp=base_time + timedelta(hours=2), pnl=100.0),
            Trade(timestamp=base_time + timedelta(hours=3), pnl=100.0),
            Trade(timestamp=base_time + timedelta(hours=4), pnl=100.0),
            Trade(timestamp=base_time + timedelta(hours=5), pnl=100.0),
            
            # 4 losses of $50 each = $200
            Trade(timestamp=base_time + timedelta(hours=6), pnl=-50.0),
            Trade(timestamp=base_time + timedelta(hours=7), pnl=-50.0),
            Trade(timestamp=base_time + timedelta(hours=8), pnl=-50.0),
            Trade(timestamp=base_time + timedelta(hours=9), pnl=-50.0),
        ]
    
    def test_total_trades(self, tracker, sample_trades):
        """Test total trade count."""
        metrics = tracker.calculate_metrics(sample_trades)
        
        assert metrics.total_trades == 10
    
    def test_win_rate(self, tracker, sample_trades):
        """Test win rate calculation."""
        metrics = tracker.calculate_metrics(sample_trades)
        
        assert metrics.winning_trades == 6
        assert metrics.losing_trades == 4
        assert metrics.win_rate == 0.6
    
    def test_gross_profit_loss(self, tracker, sample_trades):
        """Test gross profit and loss."""
        metrics = tracker.calculate_metrics(sample_trades)
        
        assert metrics.gross_profit == 600.0
        assert metrics.gross_loss == 200.0
    
    def test_net_profit(self, tracker, sample_trades):
        """Test net profit."""
        metrics = tracker.calculate_metrics(sample_trades)
        
        assert metrics.net_profit == 400.0
    
    def test_avg_win_loss(self, tracker, sample_trades):
        """Test average win and loss."""
        metrics = tracker.calculate_metrics(sample_trades)
        
        assert metrics.avg_win == 100.0
        assert metrics.avg_loss == -50.0
    
    def test_profit_factor(self, tracker, sample_trades):
        """Test profit factor."""
        metrics = tracker.calculate_metrics(sample_trades)
        
        # PF = Gross Profit / Gross Loss = 600 / 200 = 3.0
        assert metrics.profit_factor == 3.0
    
    def test_expectancy(self, tracker, sample_trades):
        """Test expectancy calculation."""
        metrics = tracker.calculate_metrics(sample_trades)
        
        # Expectancy = (Win Rate * Avg Win) - (Loss Rate * |Avg Loss|)
        # = (0.6 * 100) - (0.4 * 50)
        # = 60 - 20 = 40
        assert metrics.expectancy == 40.0
    
    def test_largest_win_loss(self, tracker, sample_trades):
        """Test largest win/loss tracking."""
        # Add extreme trades
        sample_trades.append(Trade(timestamp=datetime.now(timezone.utc), pnl=200.0))
        sample_trades.append(Trade(timestamp=datetime.now(timezone.utc), pnl=-100.0))
        
        metrics = tracker.calculate_metrics(sample_trades)
        
        assert metrics.largest_win == 200.0
        assert metrics.largest_loss == -100.0
    
    def test_consecutive_trades(self, tracker):
        """Test consecutive trade tracking."""
        trades = [
            Trade(pnl=100),  # Win
            Trade(pnl=100),  # Win
            Trade(pnl=100),  # Win
            Trade(pnl=-50),  # Loss
            Trade(pnl=-50),  # Loss
            Trade(pnl=100),  # Win
            Trade(pnl=-50),  # Loss
            Trade(pnl=-50),  # Loss
            Trade(pnl=-50),  # Loss
        ]
        
        metrics = tracker.calculate_metrics(trades)
        
        assert metrics.max_consecutive_wins == 3
        assert metrics.max_consecutive_losses == 3


class TestDrawdownCalculation:
    """Test maximum drawdown calculations."""
    
    def test_no_drawdown(self, tracker):
        """Test no drawdown when equity always rises."""
        # Rising equity
        for i in range(10):
            tracker.record_equity_snapshot(
                balance=1000 + (i * 100),
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i)
            )
        
        max_dd, max_dd_pct = tracker.calculate_max_drawdown()
        
        assert max_dd == 0.0
        assert max_dd_pct == 0.0
    
    def test_simple_drawdown(self, tracker):
        """Test simple drawdown calculation."""
        # Peak at 1500, drop to 1000
        tracker.record_equity_snapshot(1000, timestamp=datetime.now(timezone.utc))
        tracker.record_equity_snapshot(1500, timestamp=datetime.now(timezone.utc) + timedelta(hours=1))
        tracker.record_equity_snapshot(1200, timestamp=datetime.now(timezone.utc) + timedelta(hours=2))
        tracker.record_equity_snapshot(1000, timestamp=datetime.now(timezone.utc) + timedelta(hours=3))
        
        max_dd, max_dd_pct = tracker.calculate_max_drawdown()
        
        # Drawdown from 1500 to 1000 = 500
        assert max_dd == -500.0
        # Percentage = 500 / 1500 = 33.3%
        assert abs(max_dd_pct - 0.333) < 0.01
    
    def test_multiple_drawdowns(self, tracker):
        """Test finding largest of multiple drawdowns."""
        tracker.record_equity_snapshot(1000, timestamp=datetime.now(timezone.utc))
        tracker.record_equity_snapshot(1500, timestamp=datetime.now(timezone.utc) + timedelta(hours=1))  # Peak 1
        tracker.record_equity_snapshot(1300, timestamp=datetime.now(timezone.utc) + timedelta(hours=2))  # DD 1: 200
        tracker.record_equity_snapshot(2000, timestamp=datetime.now(timezone.utc) + timedelta(hours=3))  # Peak 2
        tracker.record_equity_snapshot(1500, timestamp=datetime.now(timezone.utc) + timedelta(hours=4))  # DD 2: 500
        
        max_dd, max_dd_pct = tracker.calculate_max_drawdown()
        
        # Should find the larger drawdown of 500
        assert max_dd == -500.0


class TestSharpeRatio:
    """Test Sharpe ratio calculations."""
    
    def test_sharpe_no_returns(self, tracker):
        """Test Sharpe with no returns."""
        sharpe = tracker.calculate_sharpe_ratio()
        
        assert sharpe == 0.0
    
    def test_sharpe_flat_returns(self, tracker):
        """Test Sharpe with zero volatility."""
        # Same equity every time
        for i in range(10):
            tracker.record_equity_snapshot(1000, timestamp=datetime.now(timezone.utc) + timedelta(days=i))
        
        sharpe = tracker.calculate_sharpe_ratio()
        
        assert sharpe == 0.0
    
    def test_sharpe_positive_returns(self, tracker):
        """Test Sharpe with positive returns."""
        # Steady positive returns
        balance = 1000
        for i in range(30):
            balance *= 1.001  # 0.1% daily return
            tracker.record_equity_snapshot(balance, timestamp=datetime.now(timezone.utc) + timedelta(days=i))
        
        sharpe = tracker.calculate_sharpe_ratio()
        
        # Should be positive with consistent returns
        assert sharpe > 0
    
    def test_sharpe_negative_returns(self, tracker):
        """Test Sharpe with negative returns."""
        # Steady negative returns
        balance = 1000
        for i in range(30):
            balance *= 0.999  # -0.1% daily return
            tracker.record_equity_snapshot(balance, timestamp=datetime.now(timezone.utc) + timedelta(days=i))
        
        sharpe = tracker.calculate_sharpe_ratio()
        
        # Should be negative
        assert sharpe < 0


class TestGenerateReport:
    """Test report generation."""
    
    @pytest.fixture
    def tracker_with_trades(self, tracker):
        """Create tracker with sample trades."""
        for i in range(5):
            trade = Trade(
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
                market_ticker="KXBTC",
                side="yes",
                entry_price=0.60,
                exit_price=0.70 if i % 2 == 0 else 0.55,
                contracts=10,
                pnl=1.0 if i % 2 == 0 else -0.5,
                exit_reason="target" if i % 2 == 0 else "stop"
            )
            tracker.record_trade(trade)
        
        return tracker
    
    def test_generate_text_report(self, tracker_with_trades):
        """Test text report generation."""
        report = tracker_with_trades.generate_report(format='text')
        
        assert "PERFORMANCE REPORT" in report
        assert "Total Trades:" in report
        assert "Win Rate:" in report
        assert "Net Profit:" in report
    
    def test_generate_json_report(self, tracker_with_trades):
        """Test JSON report generation."""
        report = tracker_with_trades.generate_report(format='json')
        
        assert '"total_trades"' in report
        assert '"win_rate"' in report
    
    def test_generate_html_report(self, tracker_with_trades):
        """Test HTML report generation."""
        report = tracker_with_trades.generate_report(format='html')
        
        assert "<html>" in report
        assert "<table>" in report
        assert "Performance Report" in report


class TestDailyStats:
    """Test daily statistics."""
    
    def test_daily_stats(self, tracker):
        """Test daily aggregation."""
        base_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        
        # Day 1: 2 trades
        tracker.record_trade(Trade(timestamp=base_time, pnl=100))
        tracker.record_trade(Trade(timestamp=base_time + timedelta(hours=2), pnl=-50))
        
        # Day 2: 1 trade
        tracker.record_trade(Trade(timestamp=base_time + timedelta(days=1), pnl=75))
        
        daily = tracker.get_daily_stats()
        
        assert len(daily) == 2
        assert daily.iloc[0]['total_trades'] == 2
        assert daily.iloc[1]['total_trades'] == 1


class TestExport:
    """Test data export."""
    
    def test_export_to_csv(self, tracker):
        """Test CSV export."""
        # Add trades
        for i in range(3):
            tracker.record_trade(Trade(
                timestamp=datetime.now(timezone.utc),
                market_ticker="KXBTC",
                pnl=100.0
            ))
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            tracker.export_trades_to_csv(csv_path)
            
            # Verify file was created and has content
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
                assert "timestamp" in content
                assert "pnl" in content
        finally:
            os.unlink(csv_path)


class TestReset:
    """Test data reset."""
    
    def test_reset_clears_data(self, tracker):
        """Test reset clears all data."""
        # Add data
        tracker.record_trade(Trade(pnl=100))
        tracker.record_equity_snapshot(1000)
        
        # Verify exists
        assert len(tracker.get_trades()) == 1
        
        # Reset
        tracker.reset()
        
        # Verify cleared
        assert len(tracker.get_trades()) == 0
        df = tracker.get_equity_curve()
        assert len(df) == 0


class TestGetMetrics:
    """Test get_metrics convenience method."""
    
    def test_get_metrics_returns_complete(self, tracker):
        """Test get_metrics returns all metrics."""
        # Add trades and equity
        tracker.record_trade(Trade(pnl=100))
        tracker.record_trade(Trade(pnl=-50))
        tracker.record_equity_snapshot(1000)
        tracker.record_equity_snapshot(1100)
        
        metrics = tracker.get_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 2
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown is not None


class TestKnownDataset:
    """Test with known dataset for verification."""
    
    def test_known_trading_results(self, tracker):
        """
        Test with known dataset:
        10 trades: 6 wins @ $100, 4 losses @ $50
        Expected: Win rate 60%, Net profit $400, PF 3.0
        """
        trades = []
        
        # 6 wins of $100
        for i in range(6):
            trades.append(Trade(
                timestamp=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                pnl=100.0
            ))
        
        # 4 losses of $50
        for i in range(6, 10):
            trades.append(Trade(
                timestamp=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                pnl=-50.0
            ))
        
        # Record all
        for trade in trades:
            tracker.record_trade(trade)
        
        # Get metrics
        metrics = tracker.get_metrics()
        
        # Verify
        assert metrics.total_trades == 10
        assert metrics.winning_trades == 6
        assert metrics.losing_trades == 4
        assert abs(metrics.win_rate - 0.6) < 0.001
        assert metrics.net_profit == 400.0
        assert metrics.profit_factor == 3.0
        assert abs(metrics.expectancy - 40.0) < 0.001
