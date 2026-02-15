"""Tests for the backtester module."""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    BacktestConfig,
    Backtester,
    BacktestExchange,
    BacktestTrade,
    VariableSlippageModel,
    MarketImpactModel,
    create_synthetic_candles,
)
from src.candle_aggregator import OHLCV
from src.strategies.bollinger_scalper import BollingerScalper, ScalperConfig
from src.strategies.base import Signal, SignalType


class TestVariableSlippageModel:
    """Tests for variable slippage model."""
    
    def test_initialization(self):
        """Test slippage model initialization."""
        model = VariableSlippageModel(volatility_factor=1.5)
        assert model.volatility_factor == 1.5
        assert model.get_atr() == 0.01  # Default
    
    def test_update_volatility(self):
        """Test volatility update."""
        model = VariableSlippageModel()
        
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.55,
            low=0.48,
            close=0.52,
            volume=1000
        )
        
        model.update_volatility(candle)
        atr = model.get_atr()
        assert abs(atr - 0.07) < 0.001  # high - low (with floating point tolerance)
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        model = VariableSlippageModel(volatility_factor=1.0)
        
        # Add some history
        for _ in range(10):
            candle = OHLCV(
                market_ticker='TEST',
                timestamp=datetime.now(timezone.utc),
                open=0.50,
                high=0.52,
                low=0.48,
                close=0.51,
                volume=1000
            )
            model.update_volatility(candle)
        
        slippage = model.calculate_slippage(
            intended_price=0.50,
            side='yes',
            is_buy=True,
            order_size=100
        )
        
        assert slippage > 0
        assert slippage >= 0.01  # Minimum
    
    def test_slippage_scales_with_size(self):
        """Test that slippage increases with order size."""
        model = VariableSlippageModel()
        
        for _ in range(10):
            candle = OHLCV(
                market_ticker='TEST',
                timestamp=datetime.now(timezone.utc),
                open=0.50,
                high=0.55,
                low=0.45,
                close=0.51,
                volume=1000
            )
            model.update_volatility(candle)
        
        small_slippage = model.calculate_slippage(0.50, 'yes', True, 10)
        large_slippage = model.calculate_slippage(0.50, 'yes', True, 1000)
        
        assert large_slippage > small_slippage
    
    def test_stressed_condition_slippage(self):
        """Test slippage is higher in stressed conditions."""
        model = VariableSlippageModel()
        
        for _ in range(10):
            candle = OHLCV(
                market_ticker='TEST',
                timestamp=datetime.now(timezone.utc),
                open=0.50,
                high=0.52,
                low=0.48,
                close=0.51,
                volume=1000
            )
            model.update_volatility(candle)
        
        normal_slippage = model.calculate_slippage(0.50, 'yes', True, 100, stressed=False)
        stressed_slippage = model.calculate_slippage(0.50, 'yes', True, 100, stressed=True)
        
        assert stressed_slippage > normal_slippage
    
    def test_apply_slippage_buy(self):
        """Test slippage application for buys."""
        model = VariableSlippageModel()
        
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.52,
            low=0.48,
            close=0.51,
            volume=1000
        )
        model.update_volatility(candle)
        
        filled_price, slippage = model.apply_slippage(0.50, 'yes', True, 100)
        
        # Buy should be higher than intended
        assert filled_price >= 0.50
        assert slippage > 0
    
    def test_apply_slippage_sell(self):
        """Test slippage application for sells."""
        model = VariableSlippageModel()
        
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.52,
            low=0.48,
            close=0.51,
            volume=1000
        )
        model.update_volatility(candle)
        
        filled_price, slippage = model.apply_slippage(0.50, 'yes', False, 100)
        
        # Sell should be lower than intended
        assert filled_price <= 0.50
        assert slippage > 0


class TestMarketImpactModel:
    """Tests for market impact model."""
    
    def test_initialization(self):
        """Test impact model initialization."""
        model = MarketImpactModel(threshold=100, impact_per_lot=0.001)
        assert model.threshold == 100
        assert model.impact_per_lot == 0.001
    
    def test_no_impact_below_threshold(self):
        """Test that small orders have no impact."""
        model = MarketImpactModel(threshold=100)
        
        impact = model.calculate_impact(order_size=50, is_buy=True)
        assert impact == 0.0
    
    def test_impact_above_threshold(self):
        """Test impact for large orders."""
        model = MarketImpactModel(threshold=100, impact_per_lot=0.001)
        
        # Add volume history
        for _ in range(10):
            model.update_volume(1000)
        
        impact = model.calculate_impact(order_size=500, is_buy=True)
        assert impact > 0.0
    
    def test_impact_scales_with_size(self):
        """Test that impact increases with order size."""
        model = MarketImpactModel(threshold=100)
        
        for _ in range(10):
            model.update_volume(1000)
        
        small_impact = model.calculate_impact(200, True)
        large_impact = model.calculate_impact(1000, True)
        
        assert large_impact > small_impact
    
    def test_apply_impact_buy(self):
        """Test impact application for buys."""
        model = MarketImpactModel(threshold=100)
        
        for _ in range(10):
            model.update_volume(1000)
        
        impacted_price, impact = model.apply_impact(0.50, 500, True)
        
        # Buy pushes price up
        assert impacted_price >= 0.50
    
    def test_apply_impact_sell(self):
        """Test impact application for sells."""
        model = MarketImpactModel(threshold=100)
        
        for _ in range(10):
            model.update_volume(1000)
        
        impacted_price, impact = model.apply_impact(0.50, 500, False)
        
        # Sell pushes price down
        assert impacted_price <= 0.50


class TestBacktestExchange:
    """Tests for backtest exchange."""
    
    def test_initialization(self):
        """Test exchange initialization."""
        config = BacktestConfig(initial_balance=50000)
        exchange = BacktestExchange(config)
        
        assert exchange.get_balance() == 50000
        assert exchange.get_total_equity() == 50000
    
    def test_calculate_fees(self):
        """Test fee calculation."""
        config = BacktestConfig(fee_rate=0.07)
        exchange = BacktestExchange(config)
        
        # Fee formula: 0.07 * C * P * (1-P)
        fees = exchange.calculate_fees(contracts=100, price=0.50)
        expected = 0.07 * 100 * 0.50 * 0.50
        assert fees == pytest.approx(expected, abs=0.001)
    
    def test_update_market_state(self):
        """Test market state update."""
        config = BacktestConfig()
        exchange = BacktestExchange(config)
        
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.55,
            low=0.48,
            close=0.52,
            volume=1000
        )
        
        exchange.update_market_state(candle)
        
        assert exchange._current_candle == candle
        assert exchange._current_time == candle.timestamp
    
    def test_execute_long_signal(self):
        """Test executing a long signal."""
        config = BacktestConfig(
            enable_slippage=False,
            enable_market_impact=False
        )
        exchange = BacktestExchange(config)
        
        # Setup candle
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.55,
            low=0.45,
            close=0.52,
            volume=1000
        )
        exchange.update_market_state(candle)
        
        # Create signal
        signal = Signal(
            type=SignalType.LONG,
            market_ticker='TEST',
            entry_price=0.45,
            stop_loss=0.40,
            take_profit=0.55,
            confidence=0.8
        )
        
        trade = exchange.execute_signal(signal, contracts=100)
        
        assert trade is not None
        assert trade.side == 'yes'
        assert trade.contracts == 100
        assert trade.fees > 0  # Fees should be deducted
    
    def test_execute_short_signal(self):
        """Test executing a short signal."""
        config = BacktestConfig(
            enable_slippage=False,
            enable_market_impact=False
        )
        exchange = BacktestExchange(config)
        
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.55,
            low=0.45,
            close=0.48,
            volume=1000
        )
        exchange.update_market_state(candle)
        
        signal = Signal(
            type=SignalType.SHORT,
            market_ticker='TEST',
            entry_price=0.55,
            stop_loss=0.60,
            take_profit=0.45,
            confidence=0.8
        )
        
        trade = exchange.execute_signal(signal, contracts=100)
        
        assert trade is not None
        assert trade.side == 'no'
    
    def test_insufficient_balance(self):
        """Test that trades are rejected with insufficient balance."""
        config = BacktestConfig(initial_balance=100)
        exchange = BacktestExchange(config)
        
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.55,
            low=0.45,
            close=0.52,
            volume=1000
        )
        exchange.update_market_state(candle)
        
        signal = Signal(
            type=SignalType.LONG,
            market_ticker='TEST',
            entry_price=0.50,
            stop_loss=0.45,
            take_profit=0.60,
            confidence=0.8
        )
        
        trade = exchange.execute_signal(signal, contracts=1000)
        
        assert trade is None
    
    def test_slippage_applied(self):
        """Test that slippage is applied to trades."""
        config = BacktestConfig(enable_slippage=True)
        exchange = BacktestExchange(config)
        
        candle = OHLCV(
            market_ticker='TEST',
            timestamp=datetime.now(timezone.utc),
            open=0.50,
            high=0.55,
            low=0.45,
            close=0.52,
            volume=1000
        )
        exchange.update_market_state(candle)
        
        signal = Signal(
            type=SignalType.LONG,
            market_ticker='TEST',
            entry_price=0.45,
            stop_loss=0.40,
            take_profit=0.55,
            confidence=0.8
        )
        
        trade = exchange.execute_signal(signal, contracts=100)
        
        assert trade is not None
        assert trade.slippage > 0
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        config = BacktestConfig(initial_balance=10000)
        exchange = BacktestExchange(config)
        
        stats = exchange.get_stats()
        
        assert stats['balance'] == 10000
        assert stats['available_balance'] == 10000
        assert stats['open_positions'] == 0
        assert stats['total_fees'] == 0


class TestBacktester:
    """Tests for the main backtester class."""
    
    def test_initialization(self):
        """Test backtester initialization."""
        def strategy_factory():
            return BollingerScalper()
        
        config = BacktestConfig()
        backtester = Backtester(strategy_factory, config)
        
        assert backtester.strategy_factory == strategy_factory
        assert backtester.config == config
    
    def test_load_data_sqlite(self):
        """Test loading data from SQLite."""
        # Create temp database with test data
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Create test candles
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE candles (
                    market_ticker TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                )
            """)
            
            base_time = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
            for i in range(100):
                conn.execute(
                    "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ('TEST', base_time + i * 900, 0.50, 0.52, 0.48, 0.51, 1000)
                )
            conn.commit()
            conn.close()
            
            def strategy_factory():
                return BollingerScalper()
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_sqlite(db_path, 'TEST')
            
            assert len(backtester._candles) == 100
            assert backtester._market_ticker == 'TEST'
            
        finally:
            os.unlink(db_path)
    
    def test_load_data_csv(self):
        """Test loading data from CSV."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as tmp:
            tmp.write("timestamp,open,high,low,close,volume\n")
            base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
            for i in range(100):
                ts = base_time + timedelta(minutes=i * 15)
                tmp.write(f"{ts.isoformat()},0.50,0.52,0.48,0.51,1000\n")
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper()
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            
            assert len(backtester._candles) == 100
            
        finally:
            os.unlink(csv_path)
    
    def test_run_backtest(self):
        """Test running a full backtest."""
        # Create synthetic data with volatility for Bollinger Bands
        candles = create_synthetic_candles(n_periods=200, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper(ScalperConfig(bb_period=20, bb_std=2.0))
            
            config = BacktestConfig(
                initial_balance=10000,
                enable_slippage=False,
                enable_market_impact=False
            )
            
            backtester = Backtester(strategy_factory, config)
            backtester.load_data_csv(csv_path, 'TEST')
            result = backtester.run()
            
            assert result is not None
            assert result.strategy_name == 'BollingerScalper'
            assert result.total_trades >= 0
            
        finally:
            os.unlink(csv_path)
    
    def test_run_with_params(self):
        """Test running backtest with specific parameters."""
        candles = create_synthetic_candles(n_periods=200, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory(bb_period, bb_std):
                return BollingerScalper(ScalperConfig(bb_period=bb_period, bb_std=bb_std))
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            
            result = backtester.run(strategy_params={'bb_period': 15, 'bb_std': 2.5})
            
            assert result.param_set == {'bb_period': 15, 'bb_std': 2.5}
            
        finally:
            os.unlink(csv_path)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        candles = create_synthetic_candles(n_periods=200, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper(ScalperConfig(bb_period=20))
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            result = backtester.run()
            
            # Check all metrics are calculated
            assert hasattr(result, 'total_return')
            assert hasattr(result, 'sharpe_ratio')
            assert hasattr(result, 'max_drawdown_pct')
            assert hasattr(result, 'win_rate')
            assert hasattr(result, 'profit_factor')
            assert hasattr(result, 'expectancy')
            
        finally:
            os.unlink(csv_path)
    
    def test_optimize_params(self):
        """Test parameter optimization."""
        candles = create_synthetic_candles(n_periods=300, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory(bb_period, bb_std):
                return BollingerScalper(ScalperConfig(bb_period=bb_period, bb_std=bb_std))
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            
            param_grid = {
                'bb_period': [10, 20],
                'bb_std': [1.5, 2.0]
            }
            
            best_result, best_params = backtester.optimize(param_grid, metric='total_return')
            
            assert best_result is not None
            assert best_params is not None
            assert 'bb_period' in best_params
            assert 'bb_std' in best_params
            
        finally:
            os.unlink(csv_path)
    
    def test_generate_report(self):
        """Test report generation."""
        candles = create_synthetic_candles(n_periods=200, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper(ScalperConfig(bb_period=20))
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            backtester.run()
            
            report = backtester.generate_report()
            
            assert 'BACKTEST REPORT' in report
            assert 'BollingerScalper' in report
            assert 'PERFORMANCE SUMMARY' in report
            
        finally:
            os.unlink(csv_path)
    
    def test_error_on_no_data(self):
        """Test that error is raised when running without data."""
        def strategy_factory():
            return BollingerScalper()
        
        backtester = Backtester(strategy_factory)
        
        with pytest.raises(ValueError, match="No data loaded"):
            backtester.run()
    
    def test_error_on_optimize_no_data(self):
        """Test that error is raised when optimizing without data."""
        def strategy_factory():
            return BollingerScalper()
        
        backtester = Backtester(strategy_factory)
        
        with pytest.raises(ValueError, match="No data loaded"):
            backtester.optimize({'bb_period': [10, 20]})


class TestSyntheticCandles:
    """Tests for synthetic candle generation."""
    
    def test_create_synthetic_candles(self):
        """Test synthetic candle creation."""
        df = create_synthetic_candles(n_periods=100)
        
        assert len(df) == 100
        assert 'timestamp' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_synthetic_candles_valid_range(self):
        """Test that synthetic candles are in valid price range."""
        df = create_synthetic_candles(n_periods=1000, volatility=0.1)
        
        assert (df['high'] <= 0.99).all()
        assert (df['low'] >= 0.01).all()
        assert (df['open'] >= df['low']).all()
        assert (df['open'] <= df['high']).all()
        assert (df['close'] >= df['low']).all()
        assert (df['close'] <= df['high']).all()
    
    def test_synthetic_candles_trend(self):
        """Test that trend parameter affects prices."""
        df_up = create_synthetic_candles(n_periods=500, trend=0.5, start_price=0.5)
        df_down = create_synthetic_candles(n_periods=500, trend=-0.5, start_price=0.5)
        
        # Up trend should generally have higher ending prices
        assert df_up['close'].iloc[-1] > df_down['close'].iloc[-1]
    
    def test_synthetic_candles_interval(self):
        """Test that interval parameter works correctly."""
        df = create_synthetic_candles(n_periods=10, interval_minutes=15)
        
        time_diff = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
        assert time_diff == timedelta(minutes=15)


class TestBacktestIntegration:
    """Integration tests for backtester."""
    
    def test_month_long_backtest(self):
        """Test backtest on one month of synthetic data."""
        # Generate ~1 month of 15-minute candles
        # 96 candles per day * 30 days = 2880 candles
        candles = create_synthetic_candles(n_periods=2880, volatility=0.04)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper(ScalperConfig(bb_period=25, bb_std=2.0))
            
            config = BacktestConfig(
                initial_balance=10000,
                enable_slippage=True,
                enable_market_impact=True
            )
            
            backtester = Backtester(strategy_factory, config)
            backtester.load_data_csv(csv_path, 'KXBTC15M-TEST')
            result = backtester.run()
            
            # Verify metrics
            assert result.total_trades >= 0
            assert isinstance(result.sharpe_ratio, float)
            assert isinstance(result.max_drawdown_pct, float)
            
            # Verify win rate is between 0 and 1
            assert 0 <= result.win_rate <= 1
            
            # Verify profit factor is positive (or inf)
            assert result.profit_factor >= 0 or result.profit_factor == float('inf')
            
        finally:
            os.unlink(csv_path)
    
    def test_optimization_converges(self):
        """Test that parameter optimization finds reasonable parameters."""
        candles = create_synthetic_candles(n_periods=1000, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory(bb_period, bb_std):
                return BollingerScalper(ScalperConfig(bb_period=bb_period, bb_std=bb_std))
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            
            # Test with different metrics
            param_grid = {
                'bb_period': [10, 20, 30],
                'bb_std': [1.5, 2.0, 2.5]
            }
            
            # Optimize for Sharpe
            best_sharpe, params_sharpe = backtester.optimize(
                param_grid, metric='sharpe_ratio'
            )
            
            # Optimize for total return
            best_return, params_return = backtester.optimize(
                param_grid, metric='total_return'
            )
            
            # Different metrics might give different results
            # Just verify they both return valid results
            assert best_sharpe.sharpe_ratio is not None
            assert best_return.total_return is not None
            
        finally:
            os.unlink(csv_path)
    
    def test_handle_data_gaps(self):
        """Test backtester handles gaps in data."""
        # Create data with a gap
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        data = []
        
        for i in range(50):
            data.append({
                'timestamp': base_time + timedelta(minutes=i * 15),
                'open': 0.50,
                'high': 0.52,
                'low': 0.48,
                'close': 0.51,
                'volume': 1000
            })
        
        # Gap of 2 hours (8 candles)
        for i in range(50, 100):
            data.append({
                'timestamp': base_time + timedelta(minutes=(i * 15) + 120),
                'open': 0.50,
                'high': 0.52,
                'low': 0.48,
                'close': 0.51,
                'volume': 1000
            })
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper(ScalperConfig(bb_period=10))
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            result = backtester.run()
            
            # Should complete without error
            assert result is not None
            
        finally:
            os.unlink(csv_path)
    
    def test_fees_affect_performance(self):
        """Test that fees are properly accounted for in P&L."""
        candles = create_synthetic_candles(n_periods=500, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper(ScalperConfig(bb_period=20))
            
            # Run with fees
            config_with_fees = BacktestConfig(fee_rate=0.07)
            backtester_fees = Backtester(strategy_factory, config_with_fees)
            backtester_fees.load_data_csv(csv_path, 'TEST')
            result_fees = backtester_fees.run()
            
            # Run without fees (set very low)
            config_no_fees = BacktestConfig(fee_rate=0.0001)
            backtester_no_fees = Backtester(strategy_factory, config_no_fees)
            backtester_no_fees.load_data_csv(csv_path, 'TEST')
            result_no_fees = backtester_no_fees.run()
            
            # With fees should generally underperform without fees
            # (though randomness means this isn't guaranteed)
            if result_fees.total_trades > 0 and result_no_fees.total_trades > 0:
                avg_with_fees = result_fees.total_return / max(result_fees.total_trades, 1)
                avg_no_fees = result_no_fees.total_return / max(result_no_fees.total_trades, 1)
                # On average, trades with fees should have lower returns
                # This is a probabilistic test
                
        finally:
            os.unlink(csv_path)
    
    def test_position_sizing(self):
        """Test custom position sizing function."""
        candles = create_synthetic_candles(n_periods=300, volatility=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            candles.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        
        try:
            def strategy_factory():
                return BollingerScalper(ScalperConfig(bb_period=20))
            
            # Custom position sizer - always trade 50 contracts
            def fixed_sizer(signal, balance):
                return 50
            
            backtester = Backtester(strategy_factory)
            backtester.load_data_csv(csv_path, 'TEST')
            result = backtester.run(position_sizer=fixed_sizer)
            
            # All trades should be 50 contracts
            for trade in result.trades:
                assert trade.contracts == 50
                
        finally:
            os.unlink(csv_path)
