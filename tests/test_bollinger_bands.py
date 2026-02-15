"""Tests for Bollinger Bands indicator."""

import math
import pytest
import numpy as np

from src.bollinger_bands import BollingerBands, BollingerValues


class TestBollingerBandsInit:
    """Test BollingerBands initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        bb = BollingerBands()
        
        assert bb.period == 25
        assert bb.std_dev == 2.0
        assert not bb.is_warmed_up()
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        bb = BollingerBands(period=20, std_dev=2.5)
        
        assert bb.period == 20
        assert bb.std_dev == 2.5
    
    def test_initial_values_are_none(self):
        """Test that initial values are None."""
        bb = BollingerBands()
        
        assert bb.get_last_values() is None
        assert bb.get_values() is None


class TestWarmUp:
    """Test warm-up detection."""
    
    def test_not_warmed_up_with_few_prices(self):
        """Test not warmed up with less than period prices."""
        bb = BollingerBands(period=5)
        
        for i in range(4):
            bb.update(100.0 + i)
        
        assert not bb.is_warmed_up()
        assert bb.get_last_values() is None
    
    def test_warmed_up_after_period_prices(self):
        """Test warmed up after exactly period prices."""
        bb = BollingerBands(period=5)
        
        for i in range(5):
            result = bb.update(100.0 + i)
        
        assert bb.is_warmed_up()
        assert result is not None
    
    def test_maintains_warmed_up_after_more_updates(self):
        """Test stays warmed up after more updates."""
        bb = BollingerBands(period=5)
        
        for i in range(10):
            bb.update(100.0 + i)
        
        assert bb.is_warmed_up()


class TestBandCalculation:
    """Test Bollinger Bands calculations."""
    
    def test_known_dataset_values(self):
        """Test calculation on known dataset."""
        bb = BollingerBands(period=5, std_dev=2.0)
        
        # Simple dataset: [10, 11, 12, 13, 14]
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        
        for price in prices:
            result = bb.update(price)
        
        # SMA = (10+11+12+13+14) / 5 = 12.0
        assert result is not None
        upper, middle, lower, bandwidth = result
        
        assert middle == 12.0
        
        # Variance = ((10-12)^2 + (11-12)^2 + (12-12)^2 + (13-12)^2 + (14-12)^2) / 5
        # = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
        # Std dev = sqrt(2.0) = 1.414
        expected_std = math.sqrt(2.0)
        
        assert abs(upper - (12.0 + 2.0 * expected_std)) < 0.001
        assert abs(lower - (12.0 - 2.0 * expected_std)) < 0.001
    
    def test_constant_price_flat_bands(self):
        """Test that constant price produces flat bands."""
        bb = BollingerBands(period=5)
        
        # All same price
        for _ in range(5):
            result = bb.update(100.0)
        
        upper, middle, lower, bandwidth = result
        
        assert middle == 100.0
        assert upper == 100.0  # Std dev = 0
        assert lower == 100.0
        assert bandwidth == 0.0
    
    def test_bandwidth_calculation(self):
        """Test bandwidth calculation as % of price."""
        bb = BollingerBands(period=5)
        
        # Prices: [90, 95, 100, 105, 110]
        # SMA = 100
        # Std dev calculation...
        prices = [90.0, 95.0, 100.0, 105.0, 110.0]
        
        for price in prices:
            result = bb.update(price)
        
        upper, middle, lower, bandwidth = result
        
        # Bandwidth should be ((upper - lower) / middle) * 100
        expected_bandwidth = ((upper - lower) / middle) * 100.0
        assert abs(bandwidth - expected_bandwidth) < 0.001
    
    def test_sliding_window(self):
        """Test that indicator uses sliding window."""
        bb = BollingerBands(period=3)
        
        # Add 5 prices, only last 3 should be used
        bb.update(10.0)
        bb.update(10.0)
        bb.update(10.0)
        bb.update(20.0)
        result = bb.update(20.0)  # Uses [10, 20, 20]
        
        upper, middle, lower, bandwidth = result
        
        # SMA = (10 + 20 + 20) / 3 = 16.67
        expected_middle = (10.0 + 20.0 + 20.0) / 3.0
        assert abs(middle - expected_middle) < 0.001


class TestGetValues:
    """Test value retrieval methods."""
    
    def test_get_last_values_returns_dict(self):
        """Test get_last_values returns dictionary."""
        bb = BollingerBands(period=5)
        
        for i in range(5):
            bb.update(100.0 + i)
        
        values = bb.get_last_values()
        
        assert isinstance(values, dict)
        assert 'upper' in values
        assert 'middle' in values
        assert 'lower' in values
        assert 'bandwidth' in values
    
    def test_get_values_returns_object(self):
        """Test get_values returns BollingerValues."""
        bb = BollingerBands(period=5)
        
        for i in range(5):
            bb.update(100.0 + i)
        
        values = bb.get_values()
        
        assert isinstance(values, BollingerValues)
        assert hasattr(values, 'upper')
        assert hasattr(values, 'middle')
        assert hasattr(values, 'lower')
        assert hasattr(values, 'bandwidth')
    
    def test_get_values_to_dict(self):
        """Test BollingerValues to_dict method."""
        bb = BollingerBands(period=5)
        
        for i in range(5):
            bb.update(100.0 + i)
        
        values = bb.get_values()
        d = values.to_dict()
        
        assert isinstance(d, dict)
        assert 'upper' in d
        assert 'middle' in d
        assert 'lower' in d
        assert 'bandwidth' in d


class TestSignalDetection:
    """Test signal detection methods."""
    
    @pytest.fixture
    def warmed_up_bb(self):
        """Create warmed up Bollinger Bands for testing."""
        bb = BollingerBands(period=5, std_dev=2.0)
        
        # Data: [10, 11, 12, 13, 14]
        # SMA = 12, std = 1.414
        # Upper = 12 + 2*1.414 = 14.83
        # Lower = 12 - 2*1.414 = 9.17
        for price in [10.0, 11.0, 12.0, 13.0, 14.0]:
            bb.update(price)
        
        return bb
    
    def test_is_price_below_lower_band(self, warmed_up_bb):
        """Test price below lower band detection."""
        # Lower band is ~9.17
        assert warmed_up_bb.is_price_below_lower_band(8.0) is True
        assert warmed_up_bb.is_price_below_lower_band(10.0) is False
    
    def test_is_price_above_upper_band(self, warmed_up_bb):
        """Test price above upper band detection."""
        # Upper band is ~14.83
        assert warmed_up_bb.is_price_above_upper_band(16.0) is True
        assert warmed_up_bb.is_price_above_upper_band(14.0) is False
    
    def test_is_price_inside_bands(self, warmed_up_bb):
        """Test price inside bands detection."""
        assert warmed_up_bb.is_price_inside_bands(12.0) is True
        assert warmed_up_bb.is_price_inside_bands(8.0) is False
        assert warmed_up_bb.is_price_inside_bands(16.0) is False
    
    def test_is_body_below_lower_band(self, warmed_up_bb):
        """Test body below lower band detection."""
        # Lower band ~9.17, so body with high=8.5 is below
        assert warmed_up_bb.is_body_below_lower_band(8.0, 8.5) is True
        assert warmed_up_bb.is_body_below_lower_band(9.5, 10.0) is False
    
    def test_is_body_above_upper_band(self, warmed_up_bb):
        """Test body above upper band detection."""
        # Upper band ~14.83, so body with low=15.5 is above
        assert warmed_up_bb.is_body_above_upper_band(15.5, 16.0) is True
        assert warmed_up_bb.is_body_above_upper_band(14.0, 14.5) is False
    
    def test_signal_detection_with_bearish_candle(self, warmed_up_bb):
        """Test oversold signal detection."""
        # Candle entirely below lower band
        signal = warmed_up_bb.get_signal(8.0, 8.5)
        
        assert signal == 'oversold'
    
    def test_signal_detection_with_bullish_candle(self, warmed_up_bb):
        """Test overbought signal detection."""
        # Candle entirely above upper band
        signal = warmed_up_bb.get_signal(15.5, 16.0)
        
        assert signal == 'overbought'
    
    def test_no_signal_when_inside_bands(self, warmed_up_bb):
        """Test no signal when inside bands."""
        signal = warmed_up_bb.get_signal(11.0, 13.0)
        
        assert signal is None
    
    def test_signal_returns_none_when_not_warmed_up(self):
        """Test signal returns None when not warmed up."""
        bb = BollingerBands(period=5)
        
        for i in range(3):  # Not enough
            bb.update(100.0 + i)
        
        signal = bb.get_signal(100.0, 101.0)
        
        assert signal is None


class TestPercentB:
    """Test %B indicator."""
    
    @pytest.fixture
    def warmed_up_bb(self):
        """Create warmed up Bollinger Bands."""
        bb = BollingerBands(period=5)
        
        # Data: [10, 11, 12, 13, 14]
        # Middle = 12, Upper ~14.83, Lower ~9.17
        for price in [10.0, 11.0, 12.0, 13.0, 14.0]:
            bb.update(price)
        
        return bb
    
    def test_percent_b_at_middle(self, warmed_up_bb):
        """Test %B at middle band."""
        # At middle band, %B = 0.5
        pct_b = warmed_up_bb.get_percent_b(12.0)
        
        assert abs(pct_b - 0.5) < 0.01
    
    def test_percent_b_at_upper(self, warmed_up_bb):
        """Test %B at upper band."""
        # Get upper value
        values = warmed_up_bb.get_values()
        
        pct_b = warmed_up_bb.get_percent_b(values.upper)
        
        assert abs(pct_b - 1.0) < 0.01
    
    def test_percent_b_at_lower(self, warmed_up_bb):
        """Test %B at lower band."""
        values = warmed_up_bb.get_values()
        
        pct_b = warmed_up_bb.get_percent_b(values.lower)
        
        assert abs(pct_b - 0.0) < 0.01
    
    def test_percent_b_above_upper(self, warmed_up_bb):
        """Test %B above upper band."""
        pct_b = warmed_up_bb.get_percent_b(20.0)
        
        assert pct_b > 1.0
    
    def test_percent_b_below_lower(self, warmed_up_bb):
        """Test %B below lower band."""
        pct_b = warmed_up_bb.get_percent_b(5.0)
        
        assert pct_b < 0.0
    
    def test_percent_b_returns_none_when_not_warmed_up(self):
        """Test %B returns None when not warmed up."""
        bb = BollingerBands(period=5)
        
        pct_b = bb.get_percent_b(100.0)
        
        assert pct_b is None


class TestReset:
    """Test reset functionality."""
    
    def test_reset_clears_history(self):
        """Test reset clears price history."""
        bb = BollingerBands(period=5)
        
        for i in range(5):
            bb.update(100.0 + i)
        
        assert bb.is_warmed_up()
        
        bb.reset()
        
        assert not bb.is_warmed_up()
        assert bb.get_last_values() is None
    
    def test_reset_allows_rebuild(self):
        """Test that we can rebuild after reset."""
        bb = BollingerBands(period=5)
        
        for i in range(5):
            bb.update(100.0 + i)
        
        bb.reset()
        
        # Rebuild with different data
        for i in range(5):
            result = bb.update(200.0 + i)
        
        assert bb.is_warmed_up()
        assert result is not None


class TestStats:
    """Test statistics reporting."""
    
    def test_get_stats_returns_expected_fields(self):
        """Test that get_stats returns all expected fields."""
        bb = BollingerBands(period=5)
        
        stats = bb.get_stats()
        
        assert 'period' in stats
        assert 'std_dev' in stats
        assert 'data_points' in stats
        assert 'warmed_up' in stats
        assert 'using_talib' in stats
        assert 'current_values' in stats
    
    def test_stats_reflects_state(self):
        """Test that stats reflect current state."""
        bb = BollingerBands(period=5)
        
        stats = bb.get_stats()
        
        assert stats['period'] == 5
        assert stats['data_points'] == 0
        assert stats['warmed_up'] is False
        
        # Add data
        for i in range(5):
            bb.update(100.0 + i)
        
        stats = bb.get_stats()
        
        assert stats['data_points'] == 5
        assert stats['warmed_up'] is True


class TestUpdateReturnValue:
    """Test update method return values."""
    
    def test_returns_none_before_warmup(self):
        """Test that update returns None before warm-up."""
        bb = BollingerBands(period=5)
        
        results = []
        for i in range(4):
            result = bb.update(100.0 + i)
            results.append(result)
        
        assert all(r is None for r in results)
    
    def test_returns_tuple_after_warmup(self):
        """Test that update returns tuple after warm-up."""
        bb = BollingerBands(period=5)
        
        for i in range(4):
            bb.update(100.0 + i)
        
        result = bb.update(104.0)  # 5th price
        
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 4  # upper, middle, lower, bandwidth
    
    def test_returns_tuple_for_subsequent_updates(self):
        """Test that subsequent updates also return tuple."""
        bb = BollingerBands(period=5)
        
        for i in range(5):
            bb.update(100.0 + i)
        
        result1 = bb.update(105.0)
        result2 = bb.update(106.0)
        
        assert result1 is not None
        assert result2 is not None


class TestRealWorldScenario:
    """Test with more realistic price data."""
    
    def test_volatile_price_series(self):
        """Test with volatile price series."""
        bb = BollingerBands(period=20, std_dev=2.0)
        
        # Simulate volatile prices
        np.random.seed(42)
        base_price = 100.0
        
        prices = []
        for i in range(25):
            change = np.random.randn() * 2.0  # Random walk
            price = base_price + change
            prices.append(price)
            
            result = bb.update(price)
        
        assert bb.is_warmed_up()
        assert result is not None
        
        upper, middle, lower, bandwidth = result
        
        # Verify bands are in expected order
        assert lower < middle < upper
        assert bandwidth > 0
    
    def test_trending_market(self):
        """Test with trending market."""
        bb = BollingerBands(period=10)
        
        # Uptrend
        for i in range(15):
            bb.update(100.0 + (i * 2))  # Increasing prices
        
        assert bb.is_warmed_up()
        
        values = bb.get_values()
        
        # Bands should have expanded
        assert values.bandwidth > 0
        
        # Price should be near upper band in uptrend
        pct_b = bb.get_percent_b(128.0)  # Last price
        assert pct_b > 0.5  # Above middle


class TestSignalEdgeCases:
    """Test edge cases in signal detection."""
    
    def test_body_at_exact_lower_band(self):
        """Test body exactly at lower band boundary."""
        bb = BollingerBands(period=5)
        
        # Setup bands
        for price in [10.0, 11.0, 12.0, 13.0, 14.0]:
            bb.update(price)
        
        values = bb.get_values()
        
        # Body high exactly at lower band
        # Should NOT be considered below
        result = bb.is_body_below_lower_band(
            values.lower - 0.1,  # Slightly below
            values.lower - 0.05
        )
        
        assert result is True
        
        result = bb.is_body_below_lower_band(
            values.lower,  # Exactly at
            values.lower + 0.1  # Extends above
        )
        
        assert result is False  # High is not below
    
    def test_signal_with_wick_below_body_above(self):
        """Test signal when wick is below but body is above."""
        bb = BollingerBands(period=5)
        
        for price in [10.0, 11.0, 12.0, 13.0, 14.0]:
            bb.update(price)
        
        values = bb.get_values()
        
        # Wick goes below lower band, but body is above
        open_price = values.lower + 1.0
        close_price = values.lower + 1.5
        
        # Body is not below lower band
        assert bb.is_body_below_lower_band(open_price, close_price) is False
        
        # But price did go below
        assert bb.is_price_below_lower_band(values.lower - 0.5) is True


class TestComparisonWithTA-Lib:
    """Test results match TA-Lib when available."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("talib", reason="TA-Lib not available"),
        reason="TA-Lib not installed"
    )
    def test_matches_talib_results(self):
        """Test that our calculations match TA-Lib."""
        import talib
        
        bb = BollingerBands(period=10, std_dev=2.0)
        
        # Add prices
        prices = [100.0 + i * np.random.randn() for i in range(15)]
        
        for price in prices:
            bb.update(price)
        
        # Calculate with TA-Lib directly
        prices_array = np.array(prices[-10:], dtype=np.float64)
        upper, middle, lower = talib.BBANDS(
            prices_array,
            timeperiod=10,
            nbdevup=2.0,
            nbdevdn=2.0,
            matype=0
        )
        
        # Compare last values
        values = bb.get_values()
        
        assert abs(values.upper - upper[-1]) < 0.0001
        assert abs(values.middle - middle[-1]) < 0.0001
        assert abs(values.lower - lower[-1]) < 0.0001
