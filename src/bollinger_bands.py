"""Bollinger Bands indicator using TA-Lib."""

import math
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import structlog

try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    import warnings
    warnings.warn("TA-Lib not available, using pure Python implementation")

logger = structlog.get_logger(__name__)


@dataclass
class BollingerValues:
    """Container for Bollinger Bands values."""
    upper: float
    middle: float  # SMA
    lower: float
    bandwidth: float  # % of price
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'upper': self.upper,
            'middle': self.middle,
            'lower': self.lower,
            'bandwidth': self.bandwidth,
        }


class BollingerBands:
    """
    Bollinger Bands indicator with incremental updates.
    
    Uses TA-Lib for calculations when available, falls back to
    pure Python implementation.
    """
    
    def __init__(self, period: int = 25, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: Number of periods for SMA (default: 25)
            std_dev: Standard deviation multiplier (default: 2.0)
        """
        self.period = period
        self.std_dev = std_dev
        
        # Price history for incremental calculation
        self._prices: Deque[float] = deque(maxlen=period)
        
        # Current values
        self._upper: Optional[float] = None
        self._middle: Optional[float] = None
        self._lower: Optional[float] = None
        self._bandwidth: Optional[float] = None
        
        # TA-Lib availability
        self._use_talib = TA_LIB_AVAILABLE
        
        logger.debug(
            "BollingerBands initialized",
            period=period,
            std_dev=std_dev,
            using_talib=self._use_talib
        )
    
    def update(self, close_price: float) -> Optional[Tuple[float, float, float, float]]:
        """
        Update indicator with new close price.
        
        Args:
            close_price: New closing price
            
        Returns:
            Tuple of (upper, middle, lower, bandwidth) or None if not warmed up
        """
        # Add price to history
        self._prices.append(close_price)
        
        # Check if we have enough data
        if len(self._prices) < self.period:
            return None
        
        # Calculate bands
        if self._use_talib:
            upper, middle, lower = self._calculate_talib()
        else:
            upper, middle, lower = self._calculate_pure_python()
        
        # Calculate bandwidth as % of middle band
        if middle > 0:
            bandwidth = ((upper - lower) / middle) * 100.0
        else:
            bandwidth = 0.0
        
        # Store values
        self._upper = upper
        self._middle = middle
        self._lower = lower
        self._bandwidth = bandwidth
        
        return (upper, middle, lower, bandwidth)
    
    def _calculate_talib(self) -> Tuple[float, float, float]:
        """Calculate bands using TA-Lib."""
        prices = np.array(list(self._prices), dtype=np.float64)
        
        upper, middle, lower = talib.BBANDS(
            prices,
            timeperiod=self.period,
            nbdevup=self.std_dev,
            nbdevdn=self.std_dev,
            matype=0  # SMA
        )
        
        # Return last values
        return float(upper[-1]), float(middle[-1]), float(lower[-1])
    
    def _calculate_pure_python(self) -> Tuple[float, float, float]:
        """Calculate bands using pure Python (fallback)."""
        prices = list(self._prices)
        n = len(prices)
        
        # Calculate SMA (middle band)
        middle = sum(prices) / n
        
        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in prices) / n
        std = math.sqrt(variance)
        
        # Calculate bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)
        
        return upper, middle, lower
    
    def is_warmed_up(self) -> bool:
        """
        Check if indicator has enough data to be valid.
        
        Returns:
            True if we have at least 'period' data points
        """
        return len(self._prices) >= self.period
    
    def get_last_values(self) -> Optional[Dict[str, float]]:
        """
        Get last calculated values.
        
        Returns:
            Dictionary with upper, middle, lower, bandwidth or None
        """
        if self._upper is None:
            return None
        
        return {
            'upper': self._upper,
            'middle': self._middle,
            'lower': self._lower,
            'bandwidth': self._bandwidth,
        }
    
    def get_values(self) -> Optional[BollingerValues]:
        """
        Get last calculated values as BollingerValues object.
        
        Returns:
            BollingerValues object or None
        """
        if self._upper is None:
            return None
        
        return BollingerValues(
            upper=self._upper,
            middle=self._middle,
            lower=self._lower,
            bandwidth=self._bandwidth
        )
    
    # === Signal Detection Methods ===
    
    def is_price_below_lower_band(self, price: float) -> bool:
        """
        Check if price is below the lower band.
        
        Args:
            price: Price to check
            
        Returns:
            True if price is below lower band
        """
        if self._lower is None:
            return False
        return price < self._lower
    
    def is_price_above_upper_band(self, price: float) -> bool:
        """
        Check if price is above the upper band.
        
        Args:
            price: Price to check
            
        Returns:
            True if price is above upper band
        """
        if self._upper is None:
            return False
        return price > self._upper
    
    def is_price_inside_bands(self, price: float) -> bool:
        """
        Check if price is inside the bands.
        
        Args:
            price: Price to check
            
        Returns:
            True if price is between lower and upper bands
        """
        if self._lower is None or self._upper is None:
            return False
        return self._lower <= price <= self._upper
    
    def is_body_below_lower_band(self, open_price: float, close_price: float) -> bool:
        """
        Check if entire candle body is below lower band.
        This is a key signal for potential reversal.
        
        Args:
            open_price: Candle open price
            close_price: Candle close price
            
        Returns:
            True if body is below lower band
        """
        if self._lower is None:
            return False
        
        body_high = max(open_price, close_price)
        return body_high < self._lower
    
    def is_body_above_upper_band(self, open_price: float, close_price: float) -> bool:
        """
        Check if entire candle body is above upper band.
        This is a key signal for potential reversal.
        
        Args:
            open_price: Candle open price
            close_price: Candle close price
            
        Returns:
            True if body is above upper band
        """
        if self._upper is None:
            return False
        
        body_low = min(open_price, close_price)
        return body_low > self._upper
    
    def get_percent_b(self, price: float) -> Optional[float]:
        """
        Calculate %B indicator (position within bands).
        
        %B = (Price - Lower) / (Upper - Lower)
        
        Args:
            price: Current price
            
        Returns:
            %B value (0.0 = at lower, 1.0 = at upper, < 0 = below, > 1 = above)
        """
        if self._upper is None or self._lower is None:
            return None
        
        band_range = self._upper - self._lower
        if band_range == 0:
            return 0.5
        
        return (price - self._lower) / band_range
    
    def get_signal(self, open_price: float, close_price: float) -> Optional[str]:
        """
        Generate trading signal based on candle position relative to bands.
        
        Args:
            open_price: Candle open price
            close_price: Candle close price
            
        Returns:
            Signal string: 'oversold', 'overbought', or None
        """
        if not self.is_warmed_up():
            return None
        
        # Check for body below lower band (oversold)
        if self.is_body_below_lower_band(open_price, close_price):
            return 'oversold'
        
        # Check for body above upper band (overbought)
        if self.is_body_above_upper_band(open_price, close_price):
            return 'overbought'
        
        return None
    
    def reset(self) -> None:
        """Reset the indicator, clearing all history."""
        self._prices.clear()
        self._upper = None
        self._middle = None
        self._lower = None
        self._bandwidth = None
        
        logger.debug("BollingerBands reset")
    
    def get_stats(self) -> Dict[str, any]:
        """Get indicator statistics."""
        return {
            'period': self.period,
            'std_dev': self.std_dev,
            'data_points': len(self._prices),
            'warmed_up': self.is_warmed_up(),
            'using_talib': self._use_talib,
            'current_values': self.get_last_values(),
        }
