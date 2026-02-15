"""Bollinger Scalper strategy implementation."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import structlog

from src.bollinger_bands import BollingerBands
from src.candle_aggregator import OHLCV
from src.strategies.base import Signal, SignalType, Strategy

logger = structlog.get_logger(__name__)


@dataclass
class ScalperConfig:
    """Configuration for BollingerScalper."""
    bb_period: int = 25
    bb_std: float = 2.0
    min_rr: float = 2.0
    min_time_to_expiry: timedelta = timedelta(minutes=8)
    use_adx_filter: bool = False  # Optional for MVP
    adx_threshold: float = 20.0
    use_rsi_confirmation: bool = False  # Optional for MVP


class BollingerScalper(Strategy):
    """
    Bollinger Bands scalping strategy.
    
    Enters when candle body is entirely outside bands (potential reversal).
    Targets 1:2 risk-reward minimum.
    """
    
    # Kalshi fee constants
    KALSHI_FEE_RATE = 0.07
    
    def __init__(self, config: Optional[ScalperConfig] = None):
        """
        Initialize BollingerScalper.
        
        Args:
            config: Strategy configuration
        """
        self.config = config or ScalperConfig()
        self._bands: Dict[str, BollingerBands] = {}
        self._last_signal_time: Dict[str, datetime] = {}
        
        logger.info(
            "BollingerScalper initialized",
            bb_period=self.config.bb_period,
            bb_std=self.config.bb_std,
            min_rr=self.config.min_rr
        )
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "BollingerScalper"
    
    def get_required_warmup(self) -> int:
        """Need bb_period candles to warm up Bollinger Bands."""
        return self.config.bb_period
    
    def _get_bands(self, ticker: str) -> BollingerBands:
        """Get or create BollingerBands for ticker."""
        if ticker not in self._bands:
            self._bands[ticker] = BollingerBands(
                period=self.config.bb_period,
                std_dev=self.config.bb_std
            )
        return self._bands[ticker]
    
    def _check_time_to_expiry(self, candle: OHLCV) -> bool:
        """
        Check if there's enough time before market expiration.
        
        Args:
            candle: Current candle with market info
            
        Returns:
            True if safe to trade (> 8 minutes to expiry)
        """
        # This would need market expiration info
        # For now, assume candle has time_remaining or we track it
        # MVP: Always True, implement with market data later
        return True
    
    def _calculate_fees(self, contracts: int, price: float) -> float:
        """
        Calculate Kalshi trading fees.
        
        Fee formula: 0.07 * C * P * (1-P)
        Where C = contracts, P = price
        
        Args:
            contracts: Number of contracts
            price: Entry price (0.01 to 0.99)
            
        Returns:
            Fee amount
        """
        return self.KALSHI_FEE_RATE * contracts * price * (1 - price)
    
    def _adjust_for_fees(
        self,
        entry: float,
        stop: float,
        target: float,
        contracts: int
    ) -> tuple[float, float, float]:
        """
        Adjust prices to account for fees while maintaining minimum R:R.
        
        Args:
            entry: Entry price
            stop: Stop loss price
            target: Target price
            contracts: Position size
            
        Returns:
            Adjusted (entry, stop, target)
        """
        # Calculate entry fee impact
        entry_fee = self._calculate_fees(contracts, entry)
        fee_per_contract = entry_fee / contracts
        
        # Adjust entry to account for fee (worse price)
        if target > entry:  # Long
            adjusted_entry = entry + fee_per_contract
            # Keep same stop, adjust target to maintain R:R
            risk = adjusted_entry - stop
            min_reward = risk * self.config.min_rr
            adjusted_target = adjusted_entry + min_reward
        else:  # Short
            adjusted_entry = entry - fee_per_contract
            risk = stop - adjusted_entry
            min_reward = risk * self.config.min_rr
            adjusted_target = adjusted_entry - min_reward
        
        return adjusted_entry, stop, adjusted_target
    
    def on_candle(self, market_ticker: str, candle: OHLCV) -> Optional[Signal]:
        """
        Process completed candle and generate signal if conditions met.
        
        Args:
            market_ticker: Market identifier
            candle: Completed OHLCV candle
            
        Returns:
            Signal if strategy triggers, None otherwise
        """
        bands = self._get_bands(market_ticker)
        
        # Update bands with new close
        bands.update(candle.close)
        
        # Check warmup
        if not bands.is_warmed_up():
            return None
        
        # Check time to expiry filter
        if not self._check_time_to_expiry(candle):
            logger.debug("Rejected: insufficient time to expiry", ticker=market_ticker)
            return None
        
        # Get current band values
        band_values = bands.get_values()
        if band_values is None:
            return None
        
        upper = band_values.upper
        lower = band_values.lower
        middle = band_values.middle
        
        # Calculate candle body extremes
        body_high = max(candle.open, candle.close)
        body_low = min(candle.open, candle.close)
        
        signal = None
        
        # LONG condition: Body entirely below lower band
        if body_high < lower:
            signal = self._create_long_signal(
                market_ticker, candle, lower, upper
            )
        
        # SHORT condition: Body entirely above upper band
        elif body_low > upper:
            signal = self._create_short_signal(
                market_ticker, candle, upper, lower
            )
        
        if signal:
            # Log the signal
            logger.info(
                "Signal generated",
                strategy=self.name,
                ticker=market_ticker,
                type=signal.type.value,
                entry=signal.entry_price,
                stop=signal.stop_loss,
                target=signal.take_profit,
                rr=signal.risk_reward_ratio,
                confidence=signal.confidence
            )
            
            # Track signal time to prevent spam
            self._last_signal_time[market_ticker] = datetime.now(timezone.utc)
        
        return signal
    
    def _create_long_signal(
        self,
        ticker: str,
        candle: OHLCV,
        lower: float,
        upper: float
    ) -> Optional[Signal]:
        """
        Create long signal.
        
        Entry: Candle low
        Stop: Candle high
        Target: Entry + 2 * (entry - stop)
        """
        entry = candle.low
        stop = candle.high
        
        # Calculate target for 1:2 R:R
        risk = entry - stop  # Negative for long, but we use absolute
        target = entry + (2.0 * abs(risk))
        
        # Ensure target is reasonable (not above middle band + buffer)
        max_target = upper - 0.01
        target = min(target, max_target)
        
        # Recalculate R:R after target adjustment
        actual_risk = entry - stop
        actual_reward = target - entry
        
        if actual_risk == 0:
            return None
        
        rr_ratio = actual_reward / actual_risk
        
        # Check minimum R:R
        if rr_ratio < self.config.min_rr:
            logger.debug(
                "Long signal rejected: insufficient R:R",
                ticker=ticker,
                rr=rr_ratio,
                min_required=self.config.min_rr
            )
            return None
        
        # Calculate confidence based on how far body is below band
        body_high = max(candle.open, candle.close)
        band_distance = lower - body_high
        confidence = min(0.95, 0.6 + (band_distance * 2))
        
        return Signal(
            type=SignalType.LONG,
            market_ticker=ticker,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'bb_upper': upper,
                'bb_lower': lower,
                'bb_middle': candle.close,  # approximate
                'candle_open': candle.open,
                'candle_high': candle.high,
                'candle_low': candle.low,
                'candle_close': candle.close,
                'candle_volume': candle.volume,
                'rr_ratio': rr_ratio,
            }
        )
    
    def _create_short_signal(
        self,
        ticker: str,
        candle: OHLCV,
        upper: float,
        lower: float
    ) -> Optional[Signal]:
        """
        Create short signal.
        
        Entry: Candle high
        Stop: Candle low
        Target: Entry - 2 * (stop - entry)
        """
        entry = candle.high
        stop = candle.low
        
        # Calculate target for 1:2 R:R
        risk = stop - entry  # For short
        target = entry - (2.0 * risk)
        
        # Ensure target is reasonable (not below lower band + buffer)
        min_target = lower + 0.01
        target = max(target, min_target)
        
        # Recalculate R:R after target adjustment
        actual_risk = stop - entry
        actual_reward = entry - target
        
        if actual_risk == 0:
            return None
        
        rr_ratio = actual_reward / actual_risk
        
        # Check minimum R:R
        if rr_ratio < self.config.min_rr:
            logger.debug(
                "Short signal rejected: insufficient R:R",
                ticker=ticker,
                rr=rr_ratio,
                min_required=self.config.min_rr
            )
            return None
        
        # Calculate confidence based on how far body is above band
        body_low = min(candle.open, candle.close)
        band_distance = body_low - upper
        confidence = min(0.95, 0.6 + (band_distance * 2))
        
        return Signal(
            type=SignalType.SHORT,
            market_ticker=ticker,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'bb_upper': upper,
                'bb_lower': lower,
                'bb_middle': candle.close,  # approximate
                'candle_open': candle.open,
                'candle_high': candle.high,
                'candle_low': candle.low,
                'candle_close': candle.close,
                'candle_volume': candle.volume,
                'rr_ratio': rr_ratio,
            }
        )
    
    def on_tick(self, market_ticker: str, tick) -> Optional[Signal]:
        """
        Process real-time tick.
        
        For this strategy, we trade on completed candles only.
        Ticks are used to update current price but don't generate signals.
        
        Args:
            market_ticker: Market identifier
            tick: Real-time tick data
            
        Returns:
            None (signals only on candle close)
        """
        # This strategy only trades on completed candles
        return None
    
    def get_band_values(self, ticker: str) -> Optional[Dict]:
        """Get current Bollinger Bands values for a ticker."""
        if ticker not in self._bands:
            return None
        
        bands = self._bands[ticker]
        if not bands.is_warmed_up():
            return None
        
        return bands.get_last_values()
    
    def reset_ticker(self, ticker: str) -> None:
        """Reset bands for a specific ticker."""
        if ticker in self._bands:
            del self._bands[ticker]
        if ticker in self._last_signal_time:
            del self._last_signal_time[ticker]
