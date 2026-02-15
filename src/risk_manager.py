"""Risk management for position sizing and trade validation."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set

import structlog

from src.strategies.base import Signal, SignalType

logger = structlog.get_logger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    market_ticker: str
    side: str  # 'yes' or 'no'
    entry_price: float
    count: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = None
    unrealized_pnl: float = 0.0
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now(timezone.utc)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_position_size: int = 100  # Max contracts per position
    max_positions: int = 2  # Max concurrent positions
    max_exposure_pct: float = 0.20  # Max 20% of capital exposed
    max_daily_loss_pct: float = 0.10  # Max 10% daily loss
    max_risk_per_trade_pct: float = 0.02  # Max 2% risk per trade
    min_risk_reward_ratio: float = 1.5  # Minimum R:R ratio
    min_confidence: float = 0.6  # Minimum signal confidence


class RiskManager:
    """
    Manages trading risk including position sizing and trade validation.
    """
    
    def __init__(self, config: RiskConfig = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration (uses defaults if None)
        """
        self.config = config or RiskConfig()
        self._positions: Dict[str, Position] = {}
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset_date: date = date.today()
        self._trade_history: List[Dict] = []
        
        logger.info(
            "RiskManager initialized",
            max_positions=self.config.max_positions,
            max_daily_loss_pct=self.config.max_daily_loss_pct
        )
    
    def _check_daily_reset(self) -> None:
        """Reset daily stats if date changed."""
        today = date.today()
        if today != self._last_reset_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset_date = today
            logger.info("Daily risk stats reset", date=today.isoformat())
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_pct: float,
        stop_distance: float
    ) -> int:
        """
        Calculate position size based on risk parameters.
        
        Formula: position_size = (account_balance * risk_pct) / stop_distance
        
        Args:
            account_balance: Total account balance
            risk_pct: Percentage of account to risk (e.g., 0.02 for 2%)
            stop_distance: Distance to stop loss in price terms
            
        Returns:
            Position size in contracts (integer)
        """
        if stop_distance <= 0:
            logger.warning("Invalid stop distance", stop_distance=stop_distance)
            return 0
        
        # Calculate dollar risk amount
        dollar_risk = account_balance * risk_pct
        
        # Calculate position size
        # Each contract = $1 payout, priced in cents (0.01 to 0.99)
        # Risk per contract = stop_distance (in dollar terms)
        position_size = int(dollar_risk / stop_distance)
        
        # Apply limits
        position_size = min(position_size, self.config.max_position_size)
        position_size = max(position_size, 1)  # Minimum 1 contract
        
        logger.debug(
            "Position size calculated",
            account_balance=account_balance,
            risk_pct=risk_pct,
            stop_distance=stop_distance,
            position_size=position_size
        )
        
        return position_size
    
    def calculate_kelly_size(
        self,
        account_balance: float,
        win_probability: float,
        win_loss_ratio: float
    ) -> int:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = W - [(1 - W) / R]
        Where W = win probability, R = win/loss ratio
        
        Args:
            account_balance: Total account balance
            win_probability: Probability of winning (0.0 to 1.0)
            win_loss_ratio: Average win / average loss
            
        Returns:
            Position size in contracts
        """
        if win_probability <= 0 or win_loss_ratio <= 0:
            return 0
        
        # Kelly formula
        kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)
        
        # Use half Kelly for safety
        kelly_fraction = max(0, kelly_fraction * 0.5)
        
        # Calculate position size
        position_value = account_balance * kelly_fraction
        position_size = int(position_value / 0.50)  # Assume avg price of 0.50
        
        # Apply limits
        position_size = min(position_size, self.config.max_position_size)
        position_size = max(position_size, 1)
        
        logger.debug(
            "Kelly position size calculated",
            kelly_fraction=kelly_fraction,
            position_size=position_size
        )
        
        return position_size
    
    def validate_signal(
        self,
        signal: Signal,
        current_positions: Dict[str, Position],
        account_balance: float
    ) -> bool:
        """
        Validate if a signal should be executed.
        
        Args:
            signal: Trading signal to validate
            current_positions: Dictionary of open positions
            account_balance: Current account balance
            
        Returns:
            True if signal passes all risk checks
        """
        self._check_daily_reset()
        
        # Check daily loss limit
        if not self.check_daily_limits(self._daily_pnl, account_balance * self.config.max_daily_loss_pct):
            logger.warning("Daily loss limit reached, rejecting signal")
            return False
        
        # Check position count limit
        if len(current_positions) >= self.config.max_positions:
            logger.warning(
                "Max positions reached",
                current=len(current_positions),
                max=self.config.max_positions
            )
            return False
        
        # Check if already have position in this market
        if signal.market_ticker in current_positions:
            logger.warning(
                "Already have position in market",
                ticker=signal.market_ticker
            )
            return False
        
        # Check minimum confidence
        if signal.confidence < self.config.min_confidence:
            logger.warning(
                "Signal confidence too low",
                confidence=signal.confidence,
                min_required=self.config.min_confidence
            )
            return False
        
        # Check risk/reward ratio
        if signal.risk_reward_ratio < self.config.min_risk_reward_ratio:
            logger.warning(
                "Risk/reward ratio too low",
                ratio=signal.risk_reward_ratio,
                min_required=self.config.min_risk_reward_ratio
            )
            return False
        
        # Check exposure limit
        current_exposure = sum(
            pos.entry_price * pos.count
            for pos in current_positions.values()
        )
        max_exposure = account_balance * self.config.max_exposure_pct
        
        signal_exposure = signal.entry_price * self.calculate_position_size(
            account_balance,
            self.config.max_risk_per_trade_pct,
            signal.stop_distance
        )
        
        if current_exposure + signal_exposure > max_exposure:
            logger.warning(
                "Exposure limit would be exceeded",
                current=current_exposure,
                signal=signal_exposure,
                max=max_exposure
            )
            return False
        
        logger.info(
            "Signal validated",
            ticker=signal.market_ticker,
            type=signal.type.value,
            confidence=signal.confidence,
            rr_ratio=signal.risk_reward_ratio
        )
        
        return True
    
    def check_daily_limits(
        self,
        daily_pnl: float,
        max_loss: float
    ) -> bool:
        """
        Check if daily loss limit has been reached.
        
        Args:
            daily_pnl: Current daily P&L (negative = loss)
            max_loss: Maximum allowed daily loss
            
        Returns:
            True if within limits, False if limit reached
        """
        self._check_daily_reset()
        
        if daily_pnl < -max_loss:
            logger.warning(
                "Daily loss limit reached",
                daily_pnl=daily_pnl,
                max_loss=-max_loss
            )
            return False
        
        return True
    
    def can_open_position(
        self,
        ticker: str,
        current_positions: Dict[str, Position]
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Args:
            ticker: Market ticker
            current_positions: Current open positions
            
        Returns:
            Tuple of (can_open, reason)
        """
        self._check_daily_reset()
        
        if ticker in current_positions:
            return False, f"Already have position in {ticker}"
        
        if len(current_positions) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"
        
        return True, "OK"
    
    def add_position(self, position: Position) -> None:
        """Track a new position."""
        self._positions[position.market_ticker] = position
        self._daily_trades += 1
        
        logger.info(
            "Position added",
            ticker=position.market_ticker,
            side=position.side,
            size=position.count,
            entry=position.entry_price
        )
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_time: Optional[datetime] = None
    ) -> float:
        """
        Close a position and record P&L.
        
        Args:
            ticker: Market ticker
            exit_price: Exit price
            exit_time: Exit timestamp
            
        Returns:
            Realized P&L
        """
        if ticker not in self._positions:
            logger.warning("Position not found for closing", ticker=ticker)
            return 0.0
        
        position = self._positions.pop(ticker)
        
        if exit_time is None:
            exit_time = datetime.now(timezone.utc)
        
        # Calculate P&L
        if position.side == 'yes':
            pnl = (exit_price - position.entry_price) * position.count
        else:  # 'no'
            pnl = (position.entry_price - exit_price) * position.count
        
        self._daily_pnl += pnl
        
        # Record trade
        self._trade_history.append({
            'ticker': ticker,
            'side': position.side,
            'entry': position.entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
        })
        
        logger.info(
            "Position closed",
            ticker=ticker,
            pnl=pnl,
            daily_pnl=self._daily_pnl
        )
        
        return pnl
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position by ticker."""
        return self._positions.get(ticker)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return self._positions.copy()
    
    def get_stats(self) -> Dict:
        """Get risk manager statistics."""
        self._check_daily_reset()
        
        return {
            'open_positions': len(self._positions),
            'daily_pnl': self._daily_pnl,
            'daily_trades': self._daily_trades,
            'total_trades_today': len(self._trade_history),
            'max_positions': self.config.max_positions,
            'max_daily_loss_pct': self.config.max_daily_loss_pct,
            'positions': list(self._positions.keys()),
        }


class CircuitBreaker:
    """
    Circuit breaker for halting trading when conditions are met.
    """
    
    def __init__(
        self,
        consecutive_losses_limit: int = 3,
        daily_loss_limit_pct: float = 0.10
    ):
        self.consecutive_losses_limit = consecutive_losses_limit
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self._consecutive_losses = 0
        self._is_triggered = False
        self._trigger_reason: Optional[str] = None
    
    def record_trade(self, pnl: float) -> None:
        """Record trade result."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        # Check if circuit should trip
        if self._consecutive_losses >= self.consecutive_losses_limit:
            self._trigger(f"{self.consecutive_losses_limit} consecutive losses")
    
    def check_daily_loss(self, daily_pnl: float, account_balance: float) -> bool:
        """Check if daily loss limit is breached."""
        max_loss = account_balance * self.daily_loss_limit_pct
        
        if daily_pnl < -max_loss:
            self._trigger(f"Daily loss limit exceeded: {daily_pnl}")
            return False
        
        return True
    
    def _trigger(self, reason: str) -> None:
        """Trigger circuit breaker."""
        if not self._is_triggered:
            self._is_triggered = True
            self._trigger_reason = reason
            logger.error("CIRCUIT BREAKER TRIGGERED", reason=reason)
    
    def reset(self) -> None:
        """Reset circuit breaker."""
        self._is_triggered = False
        self._trigger_reason = None
        self._consecutive_losses = 0
        logger.info("Circuit breaker reset")
    
    @property
    def is_triggered(self) -> bool:
        """Check if circuit breaker is active."""
        return self._is_triggered
    
    @property
    def trigger_reason(self) -> Optional[str]:
        """Get reason for trigger."""
        return self._trigger_reason
