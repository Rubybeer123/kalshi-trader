"""Tests for strategy base class and risk manager."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.candle_aggregator import OHLCV, Tick
from src.strategies.base import Signal, SignalType, Strategy, StrategyError
from src.risk_manager import (
    CircuitBreaker,
    Position,
    RiskConfig,
    RiskManager,
)


class TestSignalType:
    """Test SignalType enum."""
    
    def test_values(self):
        """Test enum values."""
        assert SignalType.LONG.value == "long"
        assert SignalType.SHORT.value == "short"
        assert SignalType.EXIT.value == "exit"


class TestSignal:
    """Test Signal dataclass."""
    
    def test_creation(self):
        """Test Signal creation."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85,
            confidence=0.75
        )
        
        assert signal.type == SignalType.LONG
        assert signal.market_ticker == "KXBTC15M-26FEB150330-30"
        assert signal.entry_price == 0.65
        assert signal.confidence == 0.75
    
    def test_confidence_validation(self):
        """Test confidence must be 0-1."""
        with pytest.raises(ValueError):
            Signal(
                type=SignalType.LONG,
                market_ticker="KXBTC15M-26FEB150330-30",
                entry_price=0.65,
                stop_loss=0.55,
                take_profit=0.85,
                confidence=1.5  # Invalid
            )
    
    def test_risk_reward_ratio_long(self):
        """Test R:R calculation for long."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85
        )
        
        # Risk = 0.65 - 0.55 = 0.10
        # Reward = 0.85 - 0.65 = 0.20
        # R:R = 0.20 / 0.10 = 2.0
        assert signal.risk_reward_ratio == 2.0
    
    def test_risk_reward_ratio_short(self):
        """Test R:R calculation for short."""
        signal = Signal(
            type=SignalType.SHORT,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.75,
            take_profit=0.45
        )
        
        # Risk = 0.75 - 0.65 = 0.10
        # Reward = 0.65 - 0.45 = 0.20
        # R:R = 0.20 / 0.10 = 2.0
        assert signal.risk_reward_ratio == 2.0
    
    def test_stop_distance_long(self):
        """Test stop distance for long."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85
        )
        
        assert signal.stop_distance == 0.10
    
    def test_stop_distance_short(self):
        """Test stop distance for short."""
        signal = Signal(
            type=SignalType.SHORT,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.75,
            take_profit=0.45
        )
        
        assert signal.stop_distance == 0.10
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85,
            confidence=0.75,
            metadata={'strategy': 'test'}
        )
        
        d = signal.to_dict()
        
        assert d['type'] == 'long'
        assert d['market_ticker'] == 'KXBTC15M-26FEB150330-30'
        assert d['confidence'] == 0.75
        assert d['risk_reward_ratio'] == 2.0


class ConcreteStrategy(Strategy):
    """Concrete implementation for testing."""
    
    @property
    def name(self) -> str:
        return "TestStrategy"
    
    def get_required_warmup(self) -> int:
        return 10
    
    def on_candle(self, market_ticker: str, candle: OHLCV):
        return None
    
    def on_tick(self, market_ticker: str, tick: Tick):
        return None


class TestStrategyABC:
    """Test Strategy abstract base class."""
    
    def test_name_property(self):
        """Test name property."""
        strategy = ConcreteStrategy()
        assert strategy.name == "TestStrategy"
    
    def test_get_required_warmup(self):
        """Test warmup requirement."""
        strategy = ConcreteStrategy()
        assert strategy.get_required_warmup() == 10
    
    def test_can_trade_true(self):
        """Test can_trade returns True with enough candles."""
        strategy = ConcreteStrategy()
        assert strategy.can_trade(10) is True
        assert strategy.can_trade(15) is True
    
    def test_can_trade_false(self):
        """Test can_trade returns False with insufficient candles."""
        strategy = ConcreteStrategy()
        assert strategy.can_trade(5) is False
        assert strategy.can_trade(9) is False
    
    def test_repr(self):
        """Test string representation."""
        strategy = ConcreteStrategy()
        assert "TestStrategy" in repr(strategy)


class TestRiskConfig:
    """Test RiskConfig dataclass."""
    
    def test_defaults(self):
        """Test default values."""
        config = RiskConfig()
        
        assert config.max_position_size == 100
        assert config.max_positions == 2
        assert config.max_exposure_pct == 0.20
        assert config.max_daily_loss_pct == 0.10
        assert config.max_risk_per_trade_pct == 0.02
        assert config.min_risk_reward_ratio == 1.5
        assert config.min_confidence == 0.6
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = RiskConfig(
            max_position_size=50,
            max_positions=5,
            max_daily_loss_pct=0.05
        )
        
        assert config.max_position_size == 50
        assert config.max_positions == 5
        assert config.max_daily_loss_pct == 0.05


class TestPosition:
    """Test Position dataclass."""
    
    def test_creation(self):
        """Test Position creation."""
        position = Position(
            market_ticker="KXBTC15M-26FEB150330-30",
            side="yes",
            entry_price=0.65,
            count=10
        )
        
        assert position.market_ticker == "KXBTC15M-26FEB150330-30"
        assert position.side == "yes"
        assert position.count == 10
        assert position.entry_time is not None


class TestRiskManagerPositionSizing:
    """Test RiskManager position sizing."""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager()
    
    def test_calculate_position_size_basic(self, risk_manager):
        """Test basic position size calculation."""
        # $1000 account, 2% risk, $0.10 stop distance
        # Risk amount = $1000 * 0.02 = $20
        # Position size = $20 / $0.10 = 200 contracts
        size = risk_manager.calculate_position_size(1000.0, 0.02, 0.10)
        
        assert size == 200
    
    def test_calculate_position_size_respects_max(self, risk_manager):
        """Test position size respects max limit."""
        # Large account would suggest more than max
        size = risk_manager.calculate_position_size(100000.0, 0.02, 0.01)
        
        assert size <= risk_manager.config.max_position_size
    
    def test_calculate_position_size_minimum_one(self, risk_manager):
        """Test minimum position size of 1."""
        # Very small risk or large stop
        size = risk_manager.calculate_position_size(100.0, 0.01, 1.0)
        
        assert size >= 1
    
    def test_calculate_position_size_zero_stop(self, risk_manager):
        """Test zero stop distance returns 0."""
        size = risk_manager.calculate_position_size(1000.0, 0.02, 0.0)
        
        assert size == 0
    
    def test_calculate_kelly_size(self, risk_manager):
        """Test Kelly Criterion sizing."""
        # 60% win rate, 2:1 win/loss ratio
        # Kelly = 0.60 - (0.40 / 2.0) = 0.40
        # Half Kelly = 0.20
        size = risk_manager.calculate_kelly_size(1000.0, 0.60, 2.0)
        
        assert size > 0
        assert size <= risk_manager.config.max_position_size


class TestRiskManagerValidation:
    """Test RiskManager signal validation."""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager()
    
    @pytest.fixture
    def valid_signal(self):
        return Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85,
            confidence=0.75
        )
    
    def test_valid_signal_passes(self, risk_manager, valid_signal):
        """Test valid signal passes validation."""
        result = risk_manager.validate_signal(
            valid_signal,
            {},  # No current positions
            1000.0
        )
        
        assert result is True
    
    def test_signal_fails_low_confidence(self, risk_manager, valid_signal):
        """Test signal fails with low confidence."""
        low_conf_signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85,
            confidence=0.3  # Below 0.6 minimum
        )
        
        result = risk_manager.validate_signal(
            low_conf_signal,
            {},
            1000.0
        )
        
        assert result is False
    
    def test_signal_fails_low_rr_ratio(self, risk_manager):
        """Test signal fails with low R:R ratio."""
        poor_rr_signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC15M-26FEB150330-30",
            entry_price=0.65,
            stop_loss=0.60,  # Close stop = low R:R
            take_profit=0.67,
            confidence=0.75
        )
        
        result = risk_manager.validate_signal(
            poor_rr_signal,
            {},
            1000.0
        )
        
        assert result is False
    
    def test_signal_fails_max_positions(self, risk_manager, valid_signal):
        """Test signal fails when max positions reached."""
        # Fill positions to limit
        current_positions = {
            f"TICKER{i}": Position(f"TICKER{i}", "yes", 0.50, 10)
            for i in range(risk_manager.config.max_positions)
        }
        
        result = risk_manager.validate_signal(
            valid_signal,
            current_positions,
            1000.0
        )
        
        assert result is False
    
    def test_signal_fails_duplicate_market(self, risk_manager, valid_signal):
        """Test signal fails for market with existing position."""
        current_positions = {
            "KXBTC15M-26FEB150330-30": Position(
                "KXBTC15M-26FEB150330-30", "yes", 0.50, 10
            )
        }
        
        result = risk_manager.validate_signal(
            valid_signal,
            current_positions,
            1000.0
        )
        
        assert result is False
    
    def test_signal_fails_daily_loss_limit(self, risk_manager, valid_signal):
        """Test signal fails when daily loss limit reached."""
        # Set daily loss beyond limit
        risk_manager._daily_pnl = -150.0  # Beyond 10% of $1000
        
        result = risk_manager.validate_signal(
            valid_signal,
            {},
            1000.0
        )
        
        assert result is False


class TestRiskManagerDailyLimits:
    """Test daily limits checking."""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager()
    
    def test_within_limits(self, risk_manager):
        """Test within daily loss limit."""
        result = risk_manager.check_daily_limits(-50.0, 100.0)
        
        assert result is True
    
    def test_at_limit(self, risk_manager):
        """Test exactly at daily loss limit."""
        result = risk_manager.check_daily_limits(-100.0, 100.0)
        
        assert result is True  # Equal is still within
    
    def test_beyond_limit(self, risk_manager):
        """Test beyond daily loss limit."""
        result = risk_manager.check_daily_limits(-150.0, 100.0)
        
        assert result is False
    
    def test_positive_pnl_always_ok(self, risk_manager):
        """Test positive P&L always passes."""
        result = risk_manager.check_daily_limits(500.0, 100.0)
        
        assert result is True


class TestRiskManagerPositionTracking:
    """Test position tracking."""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager()
    
    def test_add_position(self, risk_manager):
        """Test adding a position."""
        position = Position(
            market_ticker="KXBTC15M-26FEB150330-30",
            side="yes",
            entry_price=0.65,
            count=10
        )
        
        risk_manager.add_position(position)
        
        assert "KXBTC15M-26FEB150330-30" in risk_manager.get_all_positions()
    
    def test_close_position_long(self, risk_manager):
        """Test closing long position."""
        position = Position(
            market_ticker="KXBTC15M-26FEB150330-30",
            side="yes",
            entry_price=0.60,
            count=10
        )
        risk_manager.add_position(position)
        
        # Close at higher price = profit
        pnl = risk_manager.close_position("KXBTC15M-26FEB150330-30", 0.70)
        
        # P&L = (0.70 - 0.60) * 10 = $1.00
        assert pnl == 1.0
        assert "KXBTC15M-26FEB150330-30" not in risk_manager.get_all_positions()
    
    def test_close_position_short(self, risk_manager):
        """Test closing short position."""
        position = Position(
            market_ticker="KXBTC15M-26FEB150330-30",
            side="no",
            entry_price=0.70,
            count=10
        )
        risk_manager.add_position(position)
        
        # Close at lower price = profit for short
        pnl = risk_manager.close_position("KXBTC15M-26FEB150330-30", 0.60)
        
        # P&L = (0.70 - 0.60) * 10 = $1.00
        assert pnl == 1.0
    
    def test_close_position_not_found(self, risk_manager):
        """Test closing non-existent position."""
        pnl = risk_manager.close_position("UNKNOWN", 0.50)
        
        assert pnl == 0.0
    
    def test_get_position(self, risk_manager):
        """Test getting position by ticker."""
        position = Position(
            market_ticker="KXBTC15M-26FEB150330-30",
            side="yes",
            entry_price=0.65,
            count=10
        )
        risk_manager.add_position(position)
        
        retrieved = risk_manager.get_position("KXBTC15M-26FEB150330-30")
        
        assert retrieved == position
    
    def test_get_position_not_found(self, risk_manager):
        """Test getting non-existent position."""
        retrieved = risk_manager.get_position("UNKNOWN")
        
        assert retrieved is None


class TestRiskManagerStats:
    """Test statistics reporting."""
    
    def test_get_stats(self):
        """Test stats reporting."""
        risk_manager = RiskManager()
        
        stats = risk_manager.get_stats()
        
        assert 'open_positions' in stats
        assert 'daily_pnl' in stats
        assert 'daily_trades' in stats
        assert 'max_positions' in stats


class TestCircuitBreaker:
    """Test CircuitBreaker."""
    
    @pytest.fixture
    def circuit(self):
        return CircuitBreaker(
            consecutive_losses_limit=3,
            daily_loss_limit_pct=0.10
        )
    
    def test_initial_state(self, circuit):
        """Test initial state."""
        assert circuit.is_triggered is False
        assert circuit.trigger_reason is None
    
    def test_triggers_on_consecutive_losses(self, circuit):
        """Test trigger on consecutive losses."""
        circuit.record_trade(-1.0)
        assert not circuit.is_triggered
        
        circuit.record_trade(-1.0)
        assert not circuit.is_triggered
        
        circuit.record_trade(-1.0)
        assert circuit.is_triggered
        assert "consecutive losses" in circuit.trigger_reason
    
    def test_resets_on_win(self, circuit):
        """Test consecutive loss counter resets on win."""
        circuit.record_trade(-1.0)
        circuit.record_trade(-1.0)
        
        circuit.record_trade(1.0)  # Win resets counter
        
        assert not circuit.is_triggered
        circuit.record_trade(-1.0)  # Only 1 loss now
        assert not circuit.is_triggered
    
    def test_triggers_on_daily_loss(self, circuit):
        """Test trigger on daily loss limit."""
        result = circuit.check_daily_loss(-150.0, 1000.0)  # 15% loss
        
        assert result is False
        assert circuit.is_triggered
        assert "Daily loss limit" in circuit.trigger_reason
    
    def test_no_trigger_within_limits(self, circuit):
        """Test no trigger within limits."""
        result = circuit.check_daily_loss(-50.0, 1000.0)  # 5% loss
        
        assert result is True
        assert not circuit.is_triggered
    
    def test_reset(self, circuit):
        """Test circuit breaker reset."""
        circuit.record_trade(-1.0)
        circuit.record_trade(-1.0)
        circuit.record_trade(-1.0)
        
        assert circuit.is_triggered
        
        circuit.reset()
        
        assert not circuit.is_triggered
        assert circuit.trigger_reason is None


class TestCanOpenPosition:
    """Test can_open_position method."""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager()
    
    def test_can_open_when_empty(self, risk_manager):
        """Test can open when no positions."""
        can_open, reason = risk_manager.can_open_position(
            "KXBTC15M-26FEB150330-30",
            {}
        )
        
        assert can_open is True
        assert reason == "OK"
    
    def test_cannot_open_duplicate(self, risk_manager):
        """Test cannot open duplicate position."""
        positions = {
            "KXBTC15M-26FEB150330-30": Position(
                "KXBTC15M-26FEB150330-30", "yes", 0.50, 10
            )
        }
        
        can_open, reason = risk_manager.can_open_position(
            "KXBTC15M-26FEB150330-30",
            positions
        )
        
        assert can_open is False
        assert "Already have position" in reason
    
    def test_cannot_open_at_max(self, risk_manager):
        """Test cannot open at max positions."""
        # Fill to max
        positions = {
            f"TICKER{i}": Position(f"TICKER{i}", "yes", 0.50, 10)
            for i in range(risk_manager.config.max_positions)
        }
        
        can_open, reason = risk_manager.can_open_position(
            "NEW-TICKER",
            positions
        )
        
        assert can_open is False
        assert "Max positions" in reason


class TestStrategyInterfaceCompliance:
    """Test that concrete strategies implement interface correctly."""
    
    def test_must_implement_name(self):
        """Test that name property must be implemented."""
        class BadStrategy(Strategy):
            def on_candle(self, market_ticker, candle):
                return None
            def on_tick(self, market_ticker, tick):
                return None
            def get_required_warmup(self):
                return 10
            # Missing name property
        
        with pytest.raises(TypeError):
            BadStrategy()
    
    def test_must_implement_on_candle(self):
        """Test that on_candle must be implemented."""
        class BadStrategy(Strategy):
            @property
            def name(self):
                return "Bad"
            def on_tick(self, market_ticker, tick):
                return None
            def get_required_warmup(self):
                return 10
            # Missing on_candle
        
        with pytest.raises(TypeError):
            BadStrategy()
    
    def test_must_implement_on_tick(self):
        """Test that on_tick must be implemented."""
        class BadStrategy(Strategy):
            @property
            def name(self):
                return "Bad"
            def on_candle(self, market_ticker, candle):
                return None
            def get_required_warmup(self):
                return 10
            # Missing on_tick
        
        with pytest.raises(TypeError):
            BadStrategy()
    
    def test_must_implement_get_required_warmup(self):
        """Test that get_required_warmup must be implemented."""
        class BadStrategy(Strategy):
            @property
            def name(self):
                return "Bad"
            def on_candle(self, market_ticker, candle):
                return None
            def on_tick(self, market_ticker, tick):
                return None
            # Missing get_required_warmup
        
        with pytest.raises(TypeError):
            BadStrategy()
    
    def test_complete_strategy_works(self):
        """Test that complete strategy instantiation works."""
        class GoodStrategy(Strategy):
            @property
            def name(self):
                return "GoodStrategy"
            
            def get_required_warmup(self):
                return 20
            
            def on_candle(self, market_ticker, candle):
                return None
            
            def on_tick(self, market_ticker, tick):
                return None
        
        strategy = GoodStrategy()
        
        assert strategy.name == "GoodStrategy"
        assert strategy.get_required_warmup() == 20
        assert strategy.can_trade(20)
        assert not strategy.can_trade(19)
