"""Tests for ExecutionEngine."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.models import Order, OrderStatus
from src.execution_engine import (
    ExecutionEngine,
    OrderResult,
    OrderState,
    TrackedOrder,
)
from src.risk_manager import RiskManager, RiskConfig
from src.strategies.base import Signal, SignalType


class TestOrderState:
    """Test OrderState enum."""
    
    def test_values(self):
        """Test enum values."""
        assert OrderState.PENDING.value == "pending"
        assert OrderState.OPEN.value == "open"
        assert OrderState.FILLED.value == "filled"
        assert OrderState.CANCELLED.value == "cancelled"


class TestTrackedOrder:
    """Test TrackedOrder dataclass."""
    
    def test_creation(self):
        """Test TrackedOrder creation."""
        order = TrackedOrder(
            client_order_id="TEST_123",
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10
        )
        
        assert order.client_order_id == "TEST_123"
        assert order.ticker == "KXBTC"
        assert order.state == OrderState.PENDING
        assert order.filled_count == 0


class TestOrderResult:
    """Test OrderResult dataclass."""
    
    def test_success(self):
        """Test successful result."""
        result = OrderResult(
            success=True,
            order_id="ORDER_123"
        )
        
        assert result.success is True
        assert result.order_id == "ORDER_123"
    
    def test_failure(self):
        """Test failed result."""
        result = OrderResult(
            success=False,
            error_message="Insufficient funds"
        )
        
        assert result.success is False
        assert result.error_message == "Insufficient funds"


@pytest_asyncio.fixture
async def execution_engine():
    """Create ExecutionEngine with mocked dependencies."""
    # Mock REST client
    mock_rest = MagicMock()
    mock_rest.session = True  # Simulate connected
    mock_rest.connect = AsyncMock()
    mock_rest.close = AsyncMock()
    mock_rest.create_order = AsyncMock()
    mock_rest.cancel_order = AsyncMock()
    mock_rest.get_positions = AsyncMock(return_value=[])
    mock_rest.get_market = AsyncMock()
    
    # Create risk manager
    risk_manager = RiskManager(RiskConfig(max_position_size=50))
    
    engine = ExecutionEngine(mock_rest, risk_manager)
    
    yield engine
    
    await engine.stop()


class TestExecutionEngineInit:
    """Test ExecutionEngine initialization."""
    
    @pytest.mark.asyncio
    async def test_start_connects_client(self, execution_engine):
        """Test that start connects REST client."""
        await execution_engine.start()
        
        assert execution_engine._is_running is True
    
    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, execution_engine):
        """Test that stop cancels background tasks."""
        await execution_engine.start()
        await execution_engine.stop()
        
        assert execution_engine._is_running is False
    
    @pytest.mark.asyncio
    async def test_context_manager(self, execution_engine):
        """Test async context manager."""
        async with execution_engine:
            assert execution_engine._is_running
        
        assert not execution_engine._is_running


class TestSignalToOrderConversion:
    """Test signal to order parameter conversion."""
    
    @pytest.mark.asyncio
    async def test_long_signal_converts_to_yes_buy(self, execution_engine):
        """Test LONG signal converts to buy YES."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85,
            confidence=0.75
        )
        
        params = execution_engine._signal_to_order_params(signal)
        
        assert params['ticker'] == "KXBTC"
        assert params['side'] == "yes"
        assert params['price'] == 65  # 0.65 * 100 cents
        assert params['count'] > 0
    
    @pytest.mark.asyncio
    async def test_short_signal_converts_to_no_buy(self, execution_engine):
        """Test SHORT signal converts to buy NO."""
        signal = Signal(
            type=SignalType.SHORT,
            market_ticker="KXBTC",
            entry_price=0.65,
            stop_loss=0.75,
            take_profit=0.45,
            confidence=0.75
        )
        
        params = execution_engine._signal_to_order_params(signal)
        
        assert params['side'] == "no"
    
    @pytest.mark.asyncio
    async def test_exit_signal_raises_error(self, execution_engine):
        """Test EXIT signal raises error."""
        signal = Signal(
            type=SignalType.EXIT,
            market_ticker="KXBTC",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85
        )
        
        with pytest.raises(ValueError):
            execution_engine._signal_to_order_params(signal)


class TestExecuteSignal:
    """Test signal execution."""
    
    @pytest.fixture
    def long_signal(self):
        return Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85,
            confidence=0.75
        )
    
    @pytest.mark.asyncio
    async def test_execute_successful_order(self, execution_engine, long_signal):
        """Test successful order execution."""
        # Mock successful order creation
        mock_order = MagicMock()
        mock_order.order_id = "EXCHANGE_123"
        mock_order.status = OrderStatus.OPEN
        mock_order.filled_count = 0
        mock_order.remaining_count = 10
        
        execution_engine.rest_client.create_order.return_value = mock_order
        
        await execution_engine.start()
        result = await execution_engine.execute_signal(long_signal)
        
        assert result.success is True
        assert result.order_id is not None
        execution_engine.rest_client.create_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_immediate_fill(self, execution_engine, long_signal):
        """Test order that fills immediately."""
        mock_order = MagicMock()
        mock_order.order_id = "EXCHANGE_123"
        mock_order.status = OrderStatus.FILLED
        mock_order.filled_count = 10
        mock_order.remaining_count = 0
        
        execution_engine.rest_client.create_order.return_value = mock_order
        
        await execution_engine.start()
        result = await execution_engine.execute_signal(long_signal)
        
        assert result.success is True
        assert result.position is not None
    
    @pytest.mark.asyncio
    async def test_execute_failed_order(self, execution_engine, long_signal):
        """Test failed order execution."""
        execution_engine.rest_client.create_order.side_effect = Exception("API Error")
        
        await execution_engine.start()
        result = await execution_engine.execute_signal(long_signal)
        
        assert result.success is False
        assert "API Error" in result.error_message


class TestPositionTracking:
    """Test position tracking."""
    
    @pytest.fixture
    def execution_with_position(self, execution_engine):
        """Create engine with existing position."""
        from src.risk_manager import Position
        
        position = Position(
            market_ticker="KXBTC",
            side="yes",
            entry_price=0.60,
            count=10
        )
        execution_engine._positions["KXBTC"] = position
        return execution_engine
    
    def test_get_position_exists(self, execution_with_position):
        """Test getting existing position."""
        position = execution_with_position.get_position("KXBTC")
        
        assert position is not None
        assert position.market_ticker == "KXBTC"
        assert position.count == 10
    
    def test_get_position_not_found(self, execution_engine):
        """Test getting non-existent position."""
        position = execution_engine.get_position("UNKNOWN")
        
        assert position is None
    
    def test_get_all_positions(self, execution_with_position):
        """Test getting all positions."""
        positions = execution_with_position.get_all_positions()
        
        assert "KXBTC" in positions
        assert len(positions) == 1


class TestPnLCalculations:
    """Test P&L calculations."""
    
    @pytest.fixture
    def execution_with_long(self, execution_engine):
        """Create engine with long position."""
        from src.risk_manager import Position
        
        position = Position(
            market_ticker="KXBTC",
            side="yes",
            entry_price=0.60,
            count=10
        )
        execution_engine._positions["KXBTC"] = position
        return execution_engine
    
    @pytest.fixture
    def execution_with_short(self, execution_engine):
        """Create engine with short position."""
        from src.risk_manager import Position
        
        position = Position(
            market_ticker="KXBTC",
            side="no",
            entry_price=0.60,
            count=10
        )
        execution_engine._positions["KXBTC"] = position
        return execution_engine
    
    def test_unrealized_pnl_long_profit(self, execution_with_long):
        """Test unrealized P&L for profitable long."""
        # Entry 0.60, current 0.70, count 10
        # P&L = (0.70 - 0.60) * 10 = $1.00
        pnl = execution_with_long.get_unrealized_pnl("KXBTC", 0.70)
        
        assert pnl == 1.0
    
    def test_unrealized_pnl_long_loss(self, execution_with_long):
        """Test unrealized P&L for losing long."""
        # Entry 0.60, current 0.50, count 10
        # P&L = (0.50 - 0.60) * 10 = -$1.00
        pnl = execution_with_long.get_unrealized_pnl("KXBTC", 0.50)
        
        assert pnl == -1.0
    
    def test_unrealized_pnl_short_profit(self, execution_with_short):
        """Test unrealized P&L for profitable short."""
        # Entry 0.60, current 0.50, count 10
        # P&L = (0.60 - 0.50) * 10 = $1.00
        pnl = execution_with_short.get_unrealized_pnl("KXBTC", 0.50)
        
        assert pnl == 1.0
    
    def test_unrealized_pnl_no_position(self, execution_engine):
        """Test unrealized P&L with no position."""
        pnl = execution_engine.get_unrealized_pnl("KXBTC", 0.70)
        
        assert pnl == 0.0
    
    def test_total_unrealized_pnl(self, execution_engine):
        """Test total unrealized P&L."""
        from src.risk_manager import Position
        
        # Add multiple positions
        execution_engine._positions["BTC"] = Position(
            "BTC", "yes", 0.60, 10
        )
        execution_engine._positions["ETH"] = Position(
            "ETH", "no", 0.50, 10
        )
        
        prices = {"BTC": 0.70, "ETH": 0.40}
        
        # BTC: (0.70 - 0.60) * 10 = $1.00
        # ETH: (0.50 - 0.40) * 10 = $1.00
        # Total = $2.00
        total_pnl = execution_engine.get_total_unrealized_pnl(prices)
        
        assert total_pnl == 2.0


class TestCancelOrder:
    """Test order cancellation."""
    
    @pytest.mark.asyncio
    async def test_cancel_by_client_id(self, execution_engine):
        """Test cancelling by client order ID."""
        # Create tracked order
        order = TrackedOrder(
            client_order_id="CLIENT_123",
            exchange_order_id="EXCHANGE_123",
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10,
            state=OrderState.OPEN
        )
        execution_engine._orders["CLIENT_123"] = order
        execution_engine._exchange_order_map["EXCHANGE_123"] = "CLIENT_123"
        
        execution_engine.rest_client.cancel_order.return_value = None
        
        await execution_engine.start()
        result = await execution_engine.cancel_order("CLIENT_123")
        
        assert result is True
        assert order.state == OrderState.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_by_exchange_id(self, execution_engine):
        """Test cancelling by exchange order ID."""
        order = TrackedOrder(
            client_order_id="CLIENT_123",
            exchange_order_id="EXCHANGE_123",
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10
        )
        execution_engine._orders["CLIENT_123"] = order
        execution_engine._exchange_order_map["EXCHANGE_123"] = "CLIENT_123"
        
        execution_engine.rest_client.cancel_order.return_value = None
        
        await execution_engine.start()
        result = await execution_engine.cancel_order("EXCHANGE_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, execution_engine):
        """Test cancelling non-existent order."""
        await execution_engine.start()
        result = await execution_engine.cancel_order("UNKNOWN")
        
        assert result is False


class TestClosePosition:
    """Test position closing."""
    
    @pytest.mark.asyncio
    async def test_close_long_position(self, execution_engine):
        """Test closing long position."""
        from src.risk_manager import Position
        
        # Create position
        position = Position(
            market_ticker="KXBTC",
            side="yes",
            entry_price=0.60,
            count=10
        )
        execution_engine._positions["KXBTC"] = position
        
        # Mock order creation
        mock_order = MagicMock()
        mock_order.order_id = "CLOSE_123"
        execution_engine.rest_client.create_order.return_value = mock_order
        
        await execution_engine.start()
        result = await execution_engine.close_position("KXBTC", 0.70)
        
        assert result.success is True
        assert "KXBTC" not in execution_engine._positions
    
    @pytest.mark.asyncio
    async def test_close_position_not_found(self, execution_engine):
        """Test closing non-existent position."""
        await execution_engine.start()
        result = await execution_engine.close_position("UNKNOWN")
        
        assert result.success is False
        assert "No position found" in result.error_message


class TestOrderStatus:
    """Test order status retrieval."""
    
    def test_get_status_by_client_id(self, execution_engine):
        """Test getting status by client ID."""
        order = TrackedOrder(
            client_order_id="CLIENT_123",
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10
        )
        execution_engine._orders["CLIENT_123"] = order
        
        retrieved = execution_engine.get_order_status("CLIENT_123")
        
        assert retrieved == order
    
    def test_get_status_by_exchange_id(self, execution_engine):
        """Test getting status by exchange ID."""
        order = TrackedOrder(
            client_order_id="CLIENT_123",
            exchange_order_id="EXCHANGE_123",
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10
        )
        execution_engine._orders["CLIENT_123"] = order
        execution_engine._exchange_order_map["EXCHANGE_123"] = "CLIENT_123"
        
        retrieved = execution_engine.get_order_status("EXCHANGE_123")
        
        assert retrieved == order
    
    def test_get_status_not_found(self, execution_engine):
        """Test getting status for unknown order."""
        retrieved = execution_engine.get_order_status("UNKNOWN")
        
        assert retrieved is None


class TestOpenOrders:
    """Test open orders retrieval."""
    
    def test_get_open_orders(self, execution_engine):
        """Test getting open orders."""
        # Add multiple orders
        execution_engine._orders["OPEN_1"] = TrackedOrder(
            client_order_id="OPEN_1",
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10,
            state=OrderState.OPEN
        )
        execution_engine._orders["FILLED_1"] = TrackedOrder(
            client_order_id="FILLED_1",
            ticker="KETH",
            side="yes",
            price=0.60,
            count=10,
            state=OrderState.FILLED
        )
        execution_engine._orders["PARTIAL_1"] = TrackedOrder(
            client_order_id="PARTIAL_1",
            ticker="KSOL",
            side="yes",
            price=0.55,
            count=10,
            state=OrderState.PARTIAL_FILL
        )
        
        open_orders = execution_engine.get_open_orders()
        
        assert len(open_orders) == 2
        order_ids = {o.client_order_id for o in open_orders}
        assert "OPEN_1" in order_ids
        assert "PARTIAL_1" in order_ids


class TestStats:
    """Test statistics reporting."""
    
    def test_get_stats(self, execution_engine):
        """Test stats reporting."""
        stats = execution_engine.get_stats()
        
        assert 'is_running' in stats
        assert 'total_orders' in stats
        assert 'open_orders' in stats
        assert 'open_positions' in stats


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_insufficient_funds_error(self, execution_engine):
        """Test handling insufficient funds error."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85
        )
        
        execution_engine.rest_client.create_order.side_effect = Exception(
            "Insufficient funds"
        )
        
        await execution_engine.start()
        result = await execution_engine.execute_signal(signal)
        
        assert result.success is False
        assert "Insufficient funds" in result.error_message
    
    @pytest.mark.asyncio
    async def test_market_closed_error(self, execution_engine):
        """Test handling market closed error."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85
        )
        
        execution_engine.rest_client.create_order.side_effect = Exception(
            "Market is closed"
        )
        
        await execution_engine.start()
        result = await execution_engine.execute_signal(signal)
        
        assert result.success is False
        assert "closed" in result.error_message.lower()


class TestClientOrderIdGeneration:
    """Test client order ID generation."""
    
    def test_generates_unique_ids(self, execution_engine):
        """Test that generated IDs are unique."""
        ids = set()
        for _ in range(100):
            order_id = execution_engine._generate_client_order_id("KXBTC")
            ids.add(order_id)
        
        assert len(ids) == 100  # All unique
    
    def test_includes_ticker(self, execution_engine):
        """Test that ID includes ticker."""
        order_id = execution_engine._generate_client_order_id("KXBTC")
        
        assert "KXBTC" in order_id
    
    def test_includes_timestamp(self, execution_engine):
        """Test that ID includes timestamp."""
        order_id = execution_engine._generate_client_order_id("KXBTC")
        
        # Should have format: TICKER_YYYYMMDDhhmmss_uuid
        parts = order_id.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 14  # Timestamp length


class TestPartialFills:
    """Test partial fill handling."""
    
    @pytest.mark.asyncio
    async def test_partial_fill_tracked(self, execution_engine):
        """Test that partial fills are tracked."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC",
            entry_price=0.65,
            stop_loss=0.55,
            take_profit=0.85
        )
        
        # Mock partial fill
        mock_order = MagicMock()
        mock_order.order_id = "EXCHANGE_123"
        mock_order.status = OrderStatus.OPEN
        mock_order.filled_count = 5
        mock_order.remaining_count = 5
        
        execution_engine.rest_client.create_order.return_value = mock_order
        
        await execution_engine.start()
        result = await execution_engine.execute_signal(signal)
        
        assert result.success is True
        
        # Check tracked order
        tracked = execution_engine.get_order_status(result.order_id)
        assert tracked.filled_count == 5
        assert tracked.remaining_count == 5
        assert tracked.state == OrderState.PARTIAL_FILL


class TestPriceConversion:
    """Test price to cents conversion."""
    
    @pytest.mark.asyncio
    async def test_price_converted_to_cents(self, execution_engine):
        """Test that price is converted to cents for API."""
        signal = Signal(
            type=SignalType.LONG,
            market_ticker="KXBTC",
            entry_price=0.6575,  # Odd price
            stop_loss=0.55,
            take_profit=0.85
        )
        
        mock_order = MagicMock()
        mock_order.order_id = "EXCHANGE_123"
        mock_order.status = OrderStatus.OPEN
        mock_order.filled_count = 0
        mock_order.remaining_count = 10
        
        execution_engine.rest_client.create_order.return_value = mock_order
        
        await execution_engine.start()
        await execution_engine.execute_signal(signal)
        
        # Check that create_order was called with price in cents
        call_args = execution_engine.rest_client.create_order.call_args
        assert call_args.kwargs['price'] == 65  # 0.6575 * 100 = 65.75, truncated to 65
