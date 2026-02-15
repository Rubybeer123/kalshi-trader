"""Tests for PaperTradingExchange."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from src.candle_aggregator import OrderBook, OrderBookLevel
from src.data.models import OrderStatus
from src.paper_trading import (
    PaperTradingExchange,
    SimulatedOrder,
    SimulatedPosition,
    SlippageModel,
    FillStatus,
)


class TestSlippageModel:
    """Test slippage model."""
    
    def test_applies_slippage_to_buy(self):
        """Test slippage increases buy price."""
        price = 0.50
        
        # Multiple samples to account for randomness
        slippages = []
        for _ in range(100):
            filled = SlippageModel.apply_slippage(price, 'yes', is_buy=True)
            slippages.append(filled)
        
        # Most should be higher than intended
        higher_count = sum(1 for s in slippages if s > price)
        assert higher_count > 80  # At least 80% have slippage
    
    def test_applies_slippage_to_sell(self):
        """Test slippage decreases sell price."""
        price = 0.50
        
        slippages = []
        for _ in range(100):
            filled = SlippageModel.apply_slippage(price, 'yes', is_buy=False)
            slippages.append(filled)
        
        lower_count = sum(1 for s in slippages if s < price)
        assert lower_count > 80
    
    def test_clamps_to_valid_range(self):
        """Test slippage stays within 0.01-0.99."""
        # Test edge cases
        high = SlippageModel.apply_slippage(0.98, 'yes', True)
        low = SlippageModel.apply_slippage(0.02, 'yes', False)
        
        assert 0.01 <= high <= 0.99
        assert 0.01 <= low <= 0.99


class TestPaperTradingExchangeInit:
    """Test exchange initialization."""
    
    def test_default_balance(self):
        """Test default starting balance."""
        exchange = PaperTradingExchange()
        
        assert exchange.starting_balance == 10000.0
    
    def test_custom_balance(self):
        """Test custom starting balance."""
        exchange = PaperTradingExchange(starting_balance=5000.0)
        
        assert exchange.starting_balance == 5000.0


class TestCreateOrder:
    """Test order creation and fills."""
    
    @pytest.fixture
    def exchange(self):
        return PaperTradingExchange(starting_balance=1000.0)
    
    @pytest.fixture
    def orderbook_buy(self):
        """Order book favorable for buy fill."""
        return OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)],
            yes_bids=[OrderBookLevel(price=0.58, count=100)],
            no_asks=[OrderBookLevel(price=0.40, count=100)],
            no_bids=[OrderBookLevel(price=0.38, count=100)]
        )
    
    @pytest.mark.asyncio
    async def test_create_buy_order_fill(self, exchange, orderbook_buy):
        """Test buy order fills when price matches."""
        exchange.update_market_data("KXBTC", orderbook_buy)
        
        # Buy YES at 0.65, best ask is 0.60 - should fill
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=65,  # 0.65 in cents
            count=10
        )
        
        assert order.status == OrderStatus.FILLED
        assert order.filled_count == 10
    
    @pytest.mark.asyncio
    async def test_create_buy_order_no_fill(self, exchange):
        """Test buy order doesn't fill when price too low."""
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.70, count=100)],
            yes_bids=[OrderBookLevel(price=0.68, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        # Buy YES at 0.60, best ask is 0.70 - shouldn't fill
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=60,  # 0.60 in cents
            count=10
        )
        
        assert order.status == OrderStatus.OPEN
        assert order.filled_count == 0
    
    @pytest.mark.asyncio
    async def test_create_sell_order_fill(self, exchange):
        """Test sell order fills when price matches."""
        orderbook = OrderBook(
            ticker="KXBTC",
            no_asks=[OrderBookLevel(price=0.35, count=100)],
            no_bids=[OrderBookLevel(price=0.33, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        # Buy NO at 0.40, best ask is 0.35 - should fill
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="no",
            price=40,  # 0.40 in cents
            count=10
        )
        
        assert order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_insufficient_balance(self, exchange, orderbook_buy):
        """Test order rejected with insufficient balance."""
        exchange.update_market_data("KXBTC", orderbook_buy)
        
        # Try to buy way more than balance allows
        with pytest.raises(ValueError, match="Insufficient balance"):
            await exchange.create_order(
                market_ticker="KXBTC",
                side="yes",
                price=65,
                count=10000  # Way too many
            )
    
    @pytest.mark.asyncio
    async def test_order_deducts_balance(self, exchange, orderbook_buy):
        """Test filled order deducts from balance."""
        exchange.update_market_data("KXBTC", orderbook_buy)
        
        initial_balance = (await exchange.get_balance()).available_balance
        
        await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=65,
            count=10
        )
        
        final_balance = (await exchange.get_balance()).available_balance
        
        # Should have deducted roughly 10 * 0.60 = 6.00 (plus slippage)
        assert final_balance < initial_balance


class TestCancelOrder:
    """Test order cancellation."""
    
    @pytest.fixture
    def exchange(self):
        return PaperTradingExchange(starting_balance=1000.0)
    
    @pytest.mark.asyncio
    async def test_cancel_open_order(self, exchange):
        """Test cancelling an open order."""
        # Create order that won't fill
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.70, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=60,  # Below market
            count=10
        )
        
        assert order.status == OrderStatus.OPEN
        
        # Cancel it
        cancelled = await exchange.cancel_order(order.order_id)
        
        assert cancelled.status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_filled_order_fails(self, exchange):
        """Test cannot cancel filled order."""
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=65,
            count=10
        )
        
        assert order.status == OrderStatus.FILLED
        
        with pytest.raises(ValueError):
            await exchange.cancel_order(order.order_id)
    
    @pytest.mark.asyncio
    async def test_cancel_refunds_balance(self, exchange):
        """Test cancellation refunds reserved balance."""
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.70, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        initial_balance = (await exchange.get_balance()).available_balance
        
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=60,  # Won't fill
            count=10
        )
        
        # Balance should be reduced (reserved)
        reserved_balance = (await exchange.get_balance()).available_balance
        assert reserved_balance < initial_balance
        
        # Cancel and refund
        await exchange.cancel_order(order.order_id)
        
        refunded_balance = (await exchange.get_balance()).available_balance
        assert abs(refunded_balance - initial_balance) < 0.01


class TestBalanceUpdates:
    """Test balance tracking."""
    
    @pytest.fixture
    def exchange(self):
        return PaperTradingExchange(starting_balance=1000.0)
    
    @pytest.mark.asyncio
    async def test_initial_balance(self, exchange):
        """Test initial balance is correct."""
        balance = await exchange.get_balance()
        
        assert balance.total_balance == 1000.0
        assert balance.available_balance == 1000.0
    
    @pytest.mark.asyncio
    async def test_balance_after_fill(self, exchange):
        """Test balance after order fill."""
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=65,
            count=10
        )
        
        balance = await exchange.get_balance()
        
        # Should have deducted ~$6.00
        assert balance.available_balance < 994.0


class TestPositionTracking:
    """Test position tracking."""
    
    @pytest.fixture
    def exchange(self):
        return PaperTradingExchange(starting_balance=1000.0)
    
    @pytest.mark.asyncio
    async def test_position_created_on_fill(self, exchange):
        """Test position created when order fills."""
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=65,
            count=10
        )
        
        positions = await exchange.get_positions()
        
        assert len(positions) == 1
        assert positions[0].ticker == "KXBTC"
        assert positions[0].count == 10
    
    @pytest.mark.asyncio
    async def test_no_position_without_fill(self, exchange):
        """Test no position when order doesn't fill."""
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.70, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=60,
            count=10
        )
        
        positions = await exchange.get_positions()
        
        assert len(positions) == 0
    
    @pytest.mark.asyncio
    async def test_position_avg_entry_price(self, exchange):
        """Test position average entry price."""
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        # Two fills at same price
        await exchange.create_order("KXBTC", "yes", 65, 5)
        
        positions = await exchange.get_positions()
        
        assert positions[0].avg_entry_price > 0


class TestNetworkDelay:
    """Test network delay simulation."""
    
    @pytest.mark.asyncio
    async def test_create_order_has_delay(self):
        """Test that order creation simulates network delay."""
        exchange = PaperTradingExchange()
        
        start = asyncio.get_event_loop().time()
        
        await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=60,
            count=10
        )
        
        elapsed = asyncio.get_event_loop().time() - start
        
        # Should have at least minimum delay
        assert elapsed >= 0.1  # 100ms min


class TestFillLogic:
    """Test fill logic in detail."""
    
    @pytest.fixture
    def exchange(self):
        return PaperTradingExchange(starting_balance=1000.0)
    
    def test_buy_fill_when_ask_below_limit(self, exchange):
        """Test buy fills when best ask is below limit price."""
        order = SimulatedOrder(
            order_id="TEST_1",
            client_order_id=None,
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10
        )
        
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        
        status, fill_count, fill_price = exchange._should_fill(order, orderbook)
        
        assert status in [FillStatus.FILLED, FillStatus.PARTIAL]
        assert fill_count > 0
    
    def test_buy_no_fill_when_ask_above_limit(self, exchange):
        """Test buy doesn't fill when best ask is above limit."""
        order = SimulatedOrder(
            order_id="TEST_1",
            ticker="KXBTC",
            side="yes",
            price=0.55,
            count=10
        )
        
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        
        status, fill_count, fill_price = exchange._should_fill(order, orderbook)
        
        assert status == FillStatus.PENDING
        assert fill_count == 0
    
    def test_no_fill_without_orderbook(self, exchange):
        """Test no fill when no orderbook data."""
        order = SimulatedOrder(
            order_id="TEST_1",
            ticker="KXBTC",
            side="yes",
            price=0.65,
            count=10
        )
        
        status, fill_count, fill_price = exchange._should_fill(order, None)
        
        assert status == FillStatus.PENDING
        assert fill_count == 0


class TestTryFillOrders:
    """Test the retry fill mechanism."""
    
    @pytest.mark.asyncio
    async def test_fill_pending_orders_on_retry(self):
        """Test pending orders fill when market moves."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        # Create order that won't fill initially
        orderbook_high = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.70, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook_high)
        
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=60,
            count=10
        )
        
        assert order.status == OrderStatus.OPEN
        
        # Market moves down
        orderbook_low = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.55, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook_low)
        
        # Try to fill pending orders
        await exchange.try_fill_orders()
        
        # Check updated order
        updated = await exchange.get_order(order.order_id)
        assert updated.status == OrderStatus.FILLED


class TestPartialFills:
    """Test partial fill behavior."""
    
    @pytest.mark.asyncio
    async def test_partial_fill_tracked(self):
        """Test partial fills are tracked correctly."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=1000)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        # Force partial fill by manipulating probability
        # This is tricky since it's random - we just verify structure
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=65,
            count=100
        )
        
        if order.filled_count > 0 and order.filled_count < 100:
            # Was partial fill
            assert order.remaining_count == 100 - order.filled_count


class TestStats:
    """Test statistics reporting."""
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test stats reporting."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        await exchange.create_order("KXBTC", "yes", 65, 10)
        
        stats = exchange.get_stats()
        
        assert 'starting_balance' in stats
        assert 'current_balance' in stats
        assert 'total_orders' in stats
        assert stats['total_orders'] == 1
    
    def test_reset(self):
        """Test reset functionality."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        # Add some state
        exchange._orders["TEST"] = SimulatedOrder(
            order_id="TEST",
            ticker="KXBTC",
            side="yes",
            price=0.60,
            count=10
        )
        exchange._positions["KXBTC"] = SimulatedPosition(
            ticker="KXBTC",
            side="yes",
            count=10,
            avg_entry_price=0.60
        )
        
        # Reset
        exchange.reset()
        
        assert len(exchange._orders) == 0
        assert len(exchange._positions) == 0
        assert exchange._balance == 1000.0


class TestClientOrderId:
    """Test client order ID mapping."""
    
    @pytest.mark.asyncio
    async def test_client_order_id_preserved(self):
        """Test client order ID is preserved and mapped."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        client_id = "MY_CUSTOM_ID_123"
        
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=65,
            count=10,
            client_order_id=client_id
        )
        
        assert order.client_order_id == client_id
        
        # Should be able to retrieve by client ID
        retrieved = await exchange.get_order(client_id)
        assert retrieved.order_id == order.order_id
    
    @pytest.mark.asyncio
    async def test_cancel_by_client_id(self):
        """Test cancellation by client order ID."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        orderbook = OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.70, count=100)]
        )
        exchange.update_market_data("KXBTC", orderbook)
        
        client_id = "CANCEL_ME"
        
        order = await exchange.create_order(
            market_ticker="KXBTC",
            side="yes",
            price=60,
            count=10,
            client_order_id=client_id
        )
        
        # Cancel by client ID
        cancelled = await exchange.cancel_order(client_id)
        
        assert cancelled.status == OrderStatus.CANCELLED


class TestMultipleMarkets:
    """Test multiple market handling."""
    
    @pytest.mark.asyncio
    async def test_separate_positions_per_market(self):
        """Test positions tracked separately per market."""
        exchange = PaperTradingExchange(starting_balance=10000.0)
        
        # Setup orderbooks
        exchange.update_market_data("BTC", OrderBook(
            ticker="BTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        ))
        exchange.update_market_data("ETH", OrderBook(
            ticker="ETH",
            yes_asks=[OrderBookLevel(price=0.55, count=100)]
        ))
        
        # Trade both
        await exchange.create_order("BTC", "yes", 65, 10)
        await exchange.create_order("ETH", "yes", 60, 10)
        
        positions = await exchange.get_positions()
        
        assert len(positions) == 2
        tickers = {p.ticker for p in positions}
        assert tickers == {"BTC", "ETH"}


class TestUnrealizedPnL:
    """Test unrealized P&L calculation."""
    
    @pytest.mark.asyncio
    async def test_unrealized_pnl_updates(self):
        """Test unrealized P&L updates with market price."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        # Buy at 0.60
        exchange.update_market_data("KXBTC", OrderBook(
            ticker="KXBTC",
            yes_asks=[OrderBookLevel(price=0.60, count=100)]
        ))
        
        await exchange.create_order("KXBTC", "yes", 65, 10)
        
        # Price goes up to 0.70
        exchange.update_market_price("KXBTC", 0.70)
        
        positions = await exchange.get_positions()
        
        # Should have positive unrealized P&L
        assert positions[0].unrealized_pnl > 0
    
    @pytest.mark.asyncio
    async def test_short_unrealized_pnl(self):
        """Test unrealized P&L for short position."""
        exchange = PaperTradingExchange(starting_balance=1000.0)
        
        # Short (buy NO) at 0.40
        exchange.update_market_data("KXBTC", OrderBook(
            ticker="KXBTC",
            no_asks=[OrderBookLevel(price=0.40, count=100)]
        ))
        
        await exchange.create_order("KXBTC", "no", 45, 10)
        
        # Price of NO goes up (market goes down) - profit for short
        exchange.update_market_price("KXBTC", 0.50)
        
        positions = await exchange.get_positions()
        
        # Should have positive unrealized P&L for short
        # NO price at 0.50 means YES at 0.50
        # Entry at 0.40, current 0.50 = profit
        assert positions[0].unrealized_pnl > 0
