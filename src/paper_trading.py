"""Paper trading exchange that simulates Kalshi behavior for testing."""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

from src.data.models import Market, Order, OrderStatus, OrderBook, OrderBookLevel

logger = structlog.get_logger(__name__)


class FillStatus(str, Enum):
    """Fill status for paper trades."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class SimulatedOrder:
    """Internal representation of a simulated order."""
    order_id: str
    client_order_id: Optional[str]
    ticker: str
    side: str  # 'yes' or 'no'
    price: float
    count: int
    filled_count: int = 0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fill_price: Optional[float] = None


@dataclass 
class SimulatedPosition:
    """Simulated position tracking."""
    ticker: str
    side: str
    count: int
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class SimulatedBalance:
    """Simulated account balance."""
    total_balance: float
    available_balance: float
    portfolio_value: float


class SlippageModel:
    """Models price slippage for realistic fills."""
    
    # Slippage in cents (0.01 = 1 cent)
    NORMAL_SLIPPAGE_RANGE = (1, 3)  # 1-3 ticks
    STRESSED_SLIPPAGE_RANGE = (5, 10)  # 5-10 ticks
    STRESS_PROBABILITY = 0.15  # 15% chance of stressed conditions
    
    @classmethod
    def apply_slippage(
        cls,
        intended_price: float,
        side: str,
        is_buy: bool
    ) -> float:
        """
        Apply slippage to intended price.
        
        Args:
            intended_price: Target price
            side: 'yes' or 'no'
            is_buy: True if buying, False if selling
            
        Returns:
            Price with slippage applied
        """
        # Determine slippage amount
        if random.random() < cls.STRESS_PROBABILITY:
            slippage_ticks = random.randint(*cls.STRESSED_SLIPPAGE_RANGE)
        else:
            slippage_ticks = random.randint(*cls.NORMAL_SLIPPAGE_RANGE)
        
        slippage = slippage_ticks * 0.01  # Convert ticks to price
        
        # Apply slippage against the trader
        if is_buy:
            # Buy: worse price is higher
            filled_price = intended_price + slippage
        else:
            # Sell: worse price is lower
            filled_price = intended_price - slippage
        
        # Clamp to valid range
        filled_price = max(0.01, min(0.99, filled_price))
        
        return round(filled_price, 2)


class PaperTradingExchange:
    """
    Simulates Kalshi exchange behavior for paper trading.
    
    Implements same interface as KalshiRestClient for drop-in replacement.
    """
    
    # Simulation parameters
    NETWORK_DELAY_MIN = 0.1  # 100ms
    NETWORK_DELAY_MAX = 0.5  # 500ms
    PARTIAL_FILL_PROBABILITY = 0.05  # 5% chance
    PARTIAL_FILL_MIN_RATIO = 0.3  # At least 30% filled
    
    def __init__(self, starting_balance: float = 10000.0):
        """
        Initialize paper trading exchange.
        
        Args:
            starting_balance: Starting account balance (default $10,000)
        """
        self.starting_balance = starting_balance
        self._balance = starting_balance
        self._available_balance = starting_balance
        
        # Order tracking
        self._orders: Dict[str, SimulatedOrder] = {}  # order_id -> order
        self._client_order_map: Dict[str, str] = {}  # client_id -> order_id
        
        # Position tracking
        self._positions: Dict[str, SimulatedPosition] = {}
        
        # Market data for fill simulation
        self._orderbooks: Dict[str, OrderBook] = {}
        self._market_prices: Dict[str, float] = {}
        
        # Stats
        self._order_counter = 0
        self._total_filled_volume = 0
        
        logger.info(
            "PaperTradingExchange initialized",
            starting_balance=starting_balance
        )
    
    async def _simulate_network_delay(self) -> None:
        """Simulate network latency."""
        delay = random.uniform(
            self.NETWORK_DELAY_MIN,
            self.NETWORK_DELAY_MAX
        )
        await asyncio.sleep(delay)
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"PAPER_{self._order_counter}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    
    def update_market_data(self, ticker: str, orderbook: OrderBook) -> None:
        """
        Update market data for fill simulation.
        
        Args:
            ticker: Market ticker
            orderbook: Current order book
        """
        self._orderbooks[ticker] = orderbook
        
        # Update market price (midpoint)
        if orderbook.best_yes_bid and orderbook.best_yes_ask:
            mid = (orderbook.best_yes_bid + orderbook.best_yes_ask) / 2
            self._market_prices[ticker] = mid
    
    def update_market_price(self, ticker: str, price: float) -> None:
        """Update just the market price."""
        self._market_prices[ticker] = price
    
    def _should_fill(
        self,
        order: SimulatedOrder,
        orderbook: Optional[OrderBook]
    ) -> tuple[FillStatus, int, float]:
        """
        Determine if order should fill against market data.
        
        Returns:
            Tuple of (fill_status, fill_count, fill_price)
        """
        if not orderbook:
            # No market data, can't fill
            return FillStatus.PENDING, 0, 0.0
        
        ticker = order.ticker
        side = order.side
        price = order.price
        count = order.count - order.filled_count
        
        # Determine fill conditions based on side and action
        if side == 'yes':
            # Buying YES = paying YES price
            best_ask = orderbook.best_yes_ask
            if best_ask and best_ask <= price:
                # Fill condition met
                fill_price = SlippageModel.apply_slippage(
                    min(price, best_ask), side, is_buy=True
                )
                
                # Determine fill amount
                if random.random() < self.PARTIAL_FILL_PROBABILITY:
                    # Partial fill
                    fill_ratio = random.uniform(
                        self.PARTIAL_FILL_MIN_RATIO, 1.0
                    )
                    fill_count = int(count * fill_ratio)
                    return FillStatus.PARTIAL, fill_count, fill_price
                else:
                    # Full fill
                    return FillStatus.FILLED, count, fill_price
        
        elif side == 'no':
            # Buying NO = paying NO price
            best_ask = orderbook.best_no_ask
            if best_ask and best_ask <= price:
                fill_price = SlippageModel.apply_slippage(
                    min(price, best_ask), side, is_buy=True
                )
                
                if random.random() < self.PARTIAL_FILL_PROBABILITY:
                    fill_ratio = random.uniform(
                        self.PARTIAL_FILL_MIN_RATIO, 1.0
                    )
                    fill_count = int(count * fill_ratio)
                    return FillStatus.PARTIAL, fill_count, fill_price
                else:
                    return FillStatus.FILLED, count, fill_price
        
        # No fill
        return FillStatus.PENDING, 0, 0.0
    
    def _update_balance_and_position(
        self,
        order: SimulatedOrder,
        fill_count: int,
        fill_price: float
    ) -> None:
        """Update balance and position after fill."""
        # Calculate cost
        cost = fill_count * fill_price
        
        # Update balance
        self._available_balance -= cost
        
        # Update position
        ticker = order.ticker
        if ticker in self._positions:
            # Update existing position
            position = self._positions[ticker]
            total_cost = (position.avg_entry_price * position.count) + cost
            position.count += fill_count
            position.avg_entry_price = total_cost / position.count
        else:
            # Create new position
            self._positions[ticker] = SimulatedPosition(
                ticker=ticker,
                side=order.side,
                count=fill_count,
                avg_entry_price=fill_price
            )
        
        self._total_filled_volume += fill_count
    
    async def create_order(
        self,
        market_ticker: str,
        side: str,
        price: int,  # In cents
        count: int,
        client_order_id: Optional[str] = None
    ) -> Order:
        """
        Create a simulated order.
        
        Args:
            market_ticker: Market ticker
            side: 'yes' or 'no'
            price: Price in cents
            count: Number of contracts
            client_order_id: Optional client order ID
            
        Returns:
            Simulated Order object
        """
        await self._simulate_network_delay()
        
        order_id = self._generate_order_id()
        price_decimal = price / 100.0
        
        # Check if sufficient balance
        estimated_cost = count * price_decimal
        if estimated_cost > self._available_balance:
            logger.warning(
                "Order rejected: insufficient balance",
                order_id=order_id,
                required=estimated_cost,
                available=self._available_balance
            )
            raise ValueError(f"Insufficient balance: {self._available_balance}")
        
        # Create simulated order
        simulated = SimulatedOrder(
            order_id=order_id,
            client_order_id=client_order_id,
            ticker=market_ticker,
            side=side,
            price=price_decimal,
            count=count
        )
        
        self._orders[order_id] = simulated
        if client_order_id:
            self._client_order_map[client_order_id] = order_id
        
        # Try to fill immediately
        orderbook = self._orderbooks.get(market_ticker)
        fill_status, fill_count, fill_price = self._should_fill(
            simulated, orderbook
        )
        
        if fill_status == FillStatus.FILLED:
            simulated.filled_count = fill_count
            simulated.fill_price = fill_price
            simulated.status = OrderStatus.FILLED
            self._update_balance_and_position(simulated, fill_count, fill_price)
            
            logger.info(
                "Order filled",
                order_id=order_id,
                ticker=market_ticker,
                side=side,
                count=fill_count,
                price=fill_price
            )
        
        elif fill_status == FillStatus.PARTIAL:
            simulated.filled_count = fill_count
            simulated.fill_price = fill_price
            simulated.status = OrderStatus.OPEN
            self._update_balance_and_position(simulated, fill_count, fill_price)
            
            logger.info(
                "Order partial fill",
                order_id=order_id,
                ticker=market_ticker,
                filled=fill_count,
                remaining=count - fill_count,
                price=fill_price
            )
        
        else:
            simulated.status = OrderStatus.OPEN
            logger.info(
                "Order open",
                order_id=order_id,
                ticker=market_ticker,
                side=side,
                price=price_decimal,
                count=count
            )
        
        # Return Order object matching Kalshi format
        return Order(
            order_id=order_id,
            client_order_id=client_order_id,
            market_ticker=market_ticker,
            side=side,
            price=Decimal(str(price_decimal)),
            count=count,
            status=simulated.status,
            filled_count=simulated.filled_count,
            remaining_count=count - simulated.filled_count,
            created_at=simulated.created_at
        )
    
    async def cancel_order(self, order_id: str) -> Order:
        """
        Cancel a simulated order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancelled Order
        """
        await self._simulate_network_delay()
        
        # Resolve order ID
        if order_id in self._client_order_map:
            order_id = self._client_order_map[order_id]
        
        if order_id not in self._orders:
            raise ValueError(f"Order not found: {order_id}")
        
        simulated = self._orders[order_id]
        
        if simulated.status == OrderStatus.FILLED:
            raise ValueError(f"Cannot fill filled order: {order_id}")
        
        # Refund reserved balance for unfilled portion
        unfilled = simulated.count - simulated.filled_count
        refund = unfilled * simulated.price
        self._available_balance += refund
        
        simulated.status = OrderStatus.CANCELLED
        
        logger.info("Order cancelled", order_id=order_id)
        
        return Order(
            order_id=order_id,
            client_order_id=simulated.client_order_id,
            market_ticker=simulated.ticker,
            side=simulated.side,
            price=Decimal(str(simulated.price)),
            count=simulated.count,
            status=OrderStatus.CANCELLED,
            filled_count=simulated.filled_count,
            remaining_count=unfilled
        )
    
    async def get_order(self, order_id: str) -> Order:
        """Get order status."""
        await self._simulate_network_delay()
        
        if order_id in self._client_order_map:
            order_id = self._client_order_map[order_id]
        
        if order_id not in self._orders:
            raise ValueError(f"Order not found: {order_id}")
        
        simulated = self._orders[order_id]
        
        return Order(
            order_id=order_id,
            client_order_id=simulated.client_order_id,
            market_ticker=simulated.ticker,
            side=simulated.side,
            price=Decimal(str(simulated.price)),
            count=simulated.count,
            status=simulated.status,
            filled_count=simulated.filled_count,
            remaining_count=simulated.count - simulated.filled_count
        )
    
    async def get_balance(self) -> SimulatedBalance:
        """Get simulated account balance."""
        await self._simulate_network_delay()
        
        # Calculate portfolio value
        portfolio_value = sum(
            pos.count * self._market_prices.get(pos.ticker, pos.avg_entry_price)
            for pos in self._positions.values()
        )
        
        return SimulatedBalance(
            total_balance=self._balance,
            available_balance=self._available_balance,
            portfolio_value=portfolio_value
        )
    
    async def get_positions(self) -> List[SimulatedPosition]:
        """Get simulated positions."""
        await self._simulate_network_delay()
        
        # Update unrealized P&L for each position
        for pos in self._positions.values():
            current_price = self._market_prices.get(pos.ticker)
            if current_price:
                if pos.side == 'yes':
                    pos.unrealized_pnl = (current_price - pos.avg_entry_price) * pos.count
                else:  # 'no'
                    pos.unrealized_pnl = (pos.avg_entry_price - current_price) * pos.count
        
        return list(self._positions.values())
    
    async def get_market(self, ticker: str) -> Optional[Market]:
        """Get simulated market data."""
        await self._simulate_network_delay()
        
        orderbook = self._orderbooks.get(ticker)
        if not orderbook:
            return None
        
        return Market(
            ticker=ticker,
            title=f"Simulated {ticker}",
            category="crypto",
            status="open",
            yes_bid=Decimal(str(orderbook.best_yes_bid)) if orderbook.best_yes_bid else None,
            yes_ask=Decimal(str(orderbook.best_yes_ask)) if orderbook.best_yes_ask else None,
            no_bid=Decimal(str(orderbook.best_no_bid)) if orderbook.best_no_bid else None,
            no_ask=Decimal(str(orderbook.best_no_ask)) if orderbook.best_no_ask else None,
        )
    
    async def get_orderbook(self, ticker: str) -> Optional[OrderBook]:
        """Get order book."""
        await self._simulate_network_delay()
        return self._orderbooks.get(ticker)
    
    async def try_fill_orders(self) -> None:
        """
        Attempt to fill open orders against current market data.
        
        Call this periodically to simulate market movement fills.
        """
        for order in list(self._orders.values()):
            if order.status != OrderStatus.OPEN:
                continue
            
            orderbook = self._orderbooks.get(order.ticker)
            if not orderbook:
                continue
            
            fill_status, fill_count, fill_price = self._should_fill(
                order, orderbook
            )
            
            if fill_status in [FillStatus.FILLED, FillStatus.PARTIAL]:
                old_filled = order.filled_count
                order.filled_count += fill_count
                order.fill_price = fill_price
                
                if order.filled_count >= order.count:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.OPEN
                
                # Update for new fill amount only
                new_fill = order.filled_count - old_filled
                self._update_balance_and_position(order, new_fill, fill_price)
                
                logger.info(
                    "Order filled on retry",
                    order_id=order.order_id,
                    filled=order.filled_count,
                    total=order.count
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get paper trading statistics."""
        return {
            'starting_balance': self.starting_balance,
            'current_balance': self._balance,
            'available_balance': self._available_balance,
            'total_orders': len(self._orders),
            'open_positions': len(self._positions),
            'total_filled_volume': self._total_filled_volume,
            'positions': [
                {
                    'ticker': p.ticker,
                    'side': p.side,
                    'count': p.count,
                    'avg_entry': p.avg_entry_price,
                    'unrealized_pnl': p.unrealized_pnl
                }
                for p in self._positions.values()
            ]
        }
    
    def reset(self) -> None:
        """Reset paper trading state."""
        self._balance = self.starting_balance
        self._available_balance = self.starting_balance
        self._orders.clear()
        self._client_order_map.clear()
        self._positions.clear()
        self._order_counter = 0
        self._total_filled_volume = 0
        
        logger.info("PaperTradingExchange reset")
