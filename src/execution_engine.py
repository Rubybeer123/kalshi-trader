"""ExecutionEngine handles order lifecycle and position tracking."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

import structlog

from src.client import KalshiRestClient
from src.data.models import Order, OrderStatus, Side
from src.risk_manager import Position, RiskManager
from src.strategies.base import Signal, SignalType

logger = structlog.get_logger(__name__)


class OrderState(str, Enum):
    """Internal order state tracking."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class OrderResult:
    """Result of order execution attempt."""
    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    position: Optional[Position] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedOrder:
    """Internal order tracking with state."""
    client_order_id: str
    ticker: str
    side: str  # 'yes' or 'no'
    price: float
    count: int
    exchange_order_id: Optional[str] = None
    state: OrderState = OrderState.PENDING
    filled_count: int = 0
    remaining_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


class ExecutionEngine:
    """
    Handles order execution, lifecycle management, and position tracking.
    """
    
    def __init__(
        self,
        rest_client: KalshiRestClient,
        risk_manager: RiskManager
    ):
        """
        Initialize ExecutionEngine.
        
        Args:
            rest_client: Kalshi REST API client
            risk_manager: Risk manager for position sizing
        """
        self.rest_client = rest_client
        self.risk_manager = risk_manager
        
        # Order tracking
        self._orders: Dict[str, TrackedOrder] = {}  # client_order_id -> TrackedOrder
        self._exchange_order_map: Dict[str, str] = {}  # exchange_id -> client_id
        
        # Position tracking
        self._positions: Dict[str, Position] = {}
        
        # Running state
        self._is_running = False
        self._update_task: Optional[asyncio.Task] = None
        
        logger.info("ExecutionEngine initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start execution engine."""
        if self._is_running:
            return
        
        logger.info("Starting ExecutionEngine")
        
        # Ensure REST client is connected
        if self.rest_client.session is None:
            await self.rest_client.connect()
        
        self._is_running = True
        
        # Start position update loop
        self._update_task = asyncio.create_task(self._position_update_loop())
        
        logger.info("ExecutionEngine started")
    
    async def stop(self) -> None:
        """Stop execution engine."""
        if not self._is_running:
            return
        
        logger.info("Stopping ExecutionEngine")
        
        self._is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("ExecutionEngine stopped")
    
    def _generate_client_order_id(self, ticker: str) -> str:
        """
        Generate unique client order ID for idempotency.
        
        Format: {ticker}_{timestamp}_{uuid_short}
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"{ticker}_{timestamp}_{unique}"
    
    def _signal_to_order_params(self, signal: Signal) -> Dict[str, Any]:
        """
        Convert strategy signal to Kalshi order parameters.
        
        LONG signal -> Buy YES contracts (or sell NO for short)
        SHORT signal -> Buy NO contracts (or sell YES for short)
        
        Args:
            signal: Trading signal
            
        Returns:
            Dictionary with order parameters
        """
        ticker = signal.market_ticker
        
        # Determine side and action
        if signal.type == SignalType.LONG:
            # Long = buy YES contracts
            side = "yes"
            action = "buy"
        elif signal.type == SignalType.SHORT:
            # Short = buy NO contracts
            side = "no"
            action = "buy"
        else:  # EXIT
            # Exit logic handled separately
            raise ValueError("EXIT signals not supported in execute_signal")
        
        # Calculate position size using risk manager
        account_balance = 1000.0  # TODO: Get from actual balance
        risk_per_trade = self.risk_manager.config.max_risk_per_trade_pct
        stop_distance = signal.stop_distance
        
        count = self.risk_manager.calculate_position_size(
            account_balance,
            risk_per_trade,
            stop_distance
        )
        
        # Convert price to cents for Kalshi
        price_cents = int(signal.entry_price * 100)
        
        return {
            'ticker': ticker,
            'side': side,
            'price': price_cents,  # Kalshi expects cents
            'count': count,
        }
    
    async def execute_signal(self, signal: Signal) -> OrderResult:
        """
        Execute a trading signal by creating a Kalshi order.
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            OrderResult with execution status
        """
        ticker = signal.market_ticker
        
        logger.info(
            "Executing signal",
            ticker=ticker,
            type=signal.type.value,
            entry=signal.entry_price,
            stop=signal.stop_loss,
            target=signal.take_profit
        )
        
        try:
            # Generate client order ID
            client_order_id = self._generate_client_order_id(ticker)
            
            # Convert signal to order params
            order_params = self._signal_to_order_params(signal)
            
            # Create tracked order
            tracked = TrackedOrder(
                client_order_id=client_order_id,
                ticker=order_params['ticker'],
                side=order_params['side'],
                price=order_params['price'] / 100.0,  # Store as decimal
                count=order_params['count'],
                state=OrderState.PENDING
            )
            self._orders[client_order_id] = tracked
            
            # Submit order to exchange
            tracked.state = OrderState.SUBMITTED
            
            order = await self.rest_client.create_order(
                market_ticker=order_params['ticker'],
                side=order_params['side'],
                price=order_params['price'],
                count=order_params['count'],
                client_order_id=client_order_id
            )
            
            # Update tracking with exchange order ID
            tracked.exchange_order_id = order.order_id
            tracked.state = OrderState.OPEN
            tracked.remaining_count = order.count - order.filled_count
            self._exchange_order_map[order.order_id] = client_order_id
            
            logger.info(
                "Order submitted",
                client_id=client_order_id,
                exchange_id=order.order_id,
                ticker=ticker,
                side=order_params['side'],
                count=order_params['count'],
                price=order_params['price']
            )
            
            # Create position if filled
            if order.status == OrderStatus.FILLED or order.filled_count > 0:
                await self._handle_fill(tracked, order)
            
            return OrderResult(
                success=True,
                order_id=client_order_id,
                position=self._positions.get(ticker)
            )
            
        except Exception as e:
            logger.error("Order execution failed", ticker=ticker, error=str(e))
            
            if client_order_id in self._orders:
                self._orders[client_order_id].state = OrderState.ERROR
                self._orders[client_order_id].error_message = str(e)
            
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def _handle_fill(self, tracked: TrackedOrder, order: Order) -> None:
        """
        Handle order fill by creating/updating position.
        
        Args:
            tracked: Tracked order
            order: Order from exchange
        """
        ticker = tracked.ticker
        filled = order.filled_count
        
        if filled <= 0:
            return
        
        # Update tracked order
        tracked.filled_count = filled
        tracked.remaining_count = order.remaining_count
        
        if order.status == OrderStatus.FILLED:
            tracked.state = OrderState.FILLED
        elif order.filled_count > 0:
            tracked.state = OrderState.PARTIAL_FILL
        
        # Create or update position
        position = Position(
            market_ticker=ticker,
            side=tracked.side,
            entry_price=tracked.price,
            count=filled,
            stop_loss=None,  # Set from signal metadata
            take_profit=None,
            entry_time=datetime.now(timezone.utc)
        )
        
        self._positions[ticker] = position
        self.risk_manager.add_position(position)
        
        logger.info(
            "Position created",
            ticker=ticker,
            side=tracked.side,
            count=filled,
            entry=tracked.price
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Client order ID or exchange order ID
            
        Returns:
            True if cancellation successful
        """
        # Resolve order ID
        if order_id in self._orders:
            tracked = self._orders[order_id]
            exchange_id = tracked.exchange_order_id
        elif order_id in self._exchange_order_map:
            client_id = self._exchange_order_map[order_id]
            tracked = self._orders[client_id]
            exchange_id = order_id
        else:
            logger.warning("Order not found for cancellation", order_id=order_id)
            return False
        
        if not exchange_id:
            logger.warning("Order not yet submitted to exchange", order_id=order_id)
            return False
        
        try:
            await self.rest_client.cancel_order(exchange_id)
            tracked.state = OrderState.CANCELLED
            
            logger.info("Order cancelled", order_id=order_id)
            return True
            
        except Exception as e:
            logger.error("Order cancellation failed", order_id=order_id, error=str(e))
            return False
    
    async def update_positions(self) -> None:
        """
        Sync positions with exchange and update P&L.
        """
        try:
            # Get positions from exchange
            positions = await self.rest_client.get_positions()
            
            # Update local tracking
            for pos_data in positions:
                ticker = pos_data.market_ticker
                
                if ticker in self._positions:
                    # Update existing position
                    position = self._positions[ticker]
                    position.count = pos_data.count
                else:
                    # New position from elsewhere
                    position = Position(
                        market_ticker=ticker,
                        side=pos_data.side,
                        entry_price=pos_data.avg_entry_price or 0.0,
                        count=pos_data.count
                    )
                    self._positions[ticker] = position
            
            logger.debug("Positions updated", count=len(self._positions))
            
        except Exception as e:
            logger.error("Position update failed", error=str(e))
    
    async def _position_update_loop(self) -> None:
        """Background task to periodically update positions."""
        while self._is_running:
            try:
                await self.update_positions()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Position update loop error", error=str(e))
                await asyncio.sleep(5)
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """
        Get current position for a ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Position or None
        """
        return self._positions.get(ticker)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self._positions.copy()
    
    def get_unrealized_pnl(self, ticker: str, current_price: float) -> float:
        """
        Calculate unrealized P&L for a position.
        
        Args:
            ticker: Market ticker
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        position = self._positions.get(ticker)
        if not position:
            return 0.0
        
        if position.side == 'yes':
            pnl = (current_price - position.entry_price) * position.count
        else:  # 'no'
            pnl = (position.entry_price - current_price) * position.count
        
        return pnl
    
    def get_total_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """
        Calculate total unrealized P&L across all positions.
        
        Args:
            prices: Dictionary of ticker -> current price
            
        Returns:
            Total unrealized P&L
        """
        total = 0.0
        for ticker, position in self._positions.items():
            if ticker in prices:
                total += self.get_unrealized_pnl(ticker, prices[ticker])
        return total
    
    async def close_position(
        self,
        ticker: str,
        price: Optional[float] = None
    ) -> OrderResult:
        """
        Close an open position.
        
        Args:
            ticker: Market ticker
            price: Optional limit price (market order if None)
            
        Returns:
            OrderResult
        """
        position = self._positions.get(ticker)
        if not position:
            return OrderResult(
                success=False,
                error_message=f"No position found for {ticker}"
            )
        
        # Determine closing side (opposite of position)
        close_side = 'no' if position.side == 'yes' else 'yes'
        
        # Use current market price if not specified
        if price is None:
            try:
                market = await self.rest_client.get_market(ticker)
                if close_side == 'yes':
                    price = float(market.yes_ask) if market.yes_ask else 0.5
                else:
                    price = float(market.no_ask) if market.no_ask else 0.5
            except Exception as e:
                logger.error("Failed to get market price", ticker=ticker, error=str(e))
                return OrderResult(success=False, error_message=str(e))
        
        try:
            # Create closing order
            client_order_id = self._generate_client_order_id(f"{ticker}_close")
            
            price_cents = int(price * 100)
            
            order = await self.rest_client.create_order(
                market_ticker=ticker,
                side=close_side,
                price=price_cents,
                count=position.count
            )
            
            # Calculate realized P&L
            realized_pnl = self.risk_manager.close_position(ticker, price)
            
            # Remove from local tracking
            if ticker in self._positions:
                del self._positions[ticker]
            
            logger.info(
                "Position closed",
                ticker=ticker,
                side=close_side,
                count=position.count,
                realized_pnl=realized_pnl
            )
            
            return OrderResult(
                success=True,
                order_id=client_order_id,
                metadata={'realized_pnl': realized_pnl}
            )
            
        except Exception as e:
            logger.error("Position close failed", ticker=ticker, error=str(e))
            return OrderResult(success=False, error_message=str(e))
    
    def get_order_status(self, order_id: str) -> Optional[TrackedOrder]:
        """
        Get tracked order status.
        
        Args:
            order_id: Client or exchange order ID
            
        Returns:
            TrackedOrder or None
        """
        if order_id in self._orders:
            return self._orders[order_id]
        
        if order_id in self._exchange_order_map:
            client_id = self._exchange_order_map[order_id]
            return self._orders.get(client_id)
        
        return None
    
    def get_open_orders(self) -> List[TrackedOrder]:
        """Get all open orders."""
        return [
            order for order in self._orders.values()
            if order.state in [OrderState.OPEN, OrderState.PARTIAL_FILL]
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution engine statistics."""
        return {
            'is_running': self._is_running,
            'total_orders': len(self._orders),
            'open_orders': len(self.get_open_orders()),
            'open_positions': len(self._positions),
            'positions': list(self._positions.keys()),
        }
