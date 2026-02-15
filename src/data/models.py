"""Pydantic models for Kalshi API data."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MarketStatus(str, Enum):
    """Market status values."""
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"


class Side(str, Enum):
    """Order side."""
    YES = "yes"
    NO = "no"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Market(BaseModel):
    """Market data model."""
    model_config = ConfigDict(frozen=True)
    
    ticker: str
    title: str
    category: str
    status: MarketStatus
    expiration_date: Optional[datetime] = None
    yes_bid: Optional[Decimal] = None
    yes_ask: Optional[Decimal] = None
    no_bid: Optional[Decimal] = None
    no_ask: Optional[Decimal] = None
    volume: int = Field(default=0, ge=0)
    open_interest: int = Field(default=0, ge=0)
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price if both bid and ask exist."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate spread if both bid and ask exist."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None
    
    @field_validator("yes_bid", "yes_ask", "no_bid", "no_ask", mode="before")
    @classmethod
    def convert_price(cls, v: Any) -> Optional[Decimal]:
        """Convert price to Decimal."""
        if v is None:
            return None
        return Decimal(str(v))


class OrderBookLevel(BaseModel):
    """Single level in order book."""
    model_config = ConfigDict(frozen=True)
    
    price: Decimal
    count: int = Field(ge=0)
    
    @field_validator("price", mode="before")
    @classmethod
    def convert_price(cls, v: Any) -> Decimal:
        return Decimal(str(v))


class OrderBook(BaseModel):
    """Order book data model."""
    model_config = ConfigDict(frozen=True)
    
    ticker: str
    yes_bids: List[OrderBookLevel] = Field(default_factory=list)
    yes_asks: List[OrderBookLevel] = Field(default_factory=list)
    no_bids: List[OrderBookLevel] = Field(default_factory=list)
    no_asks: List[OrderBookLevel] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    
    @property
    def best_yes_bid(self) -> Optional[Decimal]:
        """Best YES bid price."""
        if self.yes_bids:
            return self.yes_bids[0].price
        return None
    
    @property
    def best_yes_ask(self) -> Optional[Decimal]:
        """Best YES ask price."""
        if self.yes_asks:
            return self.yes_asks[0].price
        return None
    
    @property
    def best_no_bid(self) -> Optional[Decimal]:
        """Best NO bid price."""
        if self.no_bids:
            return self.no_bids[0].price
        return None
    
    @property
    def best_no_ask(self) -> Optional[Decimal]:
        """Best NO ask price."""
        if self.no_asks:
            return self.no_asks[0].price
        return None


class Balance(BaseModel):
    """Account balance model."""
    model_config = ConfigDict(frozen=True)
    
    available_balance: Decimal
    total_balance: Decimal
    portfolio_value: Decimal
    
    @field_validator("available_balance", "total_balance", "portfolio_value", mode="before")
    @classmethod
    def convert_amount(cls, v: Any) -> Decimal:
        return Decimal(str(v))


class Position(BaseModel):
    """Position model."""
    model_config = ConfigDict(frozen=True)
    
    market_ticker: str
    side: Side
    count: int = Field(ge=0)
    avg_entry_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    
    @field_validator("avg_entry_price", "unrealized_pnl", mode="before")
    @classmethod
    def convert_optional_price(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return Decimal(str(v))


class Order(BaseModel):
    """Order model."""
    model_config = ConfigDict(frozen=True)
    
    order_id: str
    client_order_id: Optional[str] = None
    market_ticker: str
    side: Side
    price: Decimal
    count: int = Field(ge=0)
    status: OrderStatus
    created_at: Optional[datetime] = None
    filled_count: int = Field(default=0, ge=0)
    remaining_count: int = Field(default=0, ge=0)
    
    @field_validator("price", mode="before")
    @classmethod
    def convert_price(cls, v: Any) -> Decimal:
        return Decimal(str(v))
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED or self.remaining_count == 0
    
    @property
    def fill_ratio(self) -> float:
        """Calculate fill ratio."""
        if self.count == 0:
            return 0.0
        return self.filled_count / self.count


class CreateOrderRequest(BaseModel):
    """Request to create an order."""
    model_config = ConfigDict(frozen=True)
    
    market_ticker: str
    side: Side
    price: Decimal = Field(ge=0.01, le=0.99)
    count: int = Field(ge=1)
    client_order_id: Optional[str] = None
    
    @field_validator("price", mode="before")
    @classmethod
    def convert_price(cls, v: Any) -> Decimal:
        return Decimal(str(v))


class MarketsResponse(BaseModel):
    """Response from get_markets endpoint."""
    model_config = ConfigDict(frozen=True)
    
    markets: List[Market]
    cursor: Optional[str] = None
    has_more: bool = False


class KalshiAPIError(Exception):
    """Base exception for Kalshi API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body or {}


class RateLimitError(KalshiAPIError):
    """Raised when rate limit is exceeded (429)."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after


class AuthError(KalshiAPIError):
    """Raised for authentication errors (401/403)."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, **kwargs)


class MarketError(KalshiAPIError):
    """Raised for market-related errors (4xx)."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class ValidationError(KalshiAPIError):
    """Raised for request validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
