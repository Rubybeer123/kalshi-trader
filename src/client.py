"""Kalshi REST API client with rate limiting and retry logic."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import structlog
from aiohttp import ClientError, ClientResponseError, ClientSession

from src.auth import KalshiAuthenticator
from src.config import Config
from src.data.models import (
    Balance,
    CreateOrderRequest,
    KalshiAPIError,
    Market,
    MarketError,
    MarketsResponse,
    Order,
    OrderBook,
    Position,
    RateLimitError,
    AuthError,
    ValidationError,
)

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound=Any)


class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, read_rate: float = 20.0, write_rate: float = 10.0):
        """
        Initialize rate limiter.
        
        Args:
            read_rate: Maximum reads per second (default: 20 for basic tier)
            write_rate: Maximum writes per second (default: 10 for basic tier)
        """
        self.read_rate = read_rate
        self.write_rate = write_rate
        
        # Token buckets
        self.read_tokens = read_rate
        self.write_tokens = write_rate
        self.last_read_update = time.monotonic()
        self.last_write_update = time.monotonic()
        
        # Locks for thread safety
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
    
    async def acquire_read(self) -> None:
        """Acquire permission for a read request."""
        async with self._read_lock:
            now = time.monotonic()
            elapsed = now - self.last_read_update
            
            # Add tokens based on elapsed time
            self.read_tokens = min(
                self.read_rate,
                self.read_tokens + elapsed * self.read_rate
            )
            self.last_read_update = now
            
            if self.read_tokens < 1:
                # Need to wait
                wait_time = (1 - self.read_tokens) / self.read_rate
                logger.debug("Rate limit: waiting for read token", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
                self.read_tokens = 0
            else:
                self.read_tokens -= 1
    
    async def acquire_write(self) -> None:
        """Acquire permission for a write request."""
        async with self._write_lock:
            now = time.monotonic()
            elapsed = now - self.last_write_update
            
            # Add tokens based on elapsed time
            self.write_tokens = min(
                self.write_rate,
                self.write_tokens + elapsed * self.write_rate
            )
            self.last_write_update = now
            
            if self.write_tokens < 1:
                # Need to wait
                wait_time = (1 - self.write_tokens) / self.write_rate
                logger.debug("Rate limit: waiting for write token", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
                self.write_tokens = 0
            else:
                self.write_tokens -= 1


class KalshiRestClient:
    """Async REST client for Kalshi API."""
    
    # Rate limits for basic tier
    READ_RATE = 20.0  # reads per second
    WRITE_RATE = 10.0  # writes per second
    
    # Retry configuration
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    MAX_DELAY = 30.0  # seconds
    
    # Token refresh
    TOKEN_REFRESH_BUFFER = 300  # Refresh 5 minutes before expiry (30 min total)
    
    def __init__(self, config: Config, session: Optional[ClientSession] = None):
        """
        Initialize REST client.
        
        Args:
            config: Application configuration
            session: Optional aiohttp session (for testing)
        """
        self.config = config
        self.authenticator = KalshiAuthenticator(
            config.kalshi_api_key_id,
            config.kalshi_private_key_path
        )
        self.base_url = config.api_base_url
        self.session = session
        self._owned_session = session is None
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            read_rate=self.READ_RATE,
            write_rate=self.WRITE_RATE
        )
        
        # Token management
        self._session_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
        self._token_lock = asyncio.Lock()
        
        # Request tracking
        self._request_count = 0
        self._last_request_time = 0.0
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Initialize HTTP session."""
        if self.session is None:
            self.session = ClientSession(
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )
            self._owned_session = True
            logger.info("Initialized HTTP session", base_url=self.base_url)
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._owned_session and self.session:
            await self.session.close()
            self.session = None
            logger.info("Closed HTTP session")
    
    async def _get_session_token(self) -> str:
        """
        Get or refresh session token.
        
        Kalshi session tokens expire after 30 minutes.
        We refresh 5 minutes before expiry.
        """
        async with self._token_lock:
            now = time.monotonic()
            
            # Check if token is still valid
            if (
                self._session_token is not None
                and self._token_expiry is not None
                and now < (self._token_expiry - self.TOKEN_REFRESH_BUFFER)
            ):
                return self._session_token
            
            # Need to get new token
            logger.info("Refreshing session token")
            
            try:
                # Make login request
                path = "/trade-api/v2/log_in"
                auth_headers = self.authenticator.generate_headers("POST", path)
                
                async with self.session.post(
                    f"{self.base_url}{path}",
                    headers=auth_headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._session_token = data.get("token")
                        # Token expires in 30 minutes (1800 seconds)
                        self._token_expiry = now + 1800
                        logger.info(
                            "Session token refreshed",
                            expires_in=1800,
                            expires_at=self._token_expiry
                        )
                        return self._session_token
                    else:
                        error_text = await response.text()
                        raise AuthError(
                            f"Failed to get session token: {error_text}",
                            status_code=response.status
                        )
                        
            except ClientError as e:
                raise AuthError(f"Network error getting session token: {e}")
    
    def _is_write_method(self, method: str) -> bool:
        """Check if HTTP method is a write operation."""
        return method.upper() in ("POST", "PUT", "DELETE", "PATCH")
    
    async def _make_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        response_model: Optional[Type[T]] = None,
    ) -> T:
        """
        Make an API request with rate limiting and retry logic.
        
        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            json_data: JSON request body
            response_model: Pydantic model to validate response
            
        Returns:
            Parsed response data
            
        Raises:
            KalshiAPIError: On API errors
        """
        # Ensure session exists
        if self.session is None:
            await self.connect()
        
        # Apply rate limiting
        if self._is_write_method(method):
            await self.rate_limiter.acquire_write()
        else:
            await self.rate_limiter.acquire_read()
        
        # Build full URL
        url = f"{self.base_url}{path}"
        
        # Get session token for authenticated requests
        # (Login is the only unauthenticated endpoint we use)
        if path != "/trade-api/v2/log_in":
            token = await self._get_session_token()
            headers = {"Authorization": f"Bearer {token}"}
        else:
            headers = {}
        
        # Track request
        self._request_count += 1
        request_id = self._request_count
        start_time = time.monotonic()
        
        log = logger.bind(
            request_id=request_id,
            method=method,
            path=path,
        )
        
        # Retry loop with exponential backoff
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                log.debug(
                    "Making API request",
                    attempt=attempt + 1,
                    max_retries=self.MAX_RETRIES,
                )
                
                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers,
                ) as response:
                    
                    duration_ms = (time.monotonic() - start_time) * 1000
                    
                    # Handle errors
                    if response.status >= 400:
                        error_body = await response.json() if response.content_type == "application/json" else None
                        error_text = await response.text() if error_body is None else str(error_body)
                        
                        log.warning(
                            "API request failed",
                            status=response.status,
                            duration_ms=duration_ms,
                            error=error_text,
                        )
                        
                        # Map status codes to exceptions
                        if response.status == 429:
                            retry_after = int(response.headers.get("Retry-After", 1))
                            raise RateLimitError(
                                "Rate limit exceeded",
                                retry_after=retry_after,
                                status_code=429,
                                response_body=error_body,
                            )
                        elif response.status in (401, 403):
                            # Clear token on auth error - will refresh on retry
                            self._session_token = None
                            raise AuthError(
                                f"Authentication error: {error_text}",
                                status_code=response.status,
                                response_body=error_body,
                            )
                        elif response.status == 400:
                            raise ValidationError(
                                f"Validation error: {error_text}",
                                status_code=400,
                                response_body=error_body,
                            )
                        else:
                            raise MarketError(
                                f"API error: {error_text}",
                                status_code=response.status,
                                response_body=error_body,
                            )
                    
                    # Parse success response
                    if response.status == 204:  # No content
                        return None
                    
                    data = await response.json()
                    
                    log.info(
                        "API request successful",
                        status=response.status,
                        duration_ms=duration_ms,
                    )
                    
                    # Validate with Pydantic model if provided
                    if response_model:
                        return response_model.model_validate(data)
                    
                    return data
                    
            except (RateLimitError, ClientError) as e:
                last_exception = e
                
                # Don't retry on final attempt
                if attempt >= self.MAX_RETRIES:
                    break
                
                # Calculate delay with exponential backoff
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                else:
                    delay = min(
                        self.BASE_DELAY * (2 ** attempt),
                        self.MAX_DELAY
                    )
                
                log.warning(
                    "Request failed, will retry",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )
                
                await asyncio.sleep(delay)
                
                # Refresh token on auth errors
                if isinstance(e, AuthError):
                    self._session_token = None
        
        # All retries exhausted
        log.error(
            "Request failed after all retries",
            attempts=self.MAX_RETRIES + 1,
            error=str(last_exception),
        )
        
        if isinstance(last_exception, KalshiAPIError):
            raise last_exception
        elif isinstance(last_exception, ClientError):
            raise KalshiAPIError(f"Network error: {last_exception}")
        else:
            raise KalshiAPIError(f"Request failed: {last_exception}")
    
    # === Public API Methods ===
    
    async def get_markets(
        self,
        status: str = "open",
        event_type: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> MarketsResponse:
        """
        Get list of markets.
        
        Args:
            status: Market status filter (open, closed, settled)
            event_type: Filter by event type
            cursor: Pagination cursor
            limit: Max results per page
            
        Returns:
            MarketsResponse with list of markets
        """
        params = {
            "status": status,
            "limit": limit,
        }
        if event_type:
            params["event_type"] = event_type
        if cursor:
            params["cursor"] = cursor
        
        result = await self._make_request(
            method="GET",
            path="/trade-api/v2/markets",
            params=params,
        )
        
        # Wrap raw response in MarketsResponse
        markets = [Market.model_validate(m) for m in result.get("markets", [])]
        return MarketsResponse(
            markets=markets,
            cursor=result.get("cursor"),
            has_more=result.get("cursor") is not None,
        )
    
    async def get_market(self, ticker: str) -> Market:
        """
        Get details for a specific market.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            Market details
        """
        result = await self._make_request(
            method="GET",
            path=f"/trade-api/v2/markets/{ticker}",
        )
        
        return Market.model_validate(result.get("market", result))
    
    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """
        Get order book for a market.
        
        Args:
            ticker: Market ticker symbol
            depth: Number of levels to return
            
        Returns:
            OrderBook with bids and asks
        """
        result = await self._make_request(
            method="GET",
            path=f"/trade-api/v2/markets/{ticker}/orderbook",
            params={"depth": depth},
        )
        
        orderbook_data = result.get("orderbook", result)
        
        return OrderBook(
            ticker=ticker,
            yes_bids=[
                OrderBookLevel(price=b["price"], count=b["count"])
                for b in orderbook_data.get("yes_bids", [])
            ],
            yes_asks=[
                OrderBookLevel(price=a["price"], count=a["count"])
                for a in orderbook_data.get("yes_asks", [])
            ],
            no_bids=[
                OrderBookLevel(price=b["price"], count=b["count"])
                for b in orderbook_data.get("no_bids", [])
            ],
            no_asks=[
                OrderBookLevel(price=a["price"], count=a["count"])
                for a in orderbook_data.get("no_asks", [])
            ],
        )
    
    async def get_balance(self) -> Balance:
        """
        Get account balance.
        
        Returns:
            Account balance information
        """
        result = await self._make_request(
            method="GET",
            path="/trade-api/v2/portfolio/balance",
        )
        
        return Balance.model_validate(result)
    
    async def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of open positions
        """
        result = await self._make_request(
            method="GET",
            path="/trade-api/v2/portfolio/positions",
        )
        
        positions_data = result.get("positions", [])
        return [Position.model_validate(p) for p in positions_data]
    
    async def create_order(
        self,
        market_ticker: str,
        side: str,
        price: float,
        count: int,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Create a new order.
        
        Args:
            market_ticker: Market to trade
            side: "yes" or "no"
            price: Limit price (0.01 - 0.99)
            count: Number of contracts
            client_order_id: Optional client order ID
            
        Returns:
            Created order
        """
        # Validate request
        request = CreateOrderRequest(
            market_ticker=market_ticker,
            side=side,
            price=price,
            count=count,
            client_order_id=client_order_id,
        )
        
        json_data = {
            "ticker": request.market_ticker,
            "side": request.side,
            "price": float(request.price),
            "count": request.count,
        }
        
        if request.client_order_id:
            json_data["client_order_id"] = request.client_order_id
        
        result = await self._make_request(
            method="POST",
            path="/trade-api/v2/portfolio/orders",
            json_data=json_data,
        )
        
        return Order.model_validate(result.get("order", result))
    
    async def cancel_order(self, order_id: str) -> Order:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancelled order
        """
        result = await self._make_request(
            method="DELETE",
            path=f"/trade-api/v2/portfolio/orders/{order_id}",
        )
        
        return Order.model_validate(result.get("order", result))
    
    async def get_order(self, order_id: str) -> Order:
        """
        Get order details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order details
        """
        result = await self._make_request(
            method="GET",
            path=f"/trade-api/v2/portfolio/orders/{order_id}",
        )
        
        return Order.model_validate(result.get("order", result))
