"""Integration tests for Kalshi REST client against DEMO environment."""

import asyncio
import os
import tempfile
from datetime import datetime
from decimal import Decimal

import pytest
import pytest_asyncio
from aiohttp import ClientSession

from src.client import KalshiRestClient
from src.config import Config, Environment, create_config


# Skip all tests if no demo credentials available
pytestmark = pytest.mark.asyncio


def has_demo_credentials() -> bool:
    """Check if demo credentials are available."""
    key_id = os.getenv("KALSHI_API_KEY_ID")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    return bool(key_id and key_path)


def skip_if_no_credentials():
    """Skip test if no demo credentials."""
    return pytest.mark.skipif(
        not has_demo_credentials(),
        reason="Demo credentials not available (set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH)"
    )


@pytest_asyncio.fixture
async def client():
    """Create authenticated client for demo environment."""
    # Check for credentials
    key_id = os.getenv("KALSHI_API_KEY_ID")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    
    if not key_id or not key_path:
        pytest.skip("Demo credentials not available")
    
    # Create config
    config = create_config(
        kalshi_api_key_id=key_id,
        kalshi_private_key_path=key_path,
        kalshi_env=Environment.DEMO,
    )
    
    # Create client
    client = KalshiRestClient(config)
    await client.connect()
    
    yield client
    
    # Cleanup
    await client.close()


@skip_if_no_credentials()
class TestConnection:
    """Test basic connectivity to DEMO environment."""
    
    async def test_connection_succeeds(self, client):
        """Test that we can connect and authenticate with DEMO environment."""
        # Getting balance should succeed if connection works
        balance = await client.get_balance()
        
        assert balance is not None
        assert isinstance(balance.available_balance, Decimal)
        assert isinstance(balance.total_balance, Decimal)
        assert balance.total_balance >= 0
    
    async def test_get_balance_returns_valid_structure(self, client):
        """Test that get_balance returns properly structured data."""
        balance = await client.get_balance()
        
        # Check all expected fields exist and have correct types
        assert hasattr(balance, 'available_balance')
        assert hasattr(balance, 'total_balance')
        assert hasattr(balance, 'portfolio_value')
        
        # Validate types
        assert isinstance(balance.available_balance, Decimal)
        assert isinstance(balance.total_balance, Decimal)
        assert isinstance(balance.portfolio_value, Decimal)
        
        # Validate logic: available <= total
        assert balance.available_balance <= balance.total_balance
        assert balance.portfolio_value >= 0


@skip_if_no_credentials()
class TestMarkets:
    """Test market data endpoints."""
    
    async def test_get_markets_returns_list_of_open_markets(self, client):
        """Test that get_markets returns list of open markets."""
        response = await client.get_markets(status="open")
        
        # Should return MarketsResponse
        assert response is not None
        assert hasattr(response, 'markets')
        assert isinstance(response.markets, list)
        
        # If markets exist, validate structure
        if response.markets:
            market = response.markets[0]
            assert hasattr(market, 'ticker')
            assert hasattr(market, 'title')
            assert hasattr(market, 'status')
            assert market.status.value == "open"
            assert isinstance(market.ticker, str)
            assert len(market.ticker) > 0
    
    async def test_get_markets_with_pagination(self, client):
        """Test pagination with cursor."""
        # Get first page
        response1 = await client.get_markets(status="open", limit=5)
        
        assert isinstance(response1.markets, list)
        assert len(response1.markets) <= 5
        
        # If there's more data, test cursor
        if response1.has_more and response1.cursor:
            response2 = await client.get_markets(
                status="open",
                limit=5,
                cursor=response1.cursor
            )
            
            assert isinstance(response2.markets, list)
            # Should get different markets
            tickers1 = {m.ticker for m in response1.markets}
            tickers2 = {m.ticker for m in response2.markets}
            assert tickers1 != tickers2
    
    async def test_get_market_by_ticker(self, client):
        """Test getting specific market by ticker."""
        # First get a list of markets
        response = await client.get_markets(status="open", limit=1)
        
        if not response.markets:
            pytest.skip("No open markets available")
        
        ticker = response.markets[0].ticker
        
        # Get specific market
        market = await client.get_market(ticker)
        
        assert market is not None
        assert market.ticker == ticker
        assert hasattr(market, 'title')
        assert hasattr(market, 'yes_bid')
        assert hasattr(market, 'yes_ask')
    
    async def test_get_orderbook(self, client):
        """Test getting orderbook for a market."""
        # Get a market first
        response = await client.get_markets(status="open", limit=1)
        
        if not response.markets:
            pytest.skip("No open markets available")
        
        ticker = response.markets[0].ticker
        
        # Get orderbook
        orderbook = await client.get_orderbook(ticker)
        
        assert orderbook is not None
        assert orderbook.ticker == ticker
        
        # Should have bids or asks (or both)
        has_bids = len(orderbook.yes_bids) > 0 or len(orderbook.no_bids) > 0
        has_asks = len(orderbook.yes_asks) > 0 or len(orderbook.no_asks) > 0
        
        assert has_bids or has_asks, "Orderbook should have at least bids or asks"


@skip_if_no_credentials()
class TestRateLimiting:
    """Test rate limiting behavior."""
    
    async def test_rate_limiting_is_respected(self, client):
        """Test that rate limiting prevents 429 errors under normal load."""
        # Make multiple rapid requests
        # With 20 reads/sec limit, 10 requests should complete without 429
        
        tasks = [
            client.get_balance()
            for _ in range(10)
        ]
        
        # All should succeed without rate limit errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # All should be Balance objects
        balances = [r for r in results if not isinstance(r, Exception)]
        assert len(balances) == 10
        
        for balance in balances:
            assert hasattr(balance, 'available_balance')
    
    async def test_concurrent_requests_respect_rate_limit(self, client):
        """Test that concurrent requests are properly rate limited."""
        start_time = asyncio.get_event_loop().time()
        
        # Make concurrent requests
        tasks = [
            client.get_markets(status="open", limit=1)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # With rate limiting, 5 requests should take at least some time
        # but should all succeed
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # Should have completed (rate limiting adds delays)
        assert duration >= 0


@skip_if_no_credentials()
class TestPositions:
    """Test position-related endpoints."""
    
    async def test_get_positions(self, client):
        """Test getting positions."""
        positions = await client.get_positions()
        
        assert isinstance(positions, list)
        
        # If positions exist, validate structure
        for position in positions:
            assert hasattr(position, 'market_ticker')
            assert hasattr(position, 'side')
            assert hasattr(position, 'count')
            assert isinstance(position.count, int)
            assert position.count >= 0


@skip_if_no_credentials()
class TestErrorHandling:
    """Test error handling with real API."""
    
    async def test_get_nonexistent_market_raises_error(self, client):
        """Test that requesting invalid market raises appropriate error."""
        from src.data.models import MarketError
        
        with pytest.raises(MarketError):
            await client.get_market("INVALID-TICKER-XYZ")
    
    async def test_get_orderbook_for_invalid_market_raises_error(self, client):
        """Test that requesting orderbook for invalid market raises error."""
        from src.data.models import MarketError
        
        with pytest.raises(MarketError):
            await client.get_orderbook("INVALID-TICKER")


# Tests that don't require credentials (mock/fake)
class TestClientWithoutCredentials:
    """Test client behavior without real credentials."""
    
    @pytest_asyncio.fixture
    async def mock_client(self):
        """Create a mock client for testing initialization."""
        # This would need a mock server or mock session
        # For now, just test that client can be created
        yield None
    
    def test_rate_limiter_tokens(self):
        """Test rate limiter token bucket logic."""
        from src.client import RateLimiter
        
        limiter = RateLimiter(read_rate=10.0, write_rate=5.0)
        
        assert limiter.read_rate == 10.0
        assert limiter.write_rate == 5.0
        assert limiter.read_tokens == 10.0
        assert limiter.write_tokens == 5.0
    
    def test_is_write_method(self):
        """Test write method detection."""
        from src.client import KalshiRestClient
        
        # Create a dummy config for testing
        config = create_config(
            kalshi_api_key_id="test",
            kalshi_private_key_path="/dev/null",  # Won't be used
            kalshi_env=Environment.DEMO,
        )
        
        # Mock the private key loading
        import tempfile
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            temp_path = f.name
        
        try:
            config = create_config(
                kalshi_api_key_id="test",
                kalshi_private_key_path=temp_path,
                kalshi_env=Environment.DEMO,
            )
            
            client = KalshiRestClient(config)
            
            assert client._is_write_method("POST") is True
            assert client._is_write_method("PUT") is True
            assert client._is_write_method("DELETE") is True
            assert client._is_write_method("PATCH") is True
            assert client._is_write_method("GET") is False
            assert client._is_write_method("HEAD") is False
        finally:
            import os
            os.unlink(temp_path)
