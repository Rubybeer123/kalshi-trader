"""Shared fixtures for Kalshi Trading Bot tests."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sample_private_key() -> rsa.RSAPrivateKey:
    """Generate a sample RSA private key for testing."""
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )


@pytest.fixture(scope="session")
def sample_private_key_pem(sample_private_key) -> str:
    """Get PEM-encoded private key."""
    return sample_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')


@pytest.fixture
def temp_key_file(sample_private_key_pem) -> Generator[str, None, None]:
    """Create a temporary key file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
        f.write(sample_private_key_pem)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def mock_config(temp_key_file):
    """Create a mock configuration for testing."""
    from src.config import create_config, Environment
    
    return create_config(
        kalshi_api_key_id="test-key-id-12345",
        kalshi_private_key_path=temp_key_file,
        kalshi_env=Environment.DEMO,
    )


@pytest.fixture
def mock_market_data():
    """Sample market data for testing."""
    return {
        "ticker": "KXBTC15M-26FEB150330-30",
        "title": "BTC $30K 15-min",
        "category": "crypto",
        "status": "open",
        "yes_bid": 0.60,
        "yes_ask": 0.65,
        "no_bid": 0.35,
        "no_ask": 0.40,
        "volume": 10000,
        "open_interest": 5000
    }


@pytest.fixture
def mock_orderbook_data():
    """Sample orderbook data for testing."""
    return {
        "ticker": "KXBTC15M-26FEB150330-30",
        "yes_bids": [
            {"price": 0.60, "count": 100},
            {"price": 0.59, "count": 200},
        ],
        "yes_asks": [
            {"price": 0.65, "count": 150},
            {"price": 0.66, "count": 100},
        ],
        "no_bids": [
            {"price": 0.35, "count": 100},
            {"price": 0.34, "count": 200},
        ],
        "no_asks": [
            {"price": 0.40, "count": 150},
            {"price": 0.41, "count": 100},
        ]
    }


@pytest.fixture
def sample_candles():
    """Sample OHLCV candles for testing."""
    from src.candle_aggregator import OHLCV
    
    base_time = datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc)
    
    candles = []
    for i in range(30):
        # Create varied candles for testing
        base_price = 0.50 + (i % 10 - 5) * 0.01
        candle = OHLCV(
            market_ticker="KXBTC15M-26FEB150330-30",
            timestamp=base_time + timedelta(minutes=15*i),
            open=base_price,
            high=base_price + 0.02,
            low=base_price - 0.02,
            close=base_price + (0.01 if i % 2 == 0 else -0.01),
            volume=1000 + i * 100
        )
        candles.append(candle)
    
    return candles


@pytest.fixture
def extreme_candles():
    """Candles with extreme movements for signal testing."""
    from src.candle_aggregator import OHLCV
    
    base_time = datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc)
    
    return [
        # Normal candle
        OHLCV("KXBTC", base_time, 0.50, 0.52, 0.48, 0.51, 1000),
        # Extreme down (body below bands)
        OHLCV("KXBTC", base_time + timedelta(minutes=15), 0.30, 0.35, 0.25, 0.30, 2000),
        # Extreme up (body above bands)
        OHLCV("KXBTC", base_time + timedelta(minutes=30), 0.70, 0.75, 0.65, 0.70, 2000),
        # Recovery
        OHLCV("KXBTC", base_time + timedelta(minutes=45), 0.50, 0.55, 0.45, 0.52, 1500),
    ]


@pytest.fixture
def mock_kalshi_responses():
    """Mock responses from Kalshi API."""
    return {
        "markets": {
            "markets": [
                {
                    "ticker": "KXBTC15M-26FEB150330-30",
                    "title": "BTC $30K",
                    "status": "open",
                    "yes_bid": 60,  # in cents
                    "yes_ask": 65,
                    "no_bid": 35,
                    "no_ask": 40,
                }
            ],
            "cursor": None
        },
        "order": {
            "order_id": "test-order-123",
            "client_order_id": "client-123",
            "ticker": "KXBTC15M-26FEB150330-30",
            "side": "yes",
            "price": 65,
            "count": 10,
            "status": "filled",
            "filled_count": 10,
            "remaining_count": 0,
        },
        "balance": {
            "available_balance": 5000.0,
            "total_balance": 10000.0,
            "portfolio_value": 5000.0
        },
        "positions": [
            {
                "market_ticker": "KXBTC15M-26FEB150330-30",
                "side": "yes",
                "count": 10,
                "avg_entry_price": 0.60
            }
        ]
    }


@pytest.fixture
def temp_database() -> Generator[str, None, None]:
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_websocket_messages():
    """Sample WebSocket messages for testing."""
    return {
        "trade": {
            "type": "trade",
            "channel": "trade",
            "seq": 1,
            "data": {
                "ticker": "KXBTC15M-26FEB150330-30",
                "price": 0.65,
                "count": 100,
                "created_time": "2024-02-26T15:30:00Z"
            }
        },
        "ticker": {
            "type": "ticker",
            "channel": "ticker",
            "seq": 2,
            "data": {
                "ticker": "KXBTC15M-26FEB150330-30",
                "yes_bid": 60,
                "yes_ask": 65,
                "no_bid": 35,
                "no_ask": 40,
                "last_price": 65
            }
        },
        "heartbeat": {
            "type": "heartbeat"
        },
        "pong": {
            "type": "pong"
        }
    }


@pytest.fixture
def sample_signal():
    """Sample trading signal for testing."""
    from src.strategies.base import Signal, SignalType
    
    return Signal(
        type=SignalType.LONG,
        market_ticker="KXBTC15M-26FEB150330-30",
        entry_price=0.60,
        stop_loss=0.50,
        take_profit=0.80,
        confidence=0.75,
        metadata={"strategy": "BollingerScalper", "rr_ratio": 2.0}
    )


@pytest.fixture
def sample_trade():
    """Sample completed trade for testing."""
    from src.performance_tracker import Trade
    
    return Trade(
        id=1,
        timestamp=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
        market_ticker="KXBTC15M-26FEB150330-30",
        side="yes",
        entry_price=0.60,
        exit_price=0.70,
        contracts=10,
        pnl=1.0,
        exit_reason="target"
    )


@pytest.fixture
def winning_trades():
    """Sample winning trades for metrics testing."""
    from src.performance_tracker import Trade
    
    base_time = datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc)
    
    return [
        Trade(timestamp=base_time + timedelta(hours=i), pnl=100.0)
        for i in range(6)
    ]


@pytest.fixture
def losing_trades():
    """Sample losing trades for metrics testing."""
    from src.performance_tracker import Trade
    
    base_time = datetime(2024, 2, 26, 15, 0, tzinfo=timezone.utc)
    
    return [
        Trade(timestamp=base_time + timedelta(hours=i), pnl=-50.0)
        for i in range(6, 10)
    ]


@pytest.fixture
def mixed_trades(winning_trades, losing_trades):
    """Mixed winning and losing trades."""
    return winning_trades + losing_trades


# Import timedelta for fixtures
from datetime import timedelta
