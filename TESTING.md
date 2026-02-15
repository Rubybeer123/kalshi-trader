# Testing Guide for Kalshi Trading Bot

## Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Using the Makefile
make test
make test-unit
make test-coverage
```

## Test Structure

```
tests/
├── conftest.py                   # Shared fixtures
├── fixtures/                     # Sample data files
│   ├── markets.json
│   ├── orderbook.json
│   ├── candles.json
│   ├── responses.json
│   └── websocket.json
├── test_auth.py                  # RSA-PSS authentication
├── test_config.py                # Configuration
├── test_client_integration.py    # REST/WebSocket clients
├── test_websocket_client.py      # WebSocket specific
├── test_candle_aggregator.py     # OHLCV aggregation
├── test_bollinger_bands.py       # Technical indicator
├── test_market_discovery.py      # Market parsing
├── test_data_manager.py          # Data orchestration
├── test_bollinger_scalper.py     # Trading strategy
├── test_strategy_base.py         # Strategy base class
├── test_risk_manager.py          # Risk management
├── test_execution_engine.py      # Order execution
├── test_paper_trading.py         # Paper trading
├── test_performance_tracker.py   # Performance metrics
└── test_trading_bot_integration.py  # End-to-end tests
```

## Test Categories

### Unit Tests (Fast)
- Run in milliseconds
- No network access
- No external dependencies
- Use mocks extensively

```bash
pytest -m "not integration" -v
```

### Integration Tests (Slow)
- Test component interactions
- May use temporary databases
- Test actual async behavior
- Full end-to-end flows

```bash
pytest -m "integration" -v
```

## Fixtures

### Key Fixtures in conftest.py

| Fixture | Description |
|---------|-------------|
| `sample_private_key` | RSA key for auth tests |
| `temp_key_file` | Temporary PEM file |
| `mock_config` | Test configuration |
| `mock_market_data` | Sample market JSON |
| `mock_orderbook_data` | Sample orderbook JSON |
| `sample_candles` | 30 sample OHLCV candles |
| `extreme_candles` | Candles with extreme moves |
| `mock_kalshi_responses` | API response templates |
| `temp_database` | SQLite test database |
| `mock_websocket_messages` | WS message templates |
| `sample_signal` | Trading signal |
| `sample_trade` | Completed trade |
| `winning_trades` | 6 winning trades |
| `losing_trades` | 4 losing trades |
| `mixed_trades` | Mixed win/loss trades |

## Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| Auth | 100% | ? |
| Config | 90% | ? |
| Client | 90% | ? |
| WebSocket | 85% | ? |
| Data Models | 95% | ? |
| Candle Aggregator | 90% | ? |
| Bollinger Bands | 90% | ? |
| Market Discovery | 90% | ? |
| Data Manager | 85% | ? |
| Strategy Base | 95% | ? |
| BollingerScalper | 95% | ? |
| Risk Manager | 90% | ? |
| Execution Engine | 90% | ? |
| Paper Trading | 90% | ? |
| Performance Tracker | 90% | ? |
| **Overall** | **90%** | **?** |

## Running Specific Tests

```bash
# Test authentication
pytest tests/test_auth.py -v

# Test with coverage for specific file
pytest tests/test_bollinger_scalper.py --cov=src.strategies.bollinger_scalper --cov-report=term-missing

# Run specific test method
pytest tests/test_auth.py::TestKalshiAuthenticator::test_sign_pss_text -v

# Run failed tests only
pytest --lf

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Mocking Examples

### Mock REST Client
```python
@pytest.mark.asyncio
async def test_with_mock_client(mocker):
    mock_rest = mocker.MagicMock()
    mock_rest.get_markets = mocker.AsyncMock(return_value=mock_response)
    
    bot = TradingBot(config, exchange=mock_rest)
    # Test with mocked client
```

### Mock Time with freezegun
```python
from freezegun import freeze_time

@freeze_time("2024-02-26 15:30:00")
def test_time_dependent_logic():
    # Time is frozen at specified moment
    assert datetime.now() == datetime(2024, 2, 26, 15, 30, 00)
```

## Debugging Tests

```bash
# Stop on first failure
pytest -x

# Show local variables on failure
pytest -v --showlocals

# Enter debugger on failure
pytest --pdb

# Full traceback
pytest -v --tb=long

# Capture output even on success
pytest -v -s
```

## Performance Testing

```bash
# Profile test execution time
pytest --durations=10

# Run slow tests only
pytest -m slow

# Skip slow tests
pytest -m "not slow"
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Test Data Files

### fixtures/markets.json
Sample market data for testing market discovery and parsing.

### fixtures/orderbook.json
Sample orderbook data for fill simulation and price calculations.

### fixtures/candles.json
Sample OHLCV candles including extreme movements for signal testing.

### fixtures/responses.json
Mock Kalshi API responses for unit testing without network calls.

### fixtures/websocket.json
Sample WebSocket messages for connection and message handling tests.

## Common Issues

### Async Test Failures
Make sure to use `@pytest.mark.asyncio` decorator:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result
```

### Database Locking
Each test gets its own temp database via `temp_database` fixture.

### Timezone Issues
All fixtures use `timezone.utc` explicitly.

### Event Loop Warnings
Configured in `pytest.ini` with `asyncio_mode = auto`.

## Adding New Tests

1. Create test file: `tests/test_new_feature.py`
2. Import fixtures from `conftest.py`
3. Add tests following naming convention `test_*`
4. Use appropriate markers:
   - `@pytest.mark.unit` for fast tests
   - `@pytest.mark.integration` for slow tests
   - `@pytest.mark.slow` for long-running tests
5. Run tests: `pytest tests/test_new_feature.py -v`
6. Check coverage: `pytest --cov=src.new_feature tests/test_new_feature.py`

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [freezegun](https://github.com/spulec/freezegun)
