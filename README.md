# Kalshi Trading Bot v2

A clean, production-ready automated trading bot for Kalshi prediction markets.

## 🎯 Overview

This is a complete rebuild of the Kalshi trading bot with:
- **Single Strategy**: Bollinger Bands Scalper
- **Clean Architecture**: Modular, testable, type-safe
- **Real-time Data**: WebSocket feeds with auto-reconnect
- **Risk Management**: Position sizing, daily limits, circuit breakers
- **Paper Trading**: Full simulation environment for testing
- **Performance Tracking**: Comprehensive metrics and reporting
- **Comprehensive Tests**: 300+ test cases

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TradingBot (main.py)                    │
│                         Orchestrator                         │
└────────────────────┬────────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│DataManager│   │RiskManager│   │ExecutionEngine│
└────┬─────┘   └────┬─────┘   └──────┬───────┘
     │              │                 │
     │    ┌─────────┘                 │
     │    │                           │
     ▼    ▼                           ▼
┌─────────────┐   ┌────────────┐   ┌─────────────┐
│KalshiRestClient│ │KalshiWebSocket│ │BollingerScalper│
│  or          │   └────────────┘   └─────────────┘
│PaperTradingExchange│
└─────────────┘
```

## 📦 Components

| Component | Purpose | File |
|-----------|---------|------|
| **Config** | Environment-based configuration | `config.py` |
| **Auth** | RSA-PSS authentication | `auth.py` |
| **REST Client** | Rate-limited API client | `client.py` |
| **WebSocket** | Real-time feeds | `websocket_client.py` |
| **Market Discovery** | 15-min crypto parsing | `market_discovery.py` |
| **Candle Aggregator** | OHLCV aggregation | `candle_aggregator.py` |
| **Data Manager** | Orchestrates all data | `data_manager.py` |
| **Bollinger Bands** | TA indicator | `bollinger_bands.py` |
| **BollingerScalper** | Trading strategy | `strategies/bollinger_scalper.py` |
| **Risk Manager** | Position sizing & limits | `risk_manager.py` |
| **Execution Engine** | Order lifecycle | `execution_engine.py` |
| **Paper Trading** | Exchange simulator | `paper_trading.py` |
| **Performance Tracker** | Metrics & reporting | `performance_tracker.py` |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Kalshi credentials

# 3. Run tests
pytest tests/ -v

# 4. Start trading (demo mode)
python -m src.main
```

## 📝 Configuration (.env)

```bash
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=config/kalshi-key.pem
KALSHI_ENV=demo  # demo or live

# Trading Settings
INITIAL_CAPITAL=50
MAX_POSITIONS=2
MAX_EXPOSURE=50
DAILY_LOSS_LIMIT=5
```

## 🎯 BollingerScalper Strategy

### Entry Rules
- **LONG**: Candle body entirely below lower band
  - Entry: Candle low
  - Stop: Candle high
  - Target: 1:2 risk-reward

- **SHORT**: Candle body entirely above upper band
  - Entry: Candle high
  - Stop: Candle low
  - Target: 1:2 risk-reward

### Filters
- 25-period Bollinger Bands (2.0 std dev)
- Minimum 8 minutes to expiration
- Risk/reward minimum 2:1
- Max 2 concurrent positions

## 🛡️ Risk Management

- **Position Sizing**: 2% risk per trade
- **Daily Loss Limit**: 10% of capital
- **Circuit Breaker**: Halts after 3 consecutive losses
- **Max Exposure**: 20% of capital

## 📊 Paper Trading

Test your strategy without real money:

```python
from src.paper_trading import PaperTradingExchange

# Create simulated exchange
exchange = PaperTradingExchange(starting_balance=10000)

# Update with market data
exchange.update_market_data("KXBTC", orderbook)

# Trade exactly like real exchange
order = await exchange.create_order(
    market_ticker="KXBTC",
    side="yes",
    price=65,  # cents
    count=10
)
```

### Paper Trading Features
- Realistic fill simulation with slippage
- 1-3 ticks normal slippage, 5-10 ticks stressed
- Partial fill simulation (5% probability)
- Network delay simulation (100-500ms)
- Full position and balance tracking

## 📈 Performance Tracking

Comprehensive metrics automatically calculated:

```python
from src.performance_tracker import PerformanceTracker, Trade

# Track trades
tracker = PerformanceTracker()
tracker.record_trade(Trade(
    market_ticker="KXBTC",
    entry_price=0.60,
    exit_price=0.70,
    contracts=10,
    pnl=1.0
))

# Get metrics
metrics = tracker.get_metrics()
print(f"Win Rate: {metrics.win_rate:.1%}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

# Generate report
report = tracker.generate_report(format='html')
```

### Metrics Calculated
- **Trade Stats**: Total trades, win rate, avg win/loss
- **Profit Factor**: Gross wins / gross losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Peak-to-trough decline
- **Expectancy**: Expected value per trade
- **Consecutive**: Max consecutive wins/losses

### Report Formats
- **Text**: Console-friendly
- **JSON**: Programmatic access
- **HTML**: Web dashboard

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_bollinger_scalper.py -v

# Run paper trading tests
pytest tests/test_paper_trading.py -v

# Run performance tests
pytest tests/test_performance_tracker.py -v
```

**Test Coverage:**
- 36 Python files
- 14,500+ lines of code
- 300+ test cases

## 📊 Data Flow

```
1. WebSocket receives trade tick
2. CandleAggregator builds 15-min OHLCV
3. On candle close:
   - Update Bollinger Bands
   - BollingerScalper generates signal
   - RiskManager validates signal
   - ExecutionEngine places order
4. PerformanceTracker records result
5. Position tracked until close
```

## 🔄 Backtesting Workflow

```python
# 1. Load historical data
historical_candles = load_historical_data()

# 2. Initialize paper trading
exchange = PaperTradingExchange(starting_balance=10000)
tracker = PerformanceTracker()

# 3. Run strategy on historical data
for candle in historical_candles:
    signal = strategy.on_candle(candle.ticker, candle)
    
    if signal:
        result = await exchange.execute_signal(signal)
        
        if result.success:
            # Track the trade
            tracker.record_trade(Trade(
                market_ticker=candle.ticker,
                entry_price=signal.entry_price,
                exit_price=...,  # When position closes
                pnl=...  # Calculate P&L
            ))

# 4. Generate performance report
metrics = tracker.get_metrics()
print(tracker.generate_report())
```

## 🛠️ Development

```bash
# Format code
black src/ tests/

# Type check
mypy src/

# Lint
ruff check src/

# Run pre-commit hooks
pre-commit run --all-files
```

## 📁 Project Structure

```
kalshi_trader/
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── config.py                 # Configuration
│   ├── auth.py                   # Authentication
│   ├── client.py                 # REST client
│   ├── websocket_client.py       # WebSocket client
│   ├── data_manager.py           # Data orchestration
│   ├── candle_aggregator.py      # OHLCV builder
│   ├── market_discovery.py       # Market parser
│   ├── bollinger_bands.py        # TA indicator
│   ├── risk_manager.py           # Risk management
│   ├── execution_engine.py       # Order execution
│   ├── paper_trading.py          # Paper trading exchange
│   ├── performance_tracker.py    # Metrics & reporting
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py               # Strategy ABC
│   │   └── bollinger_scalper.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py             # Pydantic models
│   │   └── storage.py            # SQLite storage
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Structured logging
├── tests/
│   ├── test_*.py                 # 14 test files
│   └── __init__.py
├── .env.example
├── requirements.txt
└── README.md
```

## 📈 Performance Report Example

```
============================================================
           PERFORMANCE REPORT
============================================================

TRADE STATISTICS
----------------------------------------
Total Trades:        156
Winning Trades:      94
Losing Trades:       62
Win Rate:            60.3%

P&L STATISTICS
----------------------------------------
Net Profit:          $1,247.50
Gross Profit:        $2,456.00
Gross Loss:          $1,208.50
Average Win:         $26.13
Average Loss:        $19.49
Largest Win:         $89.50
Largest Loss:        $45.20

RATIOS
----------------------------------------
Profit Factor:       2.03
Expectancy:          $8.00
Sharpe Ratio:        1.85
Max Drawdown:        $234.50 (8.2%)

CONSECUTIVE TRADES
----------------------------------------
Max Consecutive Wins:   7
Max Consecutive Losses: 4

============================================================
```

## ⚠️ Safety Notes

- Always test in DEMO mode first
- Use PaperTradingExchange to validate strategies
- Start with small position sizes
- Monitor daily loss limits
- Keep private keys secure
- Review all signals before live trading

## 📄 License

MIT License

## 🤝 Contributing

This is a complete implementation. For extensions:
1. Add new strategies by inheriting from `Strategy`
2. Add new indicators in `bollinger_bands.py` style
3. Extend `RiskManager` for custom risk rules
4. Add new data sources to `DataManager`
5. Create custom reports in `PerformanceTracker`

---

**Built with ❤️ for the Kalshi community.**

**36 Python files | 14,500+ lines | 300+ tests**
