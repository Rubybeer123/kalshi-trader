# Demo Environment Validation

This directory contains automated validation for the Kalshi trading bot in the demo environment.

## Overview

The validation system runs the trading bot against the Kalshi **demo API** (paper trading) to verify:

- API connectivity and latency
- Order execution functionality
- Strategy performance vs backtest expectations
- Error handling and recovery
- Overall system stability

## Quick Start

### Run Validation

```bash
# Run with defaults (100 trades, 7 days)
./scripts/run_validation.sh

# Custom thresholds
./scripts/run_validation.sh --min-trades 50 --min-days 3

# With backtest comparison
./scripts/run_validation.sh --backtest-results backtest_results.json
```

### Environment Variables

```bash
export KALSHI_API_KEY_ID="your_key_id"
export KALSHI_PRIVATE_KEY_PATH="/path/to/private_key.pem"
```

## Validation Criteria

| Criteria | Threshold | Description |
|----------|-----------|-------------|
| **Minimum Trades** | 100 | Must execute at least 100 trades |
| **Minimum Duration** | 7 days | Must run for at least 7 days |
| **API Latency** | < 200ms | Average API latency must be under 200ms |
| **Win Rate** | ± 5% | Within 5% of backtest expectation |
| **Error Rate** | < 1% | API error rate must be under 1% |
| **Critical Errors** | 0 | No critical errors allowed |

## Output

### Report Files

- `validation_report_*.json` - Detailed JSON report
- `validation_summary_*.txt` - Human-readable summary
- `progress_*.json` - Daily progress snapshots
- `validation_performance.db` - SQLite database with all trades

### Report Structure

```json
{
  "validation_summary": {
    "status": "PASSED|FAILED|INCOMPLETE",
    "start_time": "2024-01-01T00:00:00+00:00",
    "duration_hours": 168.5,
    "stopped_reason": "trade_target_reached"
  },
  "trade_statistics": {
    "total_trades": 150,
    "win_rate": 0.55,
    "total_pnl": 250.50
  },
  "api_performance": {
    "overall_avg_latency_ms": 125.5,
    "latency_requirement_met": true
  },
  "backtest_comparison": {
    "expected_win_rate": 0.52,
    "actual_win_rate": 0.55,
    "within_tolerance": true
  },
  "acceptance_criteria": {
    "criteria_met": {
      "trade_count": true,
      "duration": true,
      "latency": true,
      "no_critical_errors": true
    }
  }
}
```

## Alert Types

The validation system generates alerts for various conditions:

### Critical Alerts
- API connection failures
- Order execution errors
- High error rates (>1%)
- Trading loop failures

### Warning Alerts
- High API latency (>200ms)
- Low win rate (<50%)
- Win rate deviation from backtest (>5%)

### Info Alerts
- Progress milestones
- Configuration updates
- Status changes

## Cron Setup

To run validation automatically on a schedule:

```bash
# Edit crontab
crontab -e

# Run daily at 9 AM
0 9 * * * cd /path/to/kalshi_trader && ./scripts/run_validation.sh --output-dir /var/log/kalshi_validation

# Or use openclaw cron
openclaw cron add --name "kalshi-validation" \
  --schedule "0 9 * * *" \
  --command "./scripts/run_validation.sh"
```

## Python API

```python
from src.validation import DemoValidator

# Create validator
validator = DemoValidator(
    api_key_id="your_key",
    private_key_path="/path/to/key.pem",
    backtest_results={"win_rate": 0.52},
    output_dir="validation_output"
)

# Run validation
try:
    await validator.start()
except KeyboardInterrupt:
    await validator.stop("user_interrupt")

# Check results
print(f"Total trades: {validator.result.total_trades}")
print(f"Win rate: {validator.result.win_rate:.1%}")
print(f"Alerts: {len(validator.result.alerts)}")
```

## Troubleshooting

### API Latency Issues

If API latency exceeds 200ms:
1. Check network connection
2. Verify Kalshi demo API status
3. Consider geographic proximity to API servers

### Low Win Rate

If win rate is significantly below backtest:
1. Review strategy parameters
2. Check market conditions (trending vs ranging)
3. Verify data feed quality

### Order Execution Failures

If orders are failing:
1. Check account balance in demo environment
2. Verify market hours and availability
3. Review rate limiting

## Backtest Comparison

To compare validation results against backtest:

1. Run backtest first:
```python
from src.backtester import Backtester
backtester.run()
results = backtester.get_last_result()
# Save to JSON
```

2. Pass to validation:
```bash
./scripts/run_validation.sh --backtest-results backtest_results.json
```

The validation will compare:
- Win rate (±5% tolerance)
- Expectancy
- Trade frequency
- Risk-adjusted returns

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Validation passed |
| 1 | Validation failed or error occurred |

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Demo Validation
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run validation
        env:
          KALSHI_API_KEY_ID: ${{ secrets.KALSHI_DEMO_KEY }}
          KALSHI_PRIVATE_KEY_PATH: ${{ secrets.KALSHI_PRIVATE_KEY_PATH }}
        run: |
          ./scripts/run_validation.sh --min-trades 50 --min-days 1
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-reports
          path: validation_output/
```

## Support

For issues or questions:
1. Check the logs in `validation_output/`
2. Review alert messages
3. Examine the performance database
4. Open an issue on GitHub
