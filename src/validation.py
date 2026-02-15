"""Demo environment validation script for Kalshi trading bot.

This script runs automated validation tests against the Kalshi demo API
to verify the trading bot implementation before live deployment.
"""

import asyncio
import json
import signal
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import structlog

from src.auth import KalshiAuthenticator
from src.client import KalshiRestClient
from src.config import create_config, Environment, Config
from src.data.models import OrderStatus
from src.execution_engine import ExecutionEngine
from src.paper_trading import PaperTradingExchange
from src.performance_tracker import PerformanceTracker, Trade
from src.risk_manager import RiskConfig, RiskManager
from src.strategies.bollinger_scalper import BollingerScalper, ScalperConfig
from src.data_manager import DataManager

logger = structlog.get_logger(__name__)


@dataclass
class APIMetrics:
    """API performance metrics."""
    endpoint: str
    call_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    error_count: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count
    
    def record_call(self, latency_ms: float, error: bool = False) -> None:
        self.call_count += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        if error:
            self.error_count += 1
    
    @property
    def error_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.error_count / self.call_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'endpoint': self.endpoint,
            'call_count': self.call_count,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'min_latency_ms': round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0,
            'max_latency_ms': round(self.max_latency_ms, 2),
            'error_count': self.error_count,
            'error_rate': round(self.error_rate, 4),
        }


@dataclass
class ValidationAlert:
    """Validation alert."""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'performance', 'api', 'trading', 'system'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'category': self.category,
            'message': self.message,
            'details': self.details,
        }


@dataclass
class ValidationResult:
    """Complete validation results."""
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Performance
    total_pnl: float = 0.0
    avg_trade_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # API metrics
    api_metrics: Dict[str, APIMetrics] = field(default_factory=dict)
    
    # Alerts
    alerts: List[ValidationAlert] = field(default_factory=list)
    
    # Backtest comparison
    backtest_win_rate: Optional[float] = None
    backtest_expectancy: Optional[float] = None
    win_rate_deviation: Optional[float] = None
    
    # Status
    is_running: bool = True
    stopped_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_hours': self.duration_hours,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 4),
            'total_pnl': round(self.total_pnl, 2),
            'avg_trade_pnl': round(self.avg_trade_pnl, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'max_drawdown_pct': round(self.max_drawdown_pct, 4),
            'api_metrics': {k: v.to_dict() for k, v in self.api_metrics.items()},
            'alert_count': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a.severity == 'critical']),
            'backtest_comparison': {
                'backtest_win_rate': self.backtest_win_rate,
                'actual_win_rate': self.win_rate,
                'deviation': self.win_rate_deviation,
                'within_tolerance': abs(self.win_rate_deviation) <= 0.05 if self.win_rate_deviation else None,
            } if self.backtest_win_rate else None,
            'is_running': self.is_running,
            'stopped_reason': self.stopped_reason,
        }
    
    @property
    def duration_hours(self) -> float:
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds() / 3600
    
    def add_alert(self, severity: str, category: str, message: str, details: Optional[Dict] = None) -> None:
        alert = ValidationAlert(
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            category=category,
            message=message,
            details=details or {}
        )
        self.alerts.append(alert)
        
        # Log immediately
        log_fn = {
            'critical': logger.error,
            'warning': logger.warning,
            'info': logger.info,
        }.get(severity, logger.info)
        
        log_fn(f"[VALIDATION ALERT] {message}", severity=severity, category=category, **(details or {}))


class ValidatedRestClient(KalshiRestClient):
    """REST client that tracks API metrics for validation."""
    
    def __init__(self, config: Config, metrics: Dict[str, APIMetrics]):
        super().__init__(config)
        self._metrics = metrics
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        start_time = time.time()
        error = False
        
        # Initialize metrics for this endpoint if needed
        if endpoint not in self._metrics:
            self._metrics[endpoint] = APIMetrics(endpoint=endpoint)
        
        try:
            result = await super()._make_request(method, endpoint, **kwargs)
            return result
        except Exception as e:
            error = True
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._metrics[endpoint].record_call(latency_ms, error)


class DemoValidator:
    """
    Validates the trading bot implementation in demo environment.
    
    Features:
    - Runs paper trading for minimum trades/days
    - Monitors API latency and errors
    - Compares performance to backtest
    - Generates validation reports
    - Automated alerts for issues
    """
    
    # Validation thresholds
    MIN_TRADES = 100
    MIN_DAYS = 7
    MAX_LATENCY_MS = 200
    WIN_RATE_TOLERANCE = 0.05  # 5%
    MIN_WIN_RATE = 0.50  # 50%
    MAX_ERROR_RATE = 0.01  # 1%
    
    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        backtest_results: Optional[Dict] = None,
        output_dir: str = "validation_output"
    ):
        """
        Initialize validator.
        
        Args:
            api_key_id: Kalshi API key ID (defaults to env var)
            private_key_path: Path to private key file (defaults to env var)
            backtest_results: Expected backtest performance metrics
            output_dir: Directory for validation reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.backtest_results = backtest_results or {}
        self.result = ValidationResult(start_time=datetime.now(timezone.utc))
        
        # Config
        self.config = create_config(
            kalshi_api_key_id=api_key_id,
            kalshi_private_key_path=private_key_path,
            kalshi_env=Environment.DEMO,
        )
        
        # Components (initialized in start())
        self.auth: Optional[KalshiAuth] = None
        self.rest_client: Optional[ValidatedRestClient] = None
        self.exchange: Optional[PaperTradingExchange] = None
        self.risk_manager: Optional[RiskManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.data_manager: Optional[DataManager] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        
        # State
        self._is_running = False
        self._stop_event = asyncio.Event()
        self._daily_tasks: List[asyncio.Task] = []
        
        logger.info("DemoValidator initialized", output_dir=str(self.output_dir))
    
    async def start(self) -> None:
        """Start validation run."""
        logger.info("Starting demo validation")
        
        try:
            # Initialize validated REST client
            self.rest_client = ValidatedRestClient(
                config=self.config,
                metrics=self.result.api_metrics
            )
            
            await self.rest_client.connect()
            logger.info("Connected to Kalshi demo API")
            
            # Test API connectivity and measure latency
            await self._test_api_connectivity()
            
            # Initialize components
            self.exchange = PaperTradingExchange(starting_balance=10000.0)
            
            risk_config = RiskConfig(
                max_positions=2,
                max_risk_per_trade_pct=0.02,
                max_daily_loss_pct=0.10,
            )
            self.risk_manager = RiskManager(risk_config)
            
            self.execution_engine = ExecutionEngine(
                rest_client=self.rest_client,
                risk_manager=self.risk_manager
            )
            
            self.performance_tracker = PerformanceTracker(
                db_path=str(self.output_dir / "validation_performance.db")
            )
            
            # Initialize data manager with strategy
            strategy = BollingerScalper(ScalperConfig(bb_period=20, bb_std=2.0))
            self.data_manager = DataManager(
                rest_client=self.rest_client,
                strategy=strategy,
                markets=[],  # Will discover crypto markets
            )
            
            # Start execution engine
            await self.execution_engine.start()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self._is_running = True
            self.result.is_running = True
            
            # Start monitoring tasks
            self._daily_tasks = [
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._daily_report_loop()),
            ]
            
            # Wait for stop condition
            await self._stop_event.wait()
            
        except Exception as e:
            logger.exception("Validation failed to start")
            self.result.add_alert(
                'critical',
                'system',
                f'Validation failed to start: {str(e)}',
                {'error': str(e)}
            )
            raise
    
    async def stop(self, reason: str = "manual") -> None:
        """Stop validation run."""
        logger.info("Stopping validation", reason=reason)
        
        self._is_running = False
        self.result.is_running = False
        self.result.end_time = datetime.now(timezone.utc)
        self.result.stopped_reason = reason
        
        # Cancel tasks
        for task in self._daily_tasks:
            task.cancel()
        
        # Stop components
        if self.execution_engine:
            await self.execution_engine.stop()
        
        if self.rest_client:
            await self.rest_client.close()
        
        self._stop_event.set()
        
        # Generate final report
        self._generate_final_report()
        
        logger.info("Validation stopped", reason=reason)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handle_signal(sig, frame):
            logger.info(f"Received signal {sig}")
            asyncio.create_task(self.stop(f"signal_{sig}"))
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    async def _test_api_connectivity(self) -> None:
        """Test API connectivity and measure baseline latency."""
        logger.info("Testing API connectivity")
        
        # Test exchange status endpoint
        start = time.time()
        try:
            # Try to get balance as a simple test
            balance = await self.rest_client.get_balance()
            latency = (time.time() - start) * 1000
            
            logger.info(
                "API connectivity test passed",
                latency_ms=round(latency, 2),
                balance=float(balance.total_balance)
            )
            
            if latency > self.MAX_LATENCY_MS:
                self.result.add_alert(
                    'warning',
                    'api',
                    f'API latency ({latency:.0f}ms) exceeds threshold ({self.MAX_LATENCY_MS}ms)',
                    {'latency_ms': latency, 'threshold_ms': self.MAX_LATENCY_MS}
                )
                
        except Exception as e:
            logger.error("API connectivity test failed", error=str(e))
            raise
    
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        logger.info("Starting trading loop")
        
        try:
            # Discover 15-minute crypto markets
            await self.data_manager.discover_crypto_markets(duration_minutes=15)
            
            if not self.data_manager.markets:
                self.result.add_alert(
                    'critical',
                    'trading',
                    'No 15-minute crypto markets found',
                    {}
                )
                await self.stop('no_markets')
                return
            
            logger.info(
                "Discovered markets",
                count=len(self.data_manager.markets),
                markets=list(self.data_manager.markets.keys())
            )
            
            # Subscribe to market data
            await self.data_manager.start_websocket()
            
            while self._is_running:
                try:
                    # Check if we've hit our trade target
                    if self.result.total_trades >= self.MIN_TRADES:
                        logger.info(
                            f"Reached target of {self.MIN_TRADES} trades",
                            trades=self.result.total_trades
                        )
                        await self.stop('trade_target_reached')
                        break
                    
                    # Check duration
                    if self.result.duration_hours >= (self.MIN_DAYS * 24):
                        logger.info(
                            f"Reached target duration of {self.MIN_DAYS} days",
                            hours=self.result.duration_hours
                        )
                        await self.stop('duration_target_reached')
                        break
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error("Error in trading loop", error=str(e))
                    self.result.add_alert(
                        'warning',
                        'trading',
                        f'Trading loop error: {str(e)}',
                        {'error': str(e)}
                    )
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.exception("Trading loop failed")
            self.result.add_alert(
                'critical',
                'trading',
                f'Trading loop failed: {str(e)}',
                {'error': str(e)}
            )
            await self.stop('trading_error')
    
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop for metrics and alerts."""
        logger.info("Starting monitoring loop")
        
        try:
            while self._is_running:
                try:
                    await self._check_metrics()
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    logger.error("Error in monitoring loop", error=str(e))
                    await asyncio.sleep(30)
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
    
    async def _check_metrics(self) -> None:
        """Check metrics and generate alerts."""
        # Check API latency
        for endpoint, metrics in self.result.api_metrics.items():
            if metrics.call_count > 0:
                if metrics.avg_latency_ms > self.MAX_LATENCY_MS:
                    self.result.add_alert(
                        'warning',
                        'api',
                        f'High API latency for {endpoint}',
                        {
                            'endpoint': endpoint,
                            'avg_latency_ms': metrics.avg_latency_ms,
                            'threshold_ms': self.MAX_LATENCY_MS,
                        }
                    )
                
                if metrics.error_rate > self.MAX_ERROR_RATE:
                    self.result.add_alert(
                        'critical',
                        'api',
                        f'High error rate for {endpoint}',
                        {
                            'endpoint': endpoint,
                            'error_rate': metrics.error_rate,
                            'error_count': metrics.error_count,
                        }
                    )
        
        # Check trading performance
        if self.result.total_trades >= 20:  # Only check after some trades
            if self.result.win_rate < self.MIN_WIN_RATE:
                self.result.add_alert(
                    'warning',
                    'performance',
                    f'Win rate ({self.result.win_rate:.1%}) below minimum ({self.MIN_WIN_RATE:.1%})',
                    {
                        'win_rate': self.result.win_rate,
                        'min_win_rate': self.MIN_WIN_RATE,
                        'trades': self.result.total_trades,
                    }
                )
            
            # Compare to backtest
            if self.backtest_results.get('win_rate'):
                expected = self.backtest_results['win_rate']
                actual = self.result.win_rate
                deviation = abs(actual - expected)
                self.result.backtest_win_rate = expected
                self.result.win_rate_deviation = actual - expected
                
                if deviation > self.WIN_RATE_TOLERANCE:
                    self.result.add_alert(
                        'warning',
                        'performance',
                        f'Win rate deviation from backtest: {deviation:.1%}',
                        {
                            'expected': expected,
                            'actual': actual,
                            'deviation': actual - expected,
                            'tolerance': self.WIN_RATE_TOLERANCE,
                        }
                    )
    
    async def _daily_report_loop(self) -> None:
        """Generate daily progress reports."""
        logger.info("Starting daily report loop")
        
        try:
            while self._is_running:
                try:
                    await asyncio.sleep(86400)  # Daily
                    self._generate_progress_report()
                    
                except Exception as e:
                    logger.error("Error generating daily report", error=str(e))
                    await asyncio.sleep(3600)
                    
        except asyncio.CancelledError:
            logger.info("Daily report loop cancelled")
    
    def _generate_progress_report(self) -> None:
        """Generate progress report."""
        report_path = self.output_dir / f"progress_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'progress': self.result.to_dict(),
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(
            "Progress report generated",
            path=str(report_path),
            trades=self.result.total_trades,
            win_rate=round(self.result.win_rate, 4),
        )
    
    def _generate_final_report(self) -> None:
        """Generate final validation report."""
        report_path = self.output_dir / f"validation_report_{self.result.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Determine overall status
        critical_alerts = len([a for a in self.result.alerts if a.severity == 'critical'])
        
        status = 'PASSED'
        if critical_alerts > 0:
            status = 'FAILED'
        elif self.result.total_trades < self.MIN_TRADES and self.result.duration_hours < (self.MIN_DAYS * 24):
            status = 'INCOMPLETE'
        elif self.backtest_results.get('win_rate'):
            if abs(self.result.win_rate_deviation or 0) > self.WIN_RATE_TOLERANCE:
                status = 'WARNING'
        
        # Check latency requirement
        avg_latencies = [m.avg_latency_ms for m in self.result.api_metrics.values() if m.call_count > 0]
        overall_avg_latency = statistics.mean(avg_latencies) if avg_latencies else 0
        latency_ok = overall_avg_latency < self.MAX_LATENCY_MS
        
        report_data = {
            'validation_summary': {
                'status': status,
                'start_time': self.result.start_time.isoformat(),
                'end_time': self.result.end_time.isoformat() if self.result.end_time else None,
                'duration_hours': round(self.result.duration_hours, 2),
                'stopped_reason': self.result.stopped_reason,
            },
            'trade_statistics': {
                'total_trades': self.result.total_trades,
                'winning_trades': self.result.winning_trades,
                'losing_trades': self.result.losing_trades,
                'win_rate': round(self.result.win_rate, 4),
                'total_pnl': round(self.result.total_pnl, 2),
                'avg_trade_pnl': round(self.result.avg_trade_pnl, 4),
            },
            'performance_metrics': {
                'sharpe_ratio': round(self.result.sharpe_ratio, 4),
                'max_drawdown_pct': round(self.result.max_drawdown_pct, 4),
            },
            'api_performance': {
                'overall_avg_latency_ms': round(overall_avg_latency, 2),
                'latency_requirement_met': latency_ok,
                'max_allowed_latency_ms': self.MAX_LATENCY_MS,
                'endpoints': {k: v.to_dict() for k, v in self.result.api_metrics.items()},
            },
            'backtest_comparison': {
                'backtest_win_rate': self.backtest_results.get('win_rate'),
                'actual_win_rate': self.result.win_rate,
                'deviation': self.result.win_rate_deviation,
                'tolerance': self.WIN_RATE_TOLERANCE,
                'within_tolerance': abs(self.result.win_rate_deviation or 0) <= self.WIN_RATE_TOLERANCE if self.result.win_rate_deviation is not None else None,
            } if self.backtest_results else None,
            'alerts': [a.to_dict() for a in self.result.alerts],
            'acceptance_criteria': {
                'min_trades': self.MIN_TRADES,
                'min_days': self.MIN_DAYS,
                'max_latency_ms': self.MAX_LATENCY_MS,
                'win_rate_tolerance': self.WIN_RATE_TOLERANCE,
                'min_win_rate': self.MIN_WIN_RATE,
                'criteria_met': {
                    'trade_count': self.result.total_trades >= self.MIN_TRADES,
                    'duration': self.result.duration_hours >= (self.MIN_DAYS * 24),
                    'latency': latency_ok,
                    'no_critical_errors': critical_alerts == 0,
                }
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Also generate a text summary
        summary_path = self.output_dir / f"validation_summary_{self.result.start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        self._write_text_summary(summary_path, report_data)
        
        logger.info(
            "Final validation report generated",
            path=str(report_path),
            status=status,
            trades=self.result.total_trades,
        )
    
    def _write_text_summary(self, path: Path, data: Dict) -> None:
        """Write human-readable text summary."""
        lines = [
            "=" * 70,
            "           KALSHI TRADING BOT - VALIDATION REPORT",
            "=" * 70,
            "",
            f"Status: {data['validation_summary']['status']}",
            f"Start: {data['validation_summary']['start_time']}",
            f"End: {data['validation_summary']['end_time']}",
            f"Duration: {data['validation_summary']['duration_hours']:.1f} hours",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades: {data['trade_statistics']['total_trades']}",
            f"Win Rate: {data['trade_statistics']['win_rate']:.1%}",
            f"Total P&L: ${data['trade_statistics']['total_pnl']:.2f}",
            "",
            "API PERFORMANCE",
            "-" * 40,
            f"Average Latency: {data['api_performance']['overall_avg_latency_ms']:.1f}ms",
            f"Latency Requirement: {'✓ MET' if data['api_performance']['latency_requirement_met'] else '✗ FAILED'}",
            "",
            "ACCEPTANCE CRITERIA",
            "-" * 40,
        ]
        
        criteria = data['acceptance_criteria']['criteria_met']
        for name, met in criteria.items():
            lines.append(f"  {name}: {'✓ PASSED' if met else '✗ FAILED'}")
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))


def load_backtest_results(path: str) -> Dict[str, Any]:
    """Load backtest results from JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load backtest results", path=path, error=str(e))
        return {}


async def main():
    """Main entry point for validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Kalshi trading bot in demo environment')
    parser.add_argument('--backtest-results', type=str, help='Path to backtest results JSON')
    parser.add_argument('--output-dir', type=str, default='validation_output', help='Output directory')
    parser.add_argument('--min-trades', type=int, default=100, help='Minimum trades to validate')
    parser.add_argument('--min-days', type=int, default=7, help='Minimum days to validate')
    parser.add_argument('--api-key', type=str, help='Kalshi API key ID')
    parser.add_argument('--private-key', type=str, help='Path to private key file')
    
    args = parser.parse_args()
    
    # Load backtest results if provided
    backtest_results = {}
    if args.backtest_results:
        backtest_results = load_backtest_results(args.backtest_results)
    
    # Create and run validator
    validator = DemoValidator(
        api_key_id=args.api_key,
        private_key_path=args.private_key,
        backtest_results=backtest_results,
        output_dir=args.output_dir
    )
    
    # Override thresholds if specified
    if args.min_trades:
        validator.MIN_TRADES = args.min_trades
    if args.min_days:
        validator.MIN_DAYS = args.min_days
    
    try:
        await validator.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await validator.stop('interrupted')
    except Exception as e:
        logger.exception("Validation failed")
        await validator.stop(f'error: {str(e)}')
        sys.exit(1)
    
    # Exit with appropriate code
    critical_alerts = len([a for a in validator.result.alerts if a.severity == 'critical'])
    if critical_alerts > 0:
        logger.error("Validation failed with critical errors")
        sys.exit(1)
    else:
        logger.info("Validation completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    asyncio.run(main())
