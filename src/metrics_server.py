"""Health check and metrics endpoint server for Kalshi Trader."""

import asyncio
import json
import time
from typing import Any, Dict, Optional

import structlog
from aiohttp import web
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

logger = structlog.get_logger(__name__)


class MetricsServer:
    """
    HTTP server for health checks and Prometheus metrics.
    
    Endpoints:
    - GET /health - Health check (liveness)
    - GET /ready - Readiness probe
    - GET /metrics - Prometheus metrics
    """
    
    def __init__(self, port: int = 8080, trading_bot=None):
        """
        Initialize metrics server.
        
        Args:
            port: HTTP server port
            trading_bot: Reference to trading bot for status
        """
        self.port = port
        self.trading_bot = trading_bot
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._start_time = time.time()
        
        # Setup routes
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_get("/ready", self.ready_handler)
        self.app.router.add_get("/metrics", self.metrics_handler)
        
        # Initialize Prometheus metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        # Trade metrics
        self.trades_total = Counter(
            "trades_total",
            "Total number of trades executed",
            ["market", "side", "status"]
        )
        
        # Latency metrics
        self.latency_ms = Histogram(
            "latency_ms",
            "API call latency in milliseconds",
            ["endpoint", "method"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        )
        
        # Error metrics
        self.errors_total = Counter(
            "errors_total",
            "Total number of errors",
            ["error_type", "market"]
        )
        
        # Business metrics
        self.pnl_total = Gauge(
            "pnl_total",
            "Total profit/loss in USD"
        )
        
        self.win_rate = Gauge(
            "win_rate",
            "Win rate as percentage (0-1)"
        )
        
        self.positions_open = Gauge(
            "positions_open",
            "Number of open positions"
        )
        
        self.signals_generated = Counter(
            "signals_generated_total",
            "Total number of signals generated",
            ["market"]
        )
        
        self.circuit_breaker_active = Gauge(
            "circuit_breaker_active",
            "Whether circuit breaker is active (1) or not (0)"
        )
        
        # System metrics
        self.uptime_seconds = Gauge(
            "uptime_seconds",
            "Application uptime in seconds"
        )
        
        self.trades_per_day = Gauge(
            "trades_per_day",
            "Number of trades in the last 24 hours"
        )
    
    async def start(self):
        """Start the metrics server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()
        
        logger.info("Metrics server started", port=self.port)
    
    async def stop(self):
        """Stop the metrics server."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Metrics server stopped")
    
    async def health_handler(self, request: web.Request) -> web.Response:
        """
        Liveness probe - indicates if the process is running.
        
        Returns:
            200 if process is alive
            503 if process is shutting down
        """
        # Check if trading bot is running
        if self.trading_bot and hasattr(self.trading_bot, '_running'):
            if not self.trading_bot._running:
                return web.json_response(
                    {"status": "unhealthy", "reason": "trading bot stopped"},
                    status=503
                )
        
        # Update uptime metric
        self.uptime_seconds.set(time.time() - self._start_time)
        
        return web.json_response({
            "status": "healthy",
            "uptime_seconds": time.time() - self._start_time,
        })
    
    async def ready_handler(self, request: web.Request) -> web.Response:
        """
        Readiness probe - indicates if the service is ready to accept traffic.
        
        Returns:
            200 if service is ready
            503 if service is still initializing
        """
        if not self.trading_bot:
            return web.json_response(
                {"status": "not_ready", "reason": "trading bot not connected"},
                status=503
            )
        
        # Check if bot has initialized components
        ready_checks = {
            "running": getattr(self.trading_bot, '_running', False),
            "has_exchange": getattr(self.trading_bot, 'exchange', None) is not None,
            "has_risk_manager": getattr(self.trading_bot, 'risk_manager', None) is not None,
        }
        
        if not all(ready_checks.values()):
            return web.json_response(
                {"status": "not_ready", "checks": ready_checks},
                status=503
            )
        
        return web.json_response({
            "status": "ready",
            "checks": ready_checks,
        })
    
    async def metrics_handler(self, request: web.Request) -> web.Response:
        """
        Prometheus metrics endpoint.
        
        Returns:
            Prometheus-formatted metrics
        """
        # Update dynamic metrics
        self._update_metrics()
        
        # Generate Prometheus output
        body = generate_latest(REGISTRY)
        
        return web.Response(
            body=body,
            content_type=CONTENT_TYPE_LATEST
        )
    
    def _update_metrics(self):
        """Update dynamic metrics from trading bot state."""
        if not self.trading_bot:
            return
        
        # Update positions
        positions = 0
        if hasattr(self.trading_bot, 'execution_engine') and self.trading_bot.execution_engine:
            positions = len(self.trading_bot.execution_engine.get_all_positions())
        self.positions_open.set(positions)
        
        # Update circuit breaker status
        circuit_active = 0
        if hasattr(self.trading_bot, 'circuit_breaker') and self.trading_bot.circuit_breaker:
            circuit_active = 1 if self.trading_bot.circuit_breaker.is_triggered else 0
        self.circuit_breaker_active.set(circuit_active)
        
        # Update uptime
        self.uptime_seconds.set(time.time() - self._start_time)
        
        # Update trades per day (calculate from signals if available)
        if hasattr(self.trading_bot, '_signals_executed'):
            # This is a simple approximation - in production you'd track daily
            self.trades_per_day.set(self.trading_bot._signals_executed)
    
    def record_trade(self, market: str, side: str, status: str = "success"):
        """Record a trade execution."""
        self.trades_total.labels(market=market, side=side, status=status).inc()
    
    def record_latency(self, endpoint: str, method: str, duration_ms: float):
        """Record API call latency."""
        self.latency_ms.labels(endpoint=endpoint, method=method).observe(duration_ms)
    
    def record_error(self, error_type: str, market: str = ""):
        """Record an error."""
        self.errors_total.labels(error_type=error_type, market=market).inc()
    
    def record_signal(self, market: str):
        """Record a signal generation."""
        self.signals_generated.labels(market=market).inc()
    
    def update_pnl(self, pnl: float):
        """Update P&L metric."""
        self.pnl_total.set(pnl)
    
    def update_win_rate(self, win_rate: float):
        """Update win rate metric (0-1)."""
        self.win_rate.set(win_rate)


# Global metrics server instance
_metrics_server: Optional[MetricsServer] = None


def get_metrics_server() -> Optional[MetricsServer]:
    """Get the global metrics server instance."""
    return _metrics_server


def create_metrics_server(port: int = 8080, trading_bot=None) -> MetricsServer:
    """Create and return a new metrics server instance."""
    global _metrics_server
    _metrics_server = MetricsServer(port=port, trading_bot=trading_bot)
    return _metrics_server
