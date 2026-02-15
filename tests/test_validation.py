"""Tests for the validation module."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.validation import (
    APIMetrics,
    DemoValidator,
    ValidationAlert,
    ValidationResult,
    load_backtest_results,
)


class TestAPIMetrics:
    """Tests for APIMetrics class."""
    
    def test_initialization(self):
        """Test APIMetrics initialization."""
        metrics = APIMetrics(endpoint="/trade")
        
        assert metrics.endpoint == "/trade"
        assert metrics.call_count == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.min_latency_ms == float('inf')
        assert metrics.max_latency_ms == 0.0
        assert metrics.error_count == 0
    
    def test_record_call(self):
        """Test recording API calls."""
        metrics = APIMetrics(endpoint="/trade")
        
        metrics.record_call(100.0, error=False)
        assert metrics.call_count == 1
        assert metrics.avg_latency_ms == 100.0
        assert metrics.min_latency_ms == 100.0
        assert metrics.max_latency_ms == 100.0
        assert metrics.error_count == 0
        
        # Record another call
        metrics.record_call(150.0, error=False)
        assert metrics.call_count == 2
        assert metrics.avg_latency_ms == 125.0
        assert metrics.min_latency_ms == 100.0
        assert metrics.max_latency_ms == 150.0
    
    def test_record_error(self):
        """Test recording errors."""
        metrics = APIMetrics(endpoint="/trade")
        
        metrics.record_call(100.0, error=True)
        assert metrics.call_count == 1
        assert metrics.error_count == 1
        assert metrics.error_rate == 1.0
        
        metrics.record_call(100.0, error=False)
        assert metrics.error_count == 1
        assert metrics.error_rate == 0.5
    
    def test_avg_latency_empty(self):
        """Test average latency when no calls made."""
        metrics = APIMetrics(endpoint="/trade")
        assert metrics.avg_latency_ms == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = APIMetrics(endpoint="/trade")
        metrics.record_call(100.0, error=False)
        metrics.record_call(200.0, error=True)
        
        data = metrics.to_dict()
        
        assert data['endpoint'] == "/trade"
        assert data['call_count'] == 2
        assert data['avg_latency_ms'] == 150.0
        assert data['min_latency_ms'] == 100.0
        assert data['max_latency_ms'] == 200.0
        assert data['error_count'] == 1
        assert data['error_rate'] == 0.5


class TestValidationAlert:
    """Tests for ValidationAlert class."""
    
    def test_initialization(self):
        """Test alert initialization."""
        alert = ValidationAlert(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            severity="warning",
            category="api",
            message="High latency detected",
            details={"latency": 300}
        )
        
        assert alert.severity == "warning"
        assert alert.category == "api"
        assert alert.message == "High latency detected"
        assert alert.details == {"latency": 300}
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        alert = ValidationAlert(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            severity="critical",
            category="trading",
            message="Test alert",
            details={"key": "value"}
        )
        
        data = alert.to_dict()
        
        assert data['timestamp'] == "2024-01-01T12:00:00+00:00"
        assert data['severity'] == "critical"
        assert data['category'] == "trading"
        assert data['message'] == "Test alert"
        assert data['details'] == {"key": "value"}


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_initialization(self):
        """Test result initialization."""
        start = datetime.now(timezone.utc)
        result = ValidationResult(start_time=start)
        
        assert result.start_time == start
        assert result.end_time is None
        assert result.total_trades == 0
        assert result.alerts == []
        assert result.is_running is True
    
    def test_duration_hours(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 2, 30, tzinfo=timezone.utc)
        
        result = ValidationResult(start_time=start, end_time=end)
        assert result.duration_hours == 2.5
    
    def test_duration_running(self):
        """Test duration when still running."""
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        result = ValidationResult(start_time=start)
        
        # Should be approximately 1 hour
        assert 0.9 < result.duration_hours < 1.1
    
    def test_add_alert(self):
        """Test adding alerts."""
        result = ValidationResult(start_time=datetime.now(timezone.utc))
        
        result.add_alert("warning", "api", "Test warning", {"detail": 123})
        
        assert len(result.alerts) == 1
        assert result.alerts[0].severity == "warning"
        assert result.alerts[0].category == "api"
        assert result.alerts[0].message == "Test warning"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = ValidationResult(start_time=start)
        result.total_trades = 50
        result.win_rate = 0.55
        
        # Add some API metrics
        metrics = APIMetrics("/trade")
        metrics.record_call(100.0)
        result.api_metrics["/trade"] = metrics
        
        data = result.to_dict()
        
        assert data['start_time'] == "2024-01-01T00:00:00+00:00"
        assert data['total_trades'] == 50
        assert data['win_rate'] == 0.55
        assert '/trade' in data['api_metrics']


class TestDemoValidator:
    """Tests for DemoValidator class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def temp_key_file(self):
        """Create temporary private key file."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            path = f.name
        
        yield path
        
        try:
            os.unlink(path)
        except:
            pass
    
    @pytest.fixture
    def validator(self, temp_output_dir, temp_key_file):
        """Create validator instance."""
        return DemoValidator(
            api_key_id="test_key",
            private_key_path=temp_key_file,
            output_dir=temp_output_dir
        )
    
    def test_initialization(self, validator, temp_output_dir):
        """Test validator initialization."""
        assert validator.output_dir == Path(temp_output_dir)
        assert validator.config.kalshi_api_key_id == "test_key"
        assert validator.config.kalshi_env.value == "demo"
        assert validator.MIN_TRADES == 100
        assert validator.MIN_DAYS == 7
        assert validator.MAX_LATENCY_MS == 200
    
    def test_with_backtest_results(self, temp_output_dir):
        """Test validator with backtest results."""
        backtest = {"win_rate": 0.6, "expectancy": 0.05}
        validator = DemoValidator(
            backtest_results=backtest,
            output_dir=temp_output_dir
        )
        
        assert validator.backtest_results == backtest
    
    @pytest.mark.asyncio
    async def test_stop(self, validator):
        """Test stopping validator."""
        validator._is_running = True
        validator.result.is_running = True
        
        # Mock the event
        validator._stop_event = asyncio.Event()
        
        await validator.stop("test_reason")
        
        assert validator._is_running is False
        assert validator.result.is_running is False
        assert validator.result.stopped_reason == "test_reason"
        assert validator.result.end_time is not None
    
    @pytest.mark.asyncio
    async def test_api_metrics_tracking(self, validator):
        """Test API metrics are tracked correctly."""
        # Add some metrics
        metrics = APIMetrics("/trade")
        metrics.record_call(100.0)
        metrics.record_call(150.0, error=True)
        validator.result.api_metrics["/trade"] = metrics
        
        assert validator.result.api_metrics["/trade"].call_count == 2
        assert validator.result.api_metrics["/trade"].error_count == 1
    
    def test_generate_progress_report(self, validator):
        """Test generating progress report."""
        validator.result.total_trades = 50
        validator.result.win_rate = 0.55
        
        validator._generate_progress_report()
        
        # Check that report was created
        reports = list(validator.output_dir.glob("progress_*.json"))
        assert len(reports) == 1
        
        # Verify content
        with open(reports[0]) as f:
            data = json.load(f)
            assert data['progress']['total_trades'] == 50
            assert data['progress']['win_rate'] == 0.55
    
    def test_generate_final_report(self, validator):
        """Test generating final validation report."""
        validator.result.total_trades = 100
        validator.result.win_rate = 0.55
        validator.result.end_time = datetime.now(timezone.utc)
        validator.result.stopped_reason = "test_complete"
        validator._is_running = False
        validator.result.is_running = False
        
        # Add API metrics
        metrics = APIMetrics("/trade")
        metrics.record_call(100.0)
        validator.result.api_metrics["/trade"] = metrics
        
        validator._generate_final_report()
        
        # Check reports were created
        json_reports = list(validator.output_dir.glob("validation_report_*.json"))
        txt_reports = list(validator.output_dir.glob("validation_summary_*.txt"))
        
        assert len(json_reports) == 1
        assert len(txt_reports) == 1
        
        # Verify JSON content
        with open(json_reports[0]) as f:
            data = json.load(f)
            assert 'validation_summary' in data
            assert 'trade_statistics' in data
            assert 'api_performance' in data
            assert data['trade_statistics']['total_trades'] == 100


class TestLoadBacktestResults:
    """Tests for load_backtest_results function."""
    
    def test_load_valid_file(self):
        """Test loading valid backtest results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"win_rate": 0.6, "sharpe": 1.5}, f)
            path = f.name
        
        try:
            results = load_backtest_results(path)
            assert results['win_rate'] == 0.6
            assert results['sharpe'] == 1.5
        finally:
            os.unlink(path)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file returns empty dict."""
        results = load_backtest_results("/nonexistent/path/results.json")
        assert results == {}
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json")
            path = f.name
        
        try:
            results = load_backtest_results(path)
            assert results == {}
        finally:
            os.unlink(path)


class TestValidationThresholds:
    """Tests for validation threshold checks."""
    
    @pytest.fixture
    def temp_key_file(self):
        """Create temporary private key file."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            path = f.name
        
        yield path
        
        try:
            os.unlink(path)
        except:
            pass
    
    @pytest.fixture
    def validator(self, temp_key_file):
        """Create validator with test output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            v = DemoValidator(
                api_key_id="test_key",
                private_key_path=temp_key_file,
                output_dir=tmpdir
            )
            yield v
    
    def test_latency_threshold(self, validator):
        """Test latency threshold checking."""
        metrics = APIMetrics("/trade")
        metrics.record_call(250.0)  # Above threshold
        validator.result.api_metrics["/trade"] = metrics
        
        # Should generate alert if we checked
        assert metrics.avg_latency_ms > validator.MAX_LATENCY_MS
    
    def test_error_rate_threshold(self, validator):
        """Test error rate threshold checking."""
        metrics = APIMetrics("/trade")
        # 2 errors out of 100 calls = 2%
        for i in range(98):
            metrics.record_call(100.0, error=False)
        for i in range(2):
            metrics.record_call(100.0, error=True)
        
        assert metrics.error_rate == 0.02
        assert metrics.error_rate > validator.MAX_ERROR_RATE
    
    def test_win_rate_threshold(self, validator):
        """Test win rate threshold checking."""
        validator.result.total_trades = 100
        validator.result.win_rate = 0.45  # Below 50%
        
        assert validator.result.win_rate < validator.MIN_WIN_RATE
    
    def test_backtest_comparison(self, validator):
        """Test backtest comparison."""
        validator.backtest_results = {"win_rate": 0.60}
        validator.result.total_trades = 100
        validator.result.win_rate = 0.54  # 6% deviation, above 5% tolerance
        
        deviation = abs(validator.result.win_rate - validator.backtest_results['win_rate'])
        assert deviation > validator.WIN_RATE_TOLERANCE


class TestIntegration:
    """Integration tests for validation module."""
    
    @pytest.fixture
    def temp_key_file(self):
        """Create temporary private key file."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            path = f.name
        
        yield path
        
        try:
            os.unlink(path)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_full_validation_lifecycle(self, temp_key_file):
        """Test complete validation lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DemoValidator(
                api_key_id="test_key",
                private_key_path=temp_key_file,
                output_dir=tmpdir
            )
            
            # Override thresholds for quick test
            validator.MIN_TRADES = 1
            
            # Add some test data
            validator.result.total_trades = 5
            validator.result.win_rate = 0.6
            
            # Add API metrics
            metrics = APIMetrics("/trade")
            metrics.record_call(50.0)
            validator.result.api_metrics["/trade"] = metrics
            
            # Generate reports
            validator._generate_progress_report()
            
            validator.result.end_time = datetime.now(timezone.utc)
            validator.result.stopped_reason = "test_complete"
            validator._is_running = False
            validator.result.is_running = False
            
            validator._generate_final_report()
            
            # Verify reports exist
            assert len(list(Path(tmpdir).glob("progress_*.json"))) == 1
            assert len(list(Path(tmpdir).glob("validation_report_*.json"))) == 1
            assert len(list(Path(tmpdir).glob("validation_summary_*.txt"))) == 1
    
    def test_multiple_alerts(self):
        """Test handling multiple alerts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DemoValidator(output_dir=tmpdir)
            
            # Add multiple alerts
            validator.result.add_alert("info", "system", "Info message")
            validator.result.add_alert("warning", "api", "Warning message")
            validator.result.add_alert("critical", "trading", "Critical message")
            
            assert len(validator.result.alerts) == 3
            assert len([a for a in validator.result.alerts if a.severity == "critical"]) == 1
            assert len([a for a in validator.result.alerts if a.severity == "warning"]) == 1
            assert len([a for a in validator.result.alerts if a.severity == "info"]) == 1
    
    def test_text_summary_generation(self):
        """Test text summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DemoValidator(output_dir=tmpdir)
            
            validator.result.total_trades = 100
            validator.result.win_rate = 0.55
            validator.result.start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
            validator.result.end_time = datetime(2024, 1, 8, tzinfo=timezone.utc)
            
            # Add API metrics
            metrics = APIMetrics("/trade")
            metrics.record_call(100.0)
            validator.result.api_metrics["/trade"] = metrics
            
            report_data = {
                'validation_summary': {
                    'status': 'PASSED',
                    'start_time': validator.result.start_time.isoformat(),
                    'end_time': validator.result.end_time.isoformat(),
                    'duration_hours': 168.0,
                    'stopped_reason': 'test',
                },
                'trade_statistics': {
                    'total_trades': 100,
                    'win_rate': 0.55,
                },
                'api_performance': {
                    'overall_avg_latency_ms': 100.0,
                    'latency_requirement_met': True,
                },
                'acceptance_criteria': {
                    'criteria_met': {
                        'trade_count': True,
                        'duration': True,
                        'latency': True,
                        'no_critical_errors': True,
                    }
                }
            }
            
            summary_path = Path(tmpdir) / "summary.txt"
            validator._write_text_summary(summary_path, report_data)
            
            assert summary_path.exists()
            
            content = summary_path.read_text()
            assert "PASSED" in content
            assert "100" in content
            assert "✓" in content or "PASSED" in content
