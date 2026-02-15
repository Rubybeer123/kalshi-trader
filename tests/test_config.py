"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Config, Environment, get_config, reset_config, create_config


class TestConfigLoading:
    """Test configuration loading from environment variables."""
    
    @pytest.fixture(autouse=True)
    def reset_global_config(self):
        """Reset global config before each test."""
        reset_config()
        yield
        reset_config()
    
    @pytest.fixture
    def temp_private_key(self):
        """Create a temporary private key file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\n")
            f.write("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n")
            f.write("-----END PRIVATE KEY-----\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_config_loads_from_env_vars(self, temp_private_key, monkeypatch):
        """Test that config loads correctly from environment variables."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key-id-12345")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("KALSHI_ENV", "demo")
        
        config = Config()
        
        assert config.kalshi_api_key_id == "test-key-id-12345"
        assert config.kalshi_private_key_path == temp_private_key
        assert config.kalshi_env == Environment.DEMO
        assert config.is_demo is True
        assert config.is_production is False
    
    def test_config_defaults_to_demo(self, temp_private_key, monkeypatch):
        """Test that config defaults to demo environment."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        # Not setting KALSHI_ENV
        
        config = Config()
        assert config.kalshi_env == Environment.DEMO
    
    def test_production_environment(self, temp_private_key, monkeypatch):
        """Test production environment configuration."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "prod-key-id")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("KALSHI_ENV", "production")
        
        config = Config()
        
        assert config.kalshi_env == Environment.PRODUCTION
        assert config.is_demo is False
        assert config.is_production is True


class TestConfigValidation:
    """Test configuration validation."""
    
    @pytest.fixture
    def temp_private_key(self):
        """Create a temporary private key file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\n")
            f.write("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n")
            f.write("-----END PRIVATE KEY-----\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_missing_api_key_id_raises_error(self, temp_private_key, monkeypatch):
        """Test that missing API key ID raises validation error."""
        monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        
        with pytest.raises(ValidationError) as exc_info:
            Config()
        
        assert "kalshi_api_key_id" in str(exc_info.value)
    
    def test_missing_private_key_path_raises_error(self, monkeypatch):
        """Test that missing private key path raises validation error."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
        
        with pytest.raises(ValidationError) as exc_info:
            Config()
        
        assert "kalshi_private_key_path" in str(exc_info.value)
    
    def test_invalid_private_key_path_raises_error(self, monkeypatch):
        """Test that non-existent private key file raises error."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", "/nonexistent/path/key.pem")
        
        with pytest.raises(ValidationError) as exc_info:
            Config()
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_directory_as_private_key_path_raises_error(self, temp_private_key, monkeypatch):
        """Test that directory path raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
            monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", tmpdir)
            
            with pytest.raises(ValidationError) as exc_info:
                Config()
            
            assert "not a file" in str(exc_info.value).lower()
    
    def test_invalid_pem_format_raises_error(self, monkeypatch):
        """Test that invalid PEM format raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("This is not a valid PEM file\n")
            temp_path = f.name
        
        try:
            monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
            monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_path)
            
            with pytest.raises(ValidationError) as exc_info:
                Config()
            
            assert "valid pem" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_path)


class TestUrlSelection:
    """Test API URL selection based on environment."""
    
    @pytest.fixture
    def temp_private_key(self):
        """Create a temporary private key file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\n")
            f.write("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n")
            f.write("-----END PRIVATE KEY-----\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_demo_url_selected_in_demo_mode(self, temp_private_key, monkeypatch):
        """Test that demo URL is used in demo mode."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("KALSHI_ENV", "demo")
        
        config = Config()
        
        assert config.api_base_url == config.kalshi_demo_url
        assert "demo-api" in config.api_base_url
    
    def test_prod_url_selected_in_production_mode(self, temp_private_key, monkeypatch):
        """Test that production URL is used in production mode."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("KALSHI_ENV", "production")
        
        config = Config()
        
        assert config.api_base_url == config.kalshi_prod_url
        assert "api.elections" in config.api_base_url
    
    def test_custom_urls_can_be_set(self, temp_private_key, monkeypatch):
        """Test that custom URLs can be configured."""
        custom_demo = "https://custom-demo.example.com"
        custom_prod = "https://custom-prod.example.com"
        
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("KALSHI_ENV", "demo")
        monkeypatch.setenv("KALSHI_DEMO_URL", custom_demo)
        monkeypatch.setenv("KALSHI_PROD_URL", custom_prod)
        
        config = Config()
        
        assert config.kalshi_demo_url == custom_demo
        assert config.kalshi_prod_url == custom_prod
        assert config.api_base_url == custom_demo


class TestGlobalConfig:
    """Test global configuration instance."""
    
    @pytest.fixture(autouse=True)
    def reset_global_config(self):
        """Reset global config before each test."""
        reset_config()
        yield
        reset_config()
    
    @pytest.fixture
    def temp_private_key(self):
        """Create a temporary private key file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\n")
            f.write("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n")
            f.write("-----END PRIVATE KEY-----\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_get_config_returns_same_instance(self, temp_private_key, monkeypatch):
        """Test that get_config returns the same instance."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_create_config_with_explicit_values(self, temp_private_key):
        """Test creating config with explicit values."""
        config = create_config(
            kalshi_api_key_id="explicit-key",
            kalshi_private_key_path=temp_private_key,
            kalshi_env=Environment.PRODUCTION
        )
        
        assert config.kalshi_api_key_id == "explicit-key"
        assert config.is_production is True


class TestLoggingConfig:
    """Test logging configuration."""
    
    @pytest.fixture
    def temp_private_key(self):
        """Create a temporary private key file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\n")
            f.write("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n")
            f.write("-----END PRIVATE KEY-----\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_valid_log_levels(self, temp_private_key, monkeypatch):
        """Test that valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
            monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
            monkeypatch.setenv("LOG_LEVEL", level)
            
            config = Config()
            assert config.logging.level == level
    
    def test_invalid_log_level_raises_error(self, temp_private_key, monkeypatch):
        """Test that invalid log level raises error."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        
        with pytest.raises(ValidationError) as exc_info:
            Config()
        
        assert "Invalid log level" in str(exc_info.value)


class TestTradingConfig:
    """Test trading configuration."""
    
    @pytest.fixture
    def temp_private_key(self):
        """Create a temporary private key file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\n")
            f.write("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n")
            f.write("-----END PRIVATE KEY-----\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_trading_defaults(self, temp_private_key, monkeypatch):
        """Test trading configuration defaults."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        
        config = Config()
        
        assert config.trading.initial_capital == 50.0
        assert config.trading.max_positions == 2
        assert config.trading.max_exposure == 50.0
        assert config.trading.daily_loss_limit == 5.0
        assert config.trading.consecutive_loss_limit == 3
    
    def test_trading_custom_values(self, temp_private_key, monkeypatch):
        """Test custom trading configuration values."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("INITIAL_CAPITAL", "100")
        monkeypatch.setenv("MAX_POSITIONS", "5")
        monkeypatch.setenv("ARBITRAGE_MIN_EDGE", "0.05")
        
        config = Config()
        
        assert config.trading.initial_capital == 100.0
        assert config.trading.max_positions == 5
        assert config.trading.arbitrage_min_edge == 0.05
    
    def test_invalid_max_positions_raises_error(self, temp_private_key, monkeypatch):
        """Test that invalid max positions raises error."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("MAX_POSITIONS", "0")
        
        with pytest.raises(ValidationError):
            Config()
    
    def test_negative_capital_raises_error(self, temp_private_key, monkeypatch):
        """Test that negative capital raises error."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        monkeypatch.setenv("INITIAL_CAPITAL", "-50")
        
        with pytest.raises(ValidationError):
            Config()


class TestConfigStringRepresentation:
    """Test config string representation."""
    
    @pytest.fixture
    def temp_private_key(self):
        """Create a temporary private key file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\n")
            f.write("MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...\n")
            f.write("-----END PRIVATE KEY-----\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_str_hides_sensitive_data(self, temp_private_key, monkeypatch):
        """Test that string representation hides sensitive data."""
        monkeypatch.setenv("KALSHI_API_KEY_ID", "super-secret-key-id")
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", temp_private_key)
        
        config = Config()
        str_repr = str(config)
        
        assert "super-secret" not in str_repr
        assert "super-sec..." in str_repr or "..." in str_repr
        assert "demo" in str_repr
