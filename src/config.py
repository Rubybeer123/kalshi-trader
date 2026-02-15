"""Configuration management using Pydantic Settings."""

import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Trading environment types."""
    DEMO = "demo"
    PRODUCTION = "production"


class TradingConfig(BaseSettings):
    """Trading-specific configuration."""
    initial_capital: float = Field(default=50.0, gt=0)
    max_positions: int = Field(default=2, ge=1)
    max_exposure: float = Field(default=50.0, gt=0)
    daily_loss_limit: float = Field(default=5.0, gt=0)
    consecutive_loss_limit: int = Field(default=3, ge=1)
    
    # Strategy-specific
    arbitrage_min_edge: float = Field(default=0.02, ge=0)


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: Literal["json", "text"] = Field(default="json")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class Config(BaseSettings):
    """Main application configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Credentials
    kalshi_api_key_id: str = Field(..., description="Kalshi API key ID")
    kalshi_private_key_path: str = Field(..., description="Path to RSA private key PEM file")
    
    # Environment
    kalshi_env: Environment = Field(default=Environment.DEMO)
    
    # URLs
    kalshi_demo_url: str = Field(default="https://demo-api.kalshi.co/trade-api/v2")
    kalshi_prod_url: str = Field(default="https://api.elections.kalshi.com/trade-api/v2")
    
    # Sub-configs
    trading: TradingConfig = Field(default_factory=TradingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @field_validator("kalshi_private_key_path")
    @classmethod
    def validate_private_key_path(cls, v: str) -> str:
        """Validate that private key file exists and is readable."""
        path = Path(v)
        if not path.is_absolute():
            # Try relative to project root
            path = Path(__file__).parent.parent.parent / v
        
        if not path.exists():
            raise ValueError(f"Private key file not found: {v}")
        
        if not path.is_file():
            raise ValueError(f"Private key path is not a file: {v}")
        
        if not os.access(path, os.R_OK):
            raise ValueError(f"Private key file is not readable: {v}")
        
        # Validate PEM format
        content = path.read_text()
        if "BEGIN PRIVATE KEY" not in content and "BEGIN RSA PRIVATE KEY" not in content:
            raise ValueError(f"Private key file does not appear to be a valid PEM: {v}")
        
        return str(path)
    
    @property
    def is_demo(self) -> bool:
        """Check if running in demo mode."""
        return self.kalshi_env == Environment.DEMO
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.kalshi_env == Environment.PRODUCTION
    
    @property
    def api_base_url(self) -> str:
        """Get the appropriate API base URL for the environment."""
        return self.kalshi_demo_url if self.is_demo else self.kalshi_prod_url
    
    @property
    def private_key_content(self) -> str:
        """Load and return the private key content."""
        path = Path(self.kalshi_private_key_path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / self.kalshi_private_key_path
        return path.read_text()
    
    def __str__(self) -> str:
        """String representation (hides sensitive data)."""
        return (
            f"Config(env={self.kalshi_env.value}, "
            f"key_id={self.kalshi_api_key_id[:8]}..., "
            f"demo={self.is_demo})"
        )


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


def create_config(**kwargs) -> Config:
    """Create a new config instance with explicit values (useful for testing)."""
    return Config(**kwargs)
