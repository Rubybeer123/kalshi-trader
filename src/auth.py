"""Kalshi RSA-PSS authentication implementation."""

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend


@dataclass(frozen=True)
class AuthHeaders:
    """Authentication headers for Kalshi API requests."""
    kalshi_access_key: str
    kalshi_access_signature: str
    kalshi_access_timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for HTTP headers."""
        return {
            "KALSHI-ACCESS-KEY": self.kalshi_access_key,
            "KALSHI-ACCESS-SIGNATURE": self.kalshi_access_signature,
            "KALSHI-ACCESS-TIMESTAMP": self.kalshi_access_timestamp,
        }


class KalshiAuthenticator:
    """Handles RSA-PSS authentication for Kalshi API."""
    
    def __init__(self, key_id: str, private_key_path: str):
        """
        Initialize authenticator with API credentials.
        
        Args:
            key_id: Kalshi API key ID
            private_key_path: Path to RSA private key PEM file
        """
        self.key_id = key_id
        self.private_key = self._load_private_key(private_key_path)
    
    @staticmethod
    def _load_private_key(path: str) -> rsa.RSAPrivateKey:
        """
        Load RSA private key from PEM file.
        
        Args:
            path: Path to private key file
            
        Returns:
            RSAPrivateKey object
            
        Raises:
            FileNotFoundError: If key file doesn't exist
            ValueError: If key is invalid
        """
        key_path = Path(path)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file not found: {path}")
        
        pem_data = key_path.read_bytes()
        
        try:
            private_key = serialization.load_pem_private_key(
                pem_data,
                password=None,
                backend=default_backend()
            )
        except Exception as e:
            raise ValueError(f"Failed to load private key: {e}")
        
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("Private key is not an RSA key")
        
        # Verify key size (Kalshi requires 2048-bit)
        key_size = private_key.key_size
        if key_size < 2048:
            raise ValueError(f"RSA key must be at least 2048 bits, got {key_size}")
        
        return private_key
    
    def sign_pss_text(self, message: str) -> str:
        """
        Sign a message using RSA-PSS with MGF1-SHA256.
        
        Args:
            message: The message to sign
            
        Returns:
            Base64-encoded signature
        """
        message_bytes = message.encode("utf-8")
        
        signature = self.private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH  # Use maximum salt length
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode("utf-8")
    
    def generate_headers(self, method: str, path: str) -> dict:
        """
        Generate authentication headers for an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path without query parameters (e.g., /trade-api/v2/portfolio/positions)
            
        Returns:
            Dictionary with authentication headers
        """
        # Get current timestamp in milliseconds
        timestamp = str(int(time.time() * 1000))
        
        # Create message: timestamp + method + path
        # Note: path should not include query parameters
        message = f"{timestamp}{method.upper()}{path}"
        
        # Sign the message
        signature = self.sign_pss_text(message)
        
        # Create headers
        headers = AuthHeaders(
            kalshi_access_key=self.key_id,
            kalshi_access_signature=signature,
            kalshi_access_timestamp=timestamp
        )
        
        return headers.to_dict()
    
    def generate_headers_for_url(self, method: str, url: str) -> dict:
        """
        Generate authentication headers from a full URL.
        
        Args:
            method: HTTP method
            url: Full URL (e.g., https://api.kalshi.com/trade-api/v2/portfolio/positions?limit=10)
            
        Returns:
            Dictionary with authentication headers
        """
        # Extract path without query string
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
        
        return self.generate_headers(method, path)


def generate_test_key_pair() -> tuple[str, rsa.RSAPrivateKey]:
    """
    Generate a test RSA key pair for unit testing.
    
    Returns:
        Tuple of (PEM string, private key object)
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode("utf-8")
    
    return pem, private_key
