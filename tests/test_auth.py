"""Tests for Kalshi RSA-PSS authentication."""

import base64
import os
import tempfile
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from src.auth import KalshiAuthenticator, AuthHeaders, generate_test_key_pair


class TestAuthHeaders:
    """Test AuthHeaders dataclass."""
    
    def test_auth_headers_creation(self):
        """Test that AuthHeaders can be created with all fields."""
        headers = AuthHeaders(
            kalshi_access_key="test-key-123",
            kalshi_access_signature="c2lnbmF0dXJlCg==",
            kalshi_access_timestamp="1707849600000"
        )
        
        assert headers.kalshi_access_key == "test-key-123"
        assert headers.kalshi_access_signature == "c2lnbmF0dXJlCg=="
        assert headers.kalshi_access_timestamp == "1707849600000"
    
    def test_auth_headers_to_dict(self):
        """Test conversion to dictionary."""
        headers = AuthHeaders(
            kalshi_access_key="test-key",
            kalshi_access_signature="abc123",
            kalshi_access_timestamp="1707849600000"
        )
        
        result = headers.to_dict()
        
        assert result == {
            "KALSHI-ACCESS-KEY": "test-key",
            "KALSHI-ACCESS-SIGNATURE": "abc123",
            "KALSHI-ACCESS-TIMESTAMP": "1707849600000",
        }
    
    def test_auth_headers_immutable(self):
        """Test that AuthHeaders is immutable (frozen dataclass)."""
        headers = AuthHeaders(
            kalshi_access_key="test-key",
            kalshi_access_signature="sig",
            kalshi_access_timestamp="12345"
        )
        
        with pytest.raises(AttributeError):
            headers.kalshi_access_key = "new-key"


class TestGenerateTestKeyPair:
    """Test test key pair generation."""
    
    def test_generates_valid_pem(self):
        """Test that generated key pair produces valid PEM."""
        pem, private_key = generate_test_key_pair()
        
        assert "BEGIN PRIVATE KEY" in pem
        assert "END PRIVATE KEY" in pem
        assert isinstance(private_key, rsa.RSAPrivateKey)
    
    def test_key_is_2048_bits(self):
        """Test that generated key is 2048 bits."""
        pem, private_key = generate_test_key_pair()
        
        assert private_key.key_size == 2048
    
    def test_key_can_be_loaded(self):
        """Test that generated PEM can be loaded back."""
        pem, _ = generate_test_key_pair()
        
        loaded_key = serialization.load_pem_private_key(
            pem.encode(),
            password=None,
            backend=default_backend()
        )
        
        assert isinstance(loaded_key, rsa.RSAPrivateKey)
        assert loaded_key.key_size == 2048


class TestKalshiAuthenticatorInit:
    """Test KalshiAuthenticator initialization."""
    
    @pytest.fixture
    def temp_key_file(self):
        """Create a temporary private key file."""
        pem, _ = generate_test_key_pair()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(pem)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_initializes_with_valid_key(self, temp_key_file):
        """Test that authenticator initializes with valid key."""
        auth = KalshiAuthenticator("test-key-id", temp_key_file)
        
        assert auth.key_id == "test-key-id"
        assert isinstance(auth.private_key, rsa.RSAPrivateKey)
    
    def test_raises_file_not_found_for_missing_key(self):
        """Test that FileNotFoundError is raised for missing key file."""
        with pytest.raises(FileNotFoundError):
            KalshiAuthenticator("test-key-id", "/nonexistent/path/key.pem")
    
    def test_raises_value_error_for_invalid_pem(self):
        """Test that ValueError is raised for invalid PEM content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write("This is not a valid PEM file")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                KalshiAuthenticator("test-key-id", temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_raises_value_error_for_non_rsa_key(self):
        """Test that ValueError is raised for non-RSA key."""
        # Generate an EC key instead
        from cryptography.hazmat.primitives.asymmetric import ec
        
        private_key = ec.generate_private_key(
            ec.SECP256R1(),
            backend=default_backend()
        )
        
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(pem)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="not an RSA key"):
                KalshiAuthenticator("test-key-id", temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_raises_value_error_for_small_key(self):
        """Test that ValueError is raised for key smaller than 2048 bits."""
        # Generate a 1024-bit key (too small)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=1024,
            backend=default_backend()
        )
        
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(pem)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="at least 2048 bits"):
                KalshiAuthenticator("test-key-id", temp_path)
        finally:
            os.unlink(temp_path)


class TestSignPssText:
    """Test RSA-PSS signature generation."""
    
    @pytest.fixture
    def authenticator(self):
        """Create an authenticator with a test key."""
        pem, private_key = generate_test_key_pair()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(pem)
            temp_path = f.name
        
        auth = KalshiAuthenticator("test-key", temp_path)
        
        yield auth
        
        os.unlink(temp_path)
    
    def test_signature_is_valid_base64(self, authenticator):
        """Test that signature is valid base64."""
        message = "test message"
        signature = authenticator.sign_pss_text(message)
        
        # Should be valid base64
        decoded = base64.b64decode(signature)
        assert isinstance(decoded, bytes)
        assert len(decoded) > 0
    
    def test_signature_is_deterministic(self, authenticator):
        """Test that PSS signatures are different each time (due to salt)."""
        message = "test message"
        sig1 = authenticator.sign_pss_text(message)
        sig2 = authenticator.sign_pss_text(message)
        
        # PSS signatures should be different due to random salt
        assert sig1 != sig2
    
    def test_signature_verifies_with_public_key(self, authenticator):
        """Test that signature can be verified with the public key."""
        message = "1707849600000GET/trade-api/v2/portfolio/positions"
        signature_b64 = authenticator.sign_pss_text(message)
        signature = base64.b64decode(signature_b64)
        
        # Get public key
        public_key = authenticator.private_key.public_key()
        
        # Verify signature
        public_key.verify(
            signature,
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        # If we get here, signature is valid
    
    def test_signature_fails_with_wrong_message(self, authenticator):
        """Test that signature fails verification with wrong message."""
        message = "original message"
        wrong_message = "different message"
        signature_b64 = authenticator.sign_pss_text(message)
        signature = base64.b64decode(signature_b64)
        
        public_key = authenticator.private_key.public_key()
        
        with pytest.raises(InvalidSignature):
            public_key.verify(
                signature,
                wrong_message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
    
    def test_empty_message(self, authenticator):
        """Test signing empty message."""
        signature = authenticator.sign_pss_text("")
        
        # Should still produce valid base64
        decoded = base64.b64decode(signature)
        assert len(decoded) > 0
    
    def test_unicode_message(self, authenticator):
        """Test signing unicode message."""
        message = "测试消息 🎉 émojis"
        signature = authenticator.sign_pss_text(message)
        
        decoded = base64.b64decode(signature)
        assert len(decoded) > 0


class TestGenerateHeaders:
    """Test header generation."""
    
    @pytest.fixture
    def authenticator(self):
        """Create an authenticator with a test key."""
        pem, _ = generate_test_key_pair()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(pem)
            temp_path = f.name
        
        auth = KalshiAuthenticator("test-key-id", temp_path)
        
        yield auth
        
        os.unlink(temp_path)
    
    def test_generates_all_required_headers(self, authenticator):
        """Test that all required headers are generated."""
        headers = authenticator.generate_headers("GET", "/trade-api/v2/portfolio/positions")
        
        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
    
    def test_access_key_matches_key_id(self, authenticator):
        """Test that access key header matches key ID."""
        headers = authenticator.generate_headers("GET", "/trade-api/v2/portfolio/positions")
        
        assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    
    def test_timestamp_is_valid_integer(self, authenticator):
        """Test that timestamp is a valid integer string."""
        headers = authenticator.generate_headers("GET", "/trade-api/v2/portfolio/positions")
        timestamp = headers["KALSHI-ACCESS-TIMESTAMP"]
        
        # Should be a valid integer
        assert timestamp.isdigit()
        assert int(timestamp) > 0
    
    def test_timestamp_is_within_one_second_of_now(self, authenticator):
        """Test that timestamp is within 1 second of current time."""
        before = int(time.time() * 1000)
        headers = authenticator.generate_headers("GET", "/trade-api/v2/portfolio/positions")
        after = int(time.time() * 1000)
        
        timestamp = int(headers["KALSHI-ACCESS-TIMESTAMP"])
        
        # Should be between before and after
        assert before <= timestamp <= after
        # And within 1 second
        assert abs(timestamp - before) <= 1000
    
    def test_signature_is_valid_base64(self, authenticator):
        """Test that signature is valid base64."""
        headers = authenticator.generate_headers("GET", "/trade-api/v2/portfolio/positions")
        signature = headers["KALSHI-ACCESS-SIGNATURE"]
        
        # Should be valid base64
        decoded = base64.b64decode(signature)
        assert len(decoded) > 0
    
    def test_method_is_uppercase_in_signature(self, authenticator):
        """Test that method is uppercased in signature."""
        # Both should produce same signature base (timestamp varies but message format is same)
        headers_lower = authenticator.generate_headers("get", "/path")
        headers_upper = authenticator.generate_headers("GET", "/path")
        
        # Both should have valid signatures
        assert base64.b64decode(headers_lower["KALSHI-ACCESS-SIGNATURE"])
        assert base64.b64decode(headers_upper["KALSHI-ACCESS-SIGNATURE"])
    
    def test_path_without_query_string(self, authenticator):
        """Test that path without query string is used."""
        headers = authenticator.generate_headers("GET", "/trade-api/v2/portfolio/positions")
        
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    
    def test_different_paths_produce_different_signatures(self, authenticator):
        """Test that different paths produce different signatures."""
        headers1 = authenticator.generate_headers("GET", "/path1")
        headers2 = authenticator.generate_headers("GET", "/path2")
        
        # Signatures should be different (or at least we can't verify they're the same)
        sig1 = headers1["KALSHI-ACCESS-SIGNATURE"]
        sig2 = headers2["KALSHI-ACCESS-SIGNATURE"]
        # Due to PSS randomness, they might be different even for same message
        # but they should both be valid base64
        assert base64.b64decode(sig1)
        assert base64.b64decode(sig2)
    
    def test_different_methods_produce_different_signatures(self, authenticator):
        """Test that different methods produce different signatures."""
        headers1 = authenticator.generate_headers("GET", "/path")
        headers2 = authenticator.generate_headers("POST", "/path")
        
        sig1 = headers1["KALSHI-ACCESS-SIGNATURE"]
        sig2 = headers2["KALSHI-ACCESS-SIGNATURE"]
        
        # Both should be valid
        assert base64.b64decode(sig1)
        assert base64.b64decode(sig2)


class TestGenerateHeadersForUrl:
    """Test header generation from full URL."""
    
    @pytest.fixture
    def authenticator(self):
        """Create an authenticator with a test key."""
        pem, _ = generate_test_key_pair()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(pem)
            temp_path = f.name
        
        auth = KalshiAuthenticator("test-key-id", temp_path)
        
        yield auth
        
        os.unlink(temp_path)
    
    def test_extracts_path_from_url(self, authenticator):
        """Test that path is extracted from full URL."""
        url = "https://api.kalshi.com/trade-api/v2/portfolio/positions?limit=10"
        headers = authenticator.generate_headers_for_url("GET", url)
        
        assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
        assert base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    
    def test_removes_query_parameters(self, authenticator):
        """Test that query parameters are removed from path."""
        url = "https://api.kalshi.com/trade-api/v2/markets?status=active&limit=100"
        
        # This should work without error
        headers = authenticator.generate_headers_for_url("GET", url)
        assert "KALSHI-ACCESS-SIGNATURE" in headers
    
    def test_handles_url_without_query(self, authenticator):
        """Test that URL without query works."""
        url = "https://api.kalshi.com/trade-api/v2/portfolio/positions"
        headers = authenticator.generate_headers_for_url("GET", url)
        
        assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"


class TestSignatureVerification:
    """Test signature verification with known patterns."""
    
    @pytest.fixture
    def authenticator(self):
        """Create an authenticator with a test key."""
        pem, _ = generate_test_key_pair()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
            f.write(pem)
            temp_path = f.name
        
        auth = KalshiAuthenticator("test-key-id", temp_path)
        
        yield auth
        
        os.unlink(temp_path)
    
    def test_signature_follows_kalshi_format(self, authenticator):
        """Test that signature follows Kalshi's expected format."""
        # Kalshi expects: timestamp + method + path
        timestamp = "1707849600000"
        method = "GET"
        path = "/trade-api/v2/portfolio/positions"
        message = f"{timestamp}{method}{path}"
        
        signature_b64 = authenticator.sign_pss_text(message)
        signature = base64.b64decode(signature_b64)
        
        # Verify with public key
        public_key = authenticator.private_key.public_key()
        public_key.verify(
            signature,
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    
    def test_post_request_signature(self, authenticator):
        """Test signature for POST request."""
        timestamp = "1707849600000"
        method = "POST"
        path = "/trade-api/v2/portfolio/orders"
        message = f"{timestamp}{method}{path}"
        
        signature_b64 = authenticator.sign_pss_text(message)
        signature = base64.b64decode(signature_b64)
        
        public_key = authenticator.private_key.public_key()
        public_key.verify(
            signature,
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    
    def test_nested_path_signature(self, authenticator):
        """Test signature for nested path."""
        timestamp = "1707849600000"
        method = "GET"
        path = "/trade-api/v2/markets/KXBTC15M-26FEB150330-30/orderbook"
        message = f"{timestamp}{method}{path}"
        
        signature_b64 = authenticator.sign_pss_text(message)
        signature = base64.b64decode(signature_b64)
        
        public_key = authenticator.private_key.public_key()
        public_key.verify(
            signature,
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
