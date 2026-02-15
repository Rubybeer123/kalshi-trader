"""Market discovery for 15-minute crypto markets."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Pattern, Tuple

import structlog

from src.data.models import Market

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CryptoMarket:
    """Represents a 15-minute crypto market on Kalshi."""
    
    ticker: str
    asset: str
    strike: float
    expiration: datetime
    time_remaining: timedelta
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    last_price: float
    
    @property
    def is_call(self) -> bool:
        """True if this is a CALL option (strike below current price)."""
        return self.strike < self.last_price
    
    @property
    def is_put(self) -> bool:
        """True if this is a PUT option (strike above current price)."""
        return self.strike > self.last_price
    
    @property
    def is_at_the_money(self) -> bool:
        """True if strike is near current price."""
        return abs(self.strike - self.last_price) / self.last_price < 0.01
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price from yes bid/ask."""
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price
    
    @property
    def spread(self) -> float:
        """Calculate spread from yes bid/ask."""
        if self.yes_bid > 0 and self.yes_ask > 0:
            return self.yes_ask - self.yes_bid
        return 0.0
    
    @property
    def is_approaching_expiration(self) -> bool:
        """True if market expires within 5 minutes."""
        return self.time_remaining < timedelta(minutes=5)
    
    @property
    def is_expired(self) -> bool:
        """True if market has expired."""
        return self.time_remaining.total_seconds() <= 0


class MarketDiscovery:
    """Discovers and filters 15-minute crypto markets."""
    
    # Supported crypto assets and their ticker prefixes
    CRYPTO_ASSETS = {
        "BTC": ["KXBTC", "KBTC"],
        "ETH": ["KETH", "KXETH"],
        "SOL": ["KSOL", "KXSOL"],
        "XRP": ["KXRP", "KXRP"],
    }
    
    # Ticker pattern: PREFIX + MMMDD + HHMM + STRIKE
    # Example: KXBTC15M-26FEB150330-30
    TICKER_PATTERN = re.compile(
        r"^(?P<prefix>[A-Z]+)"           # Asset prefix (KXBTC, KETH, etc.)
        r"(?P<duration>\d+[MH])"          # Duration (15M, 1H, etc.)
        r"-"
        r"(?P<day>\d{1,2})"               # Day
        r"(?P<month>[A-Z]{3})"            # Month
        r"(?P<hour>\d{2})"                # Hour
        r"(?P<minute>\d{2})"              # Minute
        r"-"
        r"(?P<strike>\d+)"                # Strike price
        r"$",
        re.IGNORECASE
    )
    
    # Month abbreviation to number mapping
    MONTH_MAP = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }
    
    def __init__(self, year: Optional[int] = None):
        """
        Initialize market discovery.
        
        Args:
            year: Year for expiration dates (defaults to current year)
        """
        self.year = year or datetime.now(timezone.utc).year
        self._crypto_market_cache: List[CryptoMarket] = []
        self._last_update: Optional[datetime] = None
    
    def is_crypto_market(self, ticker: str) -> bool:
        """
        Check if ticker is a crypto market.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            True if this is a crypto market
        """
        ticker_upper = ticker.upper()
        
        for asset, prefixes in self.CRYPTO_ASSETS.items():
            for prefix in prefixes:
                if ticker_upper.startswith(prefix):
                    return True
        
        return False
    
    def get_asset_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Extract asset from ticker.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            Asset code (BTC, ETH, SOL, XRP) or None
        """
        ticker_upper = ticker.upper()
        
        for asset, prefixes in self.CRYPTO_ASSETS.items():
            for prefix in prefixes:
                if ticker_upper.startswith(prefix):
                    return asset
        
        return None
    
    def parse_ticker(self, ticker: str) -> Tuple[str, datetime, float]:
        """
        Parse ticker to extract asset, expiration, and strike.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            Tuple of (asset, expiration_datetime, strike_price)
            
        Raises:
            ValueError: If ticker format is invalid
        """
        match = self.TICKER_PATTERN.match(ticker)
        
        if not match:
            raise ValueError(f"Invalid ticker format: {ticker}")
        
        groups = match.groupdict()
        
        # Extract asset
        asset = self.get_asset_from_ticker(ticker)
        if not asset:
            raise ValueError(f"Unknown asset in ticker: {ticker}")
        
        # Parse expiration
        try:
            day = int(groups["day"])
            month = self.MONTH_MAP.get(groups["month"].upper())
            if not month:
                raise ValueError(f"Invalid month: {groups['month']}")
            
            hour = int(groups["hour"])
            minute = int(groups["minute"])
            
            # Create expiration datetime in UTC
            expiration = datetime(
                year=self.year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
                tzinfo=timezone.utc
            )
            
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to parse expiration from ticker {ticker}: {e}")
        
        # Parse strike price
        try:
            strike = float(groups["strike"])
        except ValueError:
            raise ValueError(f"Invalid strike price in ticker: {ticker}")
        
        return asset, expiration, strike
    
    def is_15min_market(self, ticker: str) -> bool:
        """
        Check if ticker is a 15-minute market.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            True if this is a 15-minute market
        """
        match = self.TICKER_PATTERN.match(ticker)
        if not match:
            return False
        
        duration = match.group("duration").upper()
        return duration == "15M"
    
    def market_to_crypto_market(
        self,
        market: Market,
        reference_time: Optional[datetime] = None
    ) -> Optional[CryptoMarket]:
        """
        Convert Market model to CryptoMarket.
        
        Args:
            market: Market data model
            reference_time: Time to calculate remaining from (defaults to now)
            
        Returns:
            CryptoMarket or None if not a crypto market
        """
        if not self.is_crypto_market(market.ticker):
            return None
        
        if not self.is_15min_market(market.ticker):
            return None
        
        try:
            asset, expiration, strike = self.parse_ticker(market.ticker)
        except ValueError as e:
            logger.warning("Failed to parse ticker", ticker=market.ticker, error=str(e))
            return None
        
        # Calculate time remaining
        now = reference_time or datetime.now(timezone.utc)
        time_remaining = expiration - now
        
        # Get prices with defaults
        yes_bid = float(market.yes_bid) if market.yes_bid else 0.0
        yes_ask = float(market.yes_ask) if market.yes_ask else 0.0
        no_bid = float(market.no_bid) if market.no_bid else 0.0
        no_ask = float(market.no_ask) if market.no_ask else 0.0
        
        # Calculate last price from mid
        if yes_bid > 0 and yes_ask > 0:
            last_price = (yes_bid + yes_ask) / 2
        else:
            last_price = 0.5  # Default to 50% if no price data
        
        return CryptoMarket(
            ticker=market.ticker,
            asset=asset,
            strike=strike,
            expiration=expiration,
            time_remaining=time_remaining,
            yes_ask=yes_ask,
            yes_bid=yes_bid,
            no_ask=no_ask,
            no_bid=no_bid,
            last_price=last_price
        )
    
    def filter_crypto_markets(
        self,
        markets: List[Market],
        assets: Optional[List[str]] = None,
        only_15min: bool = True,
        status: str = "open",
        reference_time: Optional[datetime] = None
    ) -> List[CryptoMarket]:
        """
        Filter markets for crypto 15-minute markets.
        
        Args:
            markets: List of markets from API
            assets: Filter by specific assets (BTC, ETH, SOL, XRP)
            only_15min: Only include 15-minute markets
            status: Filter by status (default: "open")
            reference_time: Time to calculate remaining from
            
        Returns:
            List of CryptoMarket objects
        """
        crypto_markets = []
        
        for market in markets:
            # Filter by status
            if market.status.value != status:
                continue
            
            # Check if crypto market
            if not self.is_crypto_market(market.ticker):
                continue
            
            # Filter by asset if specified
            if assets:
                asset = self.get_asset_from_ticker(market.ticker)
                if asset not in assets:
                    continue
            
            # Check duration
            if only_15min and not self.is_15min_market(market.ticker):
                continue
            
            # Convert to CryptoMarket
            crypto_market = self.market_to_crypto_market(market, reference_time)
            if crypto_market:
                crypto_markets.append(crypto_market)
        
        # Sort by expiration time
        crypto_markets.sort(key=lambda m: m.expiration)
        
        return crypto_markets
    
    def get_active_cycles(
        self,
        crypto_markets: List[CryptoMarket]
    ) -> dict:
        """
        Group markets by their 15-minute cycle.
        
        Args:
            crypto_markets: List of crypto markets
            
        Returns:
            Dict mapping expiration time to list of markets
        """
        cycles = {}
        
        for market in crypto_markets:
            # Round expiration to 15-minute bucket
            exp = market.expiration
            minute_bucket = (exp.minute // 15) * 15
            cycle_time = exp.replace(minute=minute_bucket, second=0, microsecond=0)
            
            if cycle_time not in cycles:
                cycles[cycle_time] = []
            cycles[cycle_time].append(market)
        
        return cycles
    
    def get_expiring_markets(
        self,
        crypto_markets: List[CryptoMarket],
        warning_threshold: timedelta = timedelta(minutes=5)
    ) -> List[CryptoMarket]:
        """
        Get markets approaching expiration.
        
        Args:
            crypto_markets: List of crypto markets
            warning_threshold: Time threshold for warning
            
        Returns:
            List of markets expiring within threshold
        """
        return [
            m for m in crypto_markets
            if m.time_remaining <= warning_threshold and m.time_remaining > timedelta(0)
        ]
    
    def get_strike_ladder(
        self,
        crypto_markets: List[CryptoMarket],
        asset: str
    ) -> List[CryptoMarket]:
        """
        Get strike ladder for a specific asset.
        
        Args:
            crypto_markets: List of crypto markets
            asset: Asset to filter by (BTC, ETH, SOL, XRP)
            
        Returns:
            List of markets sorted by strike price
        """
        asset_markets = [m for m in crypto_markets if m.asset == asset]
        return sorted(asset_markets, key=lambda m: m.strike)
    
    def find_market_by_strike(
        self,
        crypto_markets: List[CryptoMarket],
        asset: str,
        strike: float,
        tolerance: float = 1.0
    ) -> Optional[CryptoMarket]:
        """
        Find a market by strike price.
        
        Args:
            crypto_markets: List of crypto markets
            asset: Asset to search
            strike: Target strike price
            tolerance: Acceptable difference from target
            
        Returns:
            Matching market or None
        """
        for market in crypto_markets:
            if market.asset == asset and abs(market.strike - strike) <= tolerance:
                return market
        return None
