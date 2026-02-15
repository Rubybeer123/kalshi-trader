"""Tests for market discovery."""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.data.models import Market, MarketStatus
from src.market_discovery import (
    CryptoMarket,
    MarketDiscovery,
)


class TestCryptoMarket:
    """Test CryptoMarket dataclass."""
    
    def test_creation(self):
        """Test CryptoMarket creation."""
        market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=30000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(minutes=10),
            yes_ask=0.65,
            yes_bid=0.63,
            no_ask=0.37,
            no_bid=0.35,
            last_price=0.64
        )
        
        assert market.ticker == "KXBTC15M-26FEB150330-30"
        assert market.asset == "BTC"
        assert market.strike == 30000.0
    
    def test_mid_price_calculation(self):
        """Test mid price calculation."""
        market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=30000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(minutes=10),
            yes_ask=0.70,
            yes_bid=0.60,
            no_ask=0.40,
            no_bid=0.30,
            last_price=0.65
        )
        
        assert market.mid_price == 0.65  # (0.60 + 0.70) / 2
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=30000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(minutes=10),
            yes_ask=0.70,
            yes_bid=0.60,
            no_ask=0.40,
            no_bid=0.30,
            last_price=0.65
        )
        
        assert market.spread == 0.10  # 0.70 - 0.60
    
    def test_is_call(self):
        """Test call option detection."""
        # Call: strike < last_price
        call_market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=30000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(minutes=10),
            yes_ask=0.70,
            yes_bid=0.60,
            no_ask=0.40,
            no_bid=0.30,
            last_price=35000.0  # Price above strike
        )
        
        assert call_market.is_call is True
        assert call_market.is_put is False
    
    def test_is_put(self):
        """Test put option detection."""
        # Put: strike > last_price
        put_market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=40000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(minutes=10),
            yes_ask=0.30,
            yes_bid=0.25,
            no_ask=0.75,
            no_bid=0.70,
            last_price=35000.0  # Price below strike
        )
        
        assert put_market.is_put is True
        assert put_market.is_call is False
    
    def test_is_at_the_money(self):
        """Test at-the-money detection."""
        atm_market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=35000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(minutes=10),
            yes_ask=0.52,
            yes_bid=0.48,
            no_ask=0.52,
            no_bid=0.48,
            last_price=35100.0  # Within 1%
        )
        
        assert atm_market.is_at_the_money is True
    
    def test_is_approaching_expiration(self):
        """Test expiration warning detection."""
        expiring_market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=30000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(minutes=3),  # Less than 5 min
            yes_ask=0.70,
            yes_bid=0.60,
            no_ask=0.40,
            no_bid=0.30,
            last_price=0.65
        )
        
        assert expiring_market.is_approaching_expiration is True
    
    def test_is_expired(self):
        """Test expired detection."""
        expired_market = CryptoMarket(
            ticker="KXBTC15M-26FEB150330-30",
            asset="BTC",
            strike=30000.0,
            expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
            time_remaining=timedelta(seconds=-10),  # Negative time
            yes_ask=0.0,
            yes_bid=0.0,
            no_ask=0.0,
            no_bid=0.0,
            last_price=0.0
        )
        
        assert expired_market.is_expired is True


class TestMarketDiscoveryInit:
    """Test MarketDiscovery initialization."""
    
    def test_default_year(self):
        """Test default year is current year."""
        discovery = MarketDiscovery()
        assert discovery.year == datetime.now(timezone.utc).year
    
    def test_custom_year(self):
        """Test custom year setting."""
        discovery = MarketDiscovery(year=2025)
        assert discovery.year == 2025


class TestIsCryptoMarket:
    """Test crypto market detection."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_detects_btc_markets(self, discovery):
        """Test BTC market detection."""
        assert discovery.is_crypto_market("KXBTC15M-26FEB150330-30") is True
        assert discovery.is_crypto_market("KBTC15M-26FEB150330-30") is True
    
    def test_detects_eth_markets(self, discovery):
        """Test ETH market detection."""
        assert discovery.is_crypto_market("KETH15M-26FEB150330-30") is True
        assert discovery.is_crypto_market("KXETH15M-26FEB150330-30") is True
    
    def test_detects_sol_markets(self, discovery):
        """Test SOL market detection."""
        assert discovery.is_crypto_market("KSOL15M-26FEB150330-30") is True
        assert discovery.is_crypto_market("KXSOL15M-26FEB150330-30") is True
    
    def test_detects_xrp_markets(self, discovery):
        """Test XRP market detection."""
        assert discovery.is_crypto_market("KXRP15M-26FEB150330-30") is True
    
    def test_rejects_non_crypto_markets(self, discovery):
        """Test non-crypto markets are rejected."""
        assert discovery.is_crypto_market("CPI-26FEB-5.0") is False
        assert discovery.is_crypto_market("WEATHER-NYC-26FEB") is False
        assert discovery.is_crypto_market("SPX-26FEB5000") is False


class TestGetAssetFromTicker:
    """Test asset extraction from ticker."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_extracts_btc(self, discovery):
        """Test BTC extraction."""
        assert discovery.get_asset_from_ticker("KXBTC15M-26FEB150330-30") == "BTC"
    
    def test_extracts_eth(self, discovery):
        """Test ETH extraction."""
        assert discovery.get_asset_from_ticker("KETH15M-26FEB150330-30") == "ETH"
    
    def test_extracts_sol(self, discovery):
        """Test SOL extraction."""
        assert discovery.get_asset_from_ticker("KSOL15M-26FEB150330-30") == "SOL"
    
    def test_extracts_xrp(self, discovery):
        """Test XRP extraction."""
        assert discovery.get_asset_from_ticker("KXRP15M-26FEB150330-30") == "XRP"
    
    def test_returns_none_for_non_crypto(self, discovery):
        """Test None returned for non-crypto."""
        assert discovery.get_asset_from_ticker("CPI-26FEB-5.0") is None


class TestParseTicker:
    """Test ticker parsing."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_parse_valid_btc_ticker(self, discovery):
        """Test parsing valid BTC ticker."""
        asset, expiration, strike = discovery.parse_ticker("KXBTC15M-26FEB150330-30")
        
        assert asset == "BTC"
        assert strike == 30.0
        assert expiration.year == 2024
        assert expiration.month == 2  # FEB
        assert expiration.day == 26
        assert expiration.hour == 15
        assert expiration.minute == 30
    
    def test_parse_valid_eth_ticker(self, discovery):
        """Test parsing valid ETH ticker."""
        asset, expiration, strike = discovery.parse_ticker("KETH15M-15MAR090045-3500")
        
        assert asset == "ETH"
        assert strike == 3500.0
        assert expiration.month == 3  # MAR
        assert expiration.day == 15
        assert expiration.hour == 9
        assert expiration.minute == 0
    
    def test_parse_all_months(self, discovery):
        """Test parsing all month abbreviations."""
        months = [
            ("JAN", 1), ("FEB", 2), ("MAR", 3), ("APR", 4),
            ("MAY", 5), ("JUN", 6), ("JUL", 7), ("AUG", 8),
            ("SEP", 9), ("OCT", 10), ("NOV", 11), ("DEC", 12)
        ]
        
        for month_abbr, month_num in months:
            ticker = f"KXBTC15M-26{month_abbr}150330-30"
            asset, expiration, strike = discovery.parse_ticker(ticker)
            assert expiration.month == month_num, f"Failed for {month_abbr}"
    
    def test_raises_on_invalid_ticker_format(self, discovery):
        """Test error on invalid format."""
        with pytest.raises(ValueError, match="Invalid ticker format"):
            discovery.parse_ticker("INVALID-TICKER")
    
    def test_raises_on_unknown_asset(self, discovery):
        """Test error on unknown asset."""
        with pytest.raises(ValueError, match="Unknown asset"):
            discovery.parse_ticker("KUNKNOWN15M-26FEB150330-30")
    
    def test_raises_on_invalid_month(self, discovery):
        """Test error on invalid month."""
        with pytest.raises(ValueError, match="Invalid month"):
            discovery.parse_ticker("KXBTC15M-26XYZ150330-30")


class TestIs15MinMarket:
    """Test 15-minute market detection."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_detects_15min_markets(self, discovery):
        """Test 15M detection."""
        assert discovery.is_15min_market("KXBTC15M-26FEB150330-30") is True
    
    def test_rejects_1hour_markets(self, discovery):
        """Test rejection of 1H markets."""
        assert discovery.is_15min_market("KXBTC1H-26FEB150330-30") is False
    
    def test_rejects_5min_markets(self, discovery):
        """Test rejection of 5M markets."""
        assert discovery.is_15min_market("KXBTC5M-26FEB150330-30") is False


class TestMarketToCryptoMarket:
    """Test Market to CryptoMarket conversion."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_converts_valid_market(self, discovery):
        """Test conversion of valid market."""
        market = Market(
            ticker="KXBTC15M-26FEB150330-30",
            title="BTC $30K 15-min",
            category="crypto",
            status=MarketStatus.OPEN,
            yes_bid=Decimal("0.60"),
            yes_ask=Decimal("0.65"),
            no_bid=Decimal("0.35"),
            no_ask=Decimal("0.40"),
            volume=1000,
            open_interest=500
        )
        
        crypto = discovery.market_to_crypto_market(market)
        
        assert crypto is not None
        assert crypto.ticker == "KXBTC15M-26FEB150330-30"
        assert crypto.asset == "BTC"
        assert crypto.strike == 30.0
    
    def test_returns_none_for_non_crypto(self, discovery):
        """Test None returned for non-crypto."""
        market = Market(
            ticker="CPI-26FEB-5.0",
            title="CPI Market",
            category="economics",
            status=MarketStatus.OPEN,
        )
        
        crypto = discovery.market_to_crypto_market(market)
        
        assert crypto is None
    
    def test_returns_none_for_non_15min(self, discovery):
        """Test None returned for non-15min."""
        market = Market(
            ticker="KXBTC1H-26FEB150330-30",
            title="BTC 1-hour",
            category="crypto",
            status=MarketStatus.OPEN,
        )
        
        crypto = discovery.market_to_crypto_market(market)
        
        assert crypto is None


class TestFilterCryptoMarkets:
    """Test market filtering."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    @pytest.fixture
    def mock_markets(self):
        """Create mock market list."""
        return [
            Market(
                ticker="KXBTC15M-26FEB150330-30",
                title="BTC $30K",
                category="crypto",
                status=MarketStatus.OPEN,
                yes_bid=Decimal("0.60"),
                yes_ask=Decimal("0.65"),
            ),
            Market(
                ticker="KETH15M-26FEB150330-3500",
                title="ETH $3500",
                category="crypto",
                status=MarketStatus.OPEN,
                yes_bid=Decimal("0.55"),
                yes_ask=Decimal("0.60"),
            ),
            Market(
                ticker="KSOL15M-26FEB150330-150",
                title="SOL $150",
                category="crypto",
                status=MarketStatus.OPEN,
                yes_bid=Decimal("0.50"),
                yes_ask=Decimal("0.55"),
            ),
            Market(
                ticker="KXBTC1H-26FEB160330-30",  # Not 15min
                title="BTC 1H",
                category="crypto",
                status=MarketStatus.OPEN,
            ),
            Market(
                ticker="CPI-26FEB-5.0",  # Not crypto
                title="CPI",
                category="economics",
                status=MarketStatus.OPEN,
            ),
            Market(
                ticker="KXBTC15M-26FEB140330-30",  # Closed
                title="BTC Closed",
                category="crypto",
                status=MarketStatus.CLOSED,
            ),
        ]
    
    def test_filters_only_open_15min_crypto(self, discovery, mock_markets):
        """Test filtering for open 15-min crypto markets."""
        crypto_markets = discovery.filter_crypto_markets(mock_markets)
        
        assert len(crypto_markets) == 3  # BTC, ETH, SOL
        
        tickers = {m.ticker for m in crypto_markets}
        assert "KXBTC15M-26FEB150330-30" in tickers
        assert "KETH15M-26FEB150330-3500" in tickers
        assert "KSOL15M-26FEB150330-150" in tickers
    
    def test_filters_by_asset(self, discovery, mock_markets):
        """Test filtering by specific asset."""
        crypto_markets = discovery.filter_crypto_markets(
            mock_markets,
            assets=["BTC"]
        )
        
        assert len(crypto_markets) == 1
        assert crypto_markets[0].asset == "BTC"
    
    def test_filters_by_multiple_assets(self, discovery, mock_markets):
        """Test filtering by multiple assets."""
        crypto_markets = discovery.filter_crypto_markets(
            mock_markets,
            assets=["BTC", "ETH"]
        )
        
        assert len(crypto_markets) == 2
        assets = {m.asset for m in crypto_markets}
        assert assets == {"BTC", "ETH"}
    
    def test_includes_non_15min_when_flag_false(self, discovery, mock_markets):
        """Test including non-15min markets when flag is False."""
        crypto_markets = discovery.filter_crypto_markets(
            mock_markets,
            only_15min=False
        )
        
        # Should include the 1H BTC market
        tickers = {m.ticker for m in crypto_markets}
        assert "KXBTC1H-26FEB160330-30" in tickers


class TestGetActiveCycles:
    """Test cycle grouping."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_groups_by_expiration(self, discovery):
        """Test grouping markets by expiration."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.65,
                yes_bid=0.60,
                no_ask=0.40,
                no_bid=0.35,
                last_price=0.62
            ),
            CryptoMarket(
                ticker="KETH15M-26FEB150330-3500",
                asset="ETH",
                strike=3500.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.55,
                yes_bid=0.50,
                no_ask=0.50,
                no_bid=0.45,
                last_price=0.52
            ),
            CryptoMarket(
                ticker="KXBTC15M-26FEB150345-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 45, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=25),
                yes_ask=0.60,
                yes_bid=0.55,
                no_ask=0.45,
                no_bid=0.40,
                last_price=0.57
            ),
        ]
        
        cycles = discovery.get_active_cycles(markets)
        
        # Should have 2 cycles (15:30 and 15:45)
        assert len(cycles) == 2
        
        # 15:30 cycle should have BTC and ETH
        cycle_1530 = datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc)
        assert len(cycles[cycle_1530]) == 2
        
        # 15:45 cycle should have just BTC
        cycle_1545 = datetime(2024, 2, 26, 15, 45, tzinfo=timezone.utc)
        assert len(cycles[cycle_1545]) == 1


class TestGetExpiringMarkets:
    """Test expiration warning."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_finds_expiring_markets(self, discovery):
        """Test finding markets near expiration."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=3),  # Expiring soon
                yes_ask=0.65,
                yes_bid=0.60,
                no_ask=0.40,
                no_bid=0.35,
                last_price=0.62
            ),
            CryptoMarket(
                ticker="KETH15M-26FEB150330-3500",
                asset="ETH",
                strike=3500.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=15),  # Not expiring
                yes_ask=0.55,
                yes_bid=0.50,
                no_ask=0.50,
                no_bid=0.45,
                last_price=0.52
            ),
        ]
        
        expiring = discovery.get_expiring_markets(markets)
        
        assert len(expiring) == 1
        assert expiring[0].ticker == "KXBTC15M-26FEB150330-30"
    
    def test_respects_custom_threshold(self, discovery):
        """Test custom expiration threshold."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.65,
                yes_bid=0.60,
                no_ask=0.40,
                no_bid=0.35,
                last_price=0.62
            ),
        ]
        
        # With 15 minute threshold, should be included
        expiring = discovery.get_expiring_markets(
            markets,
            warning_threshold=timedelta(minutes=15)
        )
        
        assert len(expiring) == 1


class TestGetStrikeLadder:
    """Test strike ladder generation."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_returns_sorted_strikes(self, discovery):
        """Test strike ladder is sorted."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-50",
                asset="BTC",
                strike=50.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.30,
                yes_bid=0.25,
                no_ask=0.75,
                no_bid=0.70,
                last_price=0.27
            ),
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.70,
                yes_bid=0.65,
                no_ask=0.35,
                no_bid=0.30,
                last_price=0.67
            ),
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-40",
                asset="BTC",
                strike=40.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.50,
                yes_bid=0.45,
                no_ask=0.55,
                no_bid=0.50,
                last_price=0.47
            ),
        ]
        
        ladder = discovery.get_strike_ladder(markets, "BTC")
        
        assert len(ladder) == 3
        assert ladder[0].strike == 30.0
        assert ladder[1].strike == 40.0
        assert ladder[2].strike == 50.0
    
    def test_filters_by_asset(self, discovery):
        """Test strike ladder filters by asset."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.65,
                yes_bid=0.60,
                no_ask=0.40,
                no_bid=0.35,
                last_price=0.62
            ),
            CryptoMarket(
                ticker="KETH15M-26FEB150330-3500",
                asset="ETH",
                strike=3500.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.55,
                yes_bid=0.50,
                no_ask=0.50,
                no_bid=0.45,
                last_price=0.52
            ),
        ]
        
        btc_ladder = discovery.get_strike_ladder(markets, "BTC")
        
        assert len(btc_ladder) == 1
        assert btc_ladder[0].asset == "BTC"


class TestFindMarketByStrike:
    """Test finding market by strike."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_finds_exact_match(self, discovery):
        """Test finding exact strike match."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.65,
                yes_bid=0.60,
                no_ask=0.40,
                no_bid=0.35,
                last_price=0.62
            ),
        ]
        
        found = discovery.find_market_by_strike(markets, "BTC", 30.0)
        
        assert found is not None
        assert found.ticker == "KXBTC15M-26FEB150330-30"
    
    def test_finds_within_tolerance(self, discovery):
        """Test finding within tolerance."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.65,
                yes_bid=0.60,
                no_ask=0.40,
                no_bid=0.35,
                last_price=0.62
            ),
        ]
        
        found = discovery.find_market_by_strike(markets, "BTC", 31.0, tolerance=2.0)
        
        assert found is not None
    
    def test_returns_none_if_not_found(self, discovery):
        """Test None returned if no match."""
        markets = [
            CryptoMarket(
                ticker="KXBTC15M-26FEB150330-30",
                asset="BTC",
                strike=30.0,
                expiration=datetime(2024, 2, 26, 15, 30, tzinfo=timezone.utc),
                time_remaining=timedelta(minutes=10),
                yes_ask=0.65,
                yes_bid=0.60,
                no_ask=0.40,
                no_bid=0.35,
                last_price=0.62
            ),
        ]
        
        found = discovery.find_market_by_strike(markets, "BTC", 100.0)
        
        assert found is None


class TestTimezoneHandling:
    """Test timezone handling."""
    
    @pytest.fixture
    def discovery(self):
        return MarketDiscovery(year=2024)
    
    def test_expiration_is_utc(self, discovery):
        """Test that parsed expiration is in UTC."""
        asset, expiration, strike = discovery.parse_ticker("KXBTC15M-26FEB150330-30")
        
        assert expiration.tzinfo is not None
        assert expiration.tzinfo == timezone.utc
    
    def test_time_remaining_calculation(self, discovery):
        """Test time remaining uses UTC."""
        from datetime import timezone
        
        # Create market with UTC expiration
        market = Market(
            ticker="KXBTC15M-26FEB150330-30",
            title="BTC",
            category="crypto",
            status=MarketStatus.OPEN,
        )
        
        # Use specific reference time
        reference = datetime(2024, 2, 26, 15, 20, tzinfo=timezone.utc)  # 10 min before
        
        crypto = discovery.market_to_crypto_market(market, reference_time=reference)
        
        assert crypto.time_remaining == timedelta(minutes=10)
