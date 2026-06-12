"""Unit tests for market ticker configuration and symbol resolution."""

import pytest

from src.ticker_config import (
    INDIAN_INDICES,
    INDIAN_STOCKS,
    US_STOCKS,
    all_supported_tickers,
    is_market_index,
    resolve_indian_ticker,
)


class TestResolveIndianTicker:
    @pytest.mark.parametrize(
        "raw, exchange, expected",
        [
            ("BEL", "NSE", "BEL.NS"),
            ("tcs", "NSE", "TCS.NS"),
            ("ADANI", "NSE", "ADANIENT.NS"),
            ("TATAMOTORS", "NSE", "TMPV.NS"),
            ("ZOMATO", "NSE", "ETERNAL.NS"),
            ("RELIANCE.NS", "NSE", "RELIANCE.NS"),
            ("SBIN", "BSE", "SBIN.BO"),
            ("^NSEI", "NSE", "^NSEI"),
            ("  infy  ", "NSE", "INFY.NS"),
        ],
    )
    def test_resolve_symbols(self, raw: str, exchange: str, expected: str) -> None:
        assert resolve_indian_ticker(raw, exchange) == expected

    def test_empty_symbol_returns_empty_string(self) -> None:
        assert resolve_indian_ticker("") == ""
        assert resolve_indian_ticker("   ") == ""


class TestTickerCatalog:
    def test_indian_indices_include_sensex_and_nifty(self) -> None:
        assert INDIAN_INDICES["SENSEX"] == "^BSESN"
        assert INDIAN_INDICES["NIFTY 50"] == "^NSEI"

    def test_indian_stocks_include_bel(self) -> None:
        assert INDIAN_STOCKS["BEL (Bharat Electronics)"] == "BEL.NS"

    def test_all_supported_tickers_unique(self) -> None:
        tickers = all_supported_tickers()
        assert len(tickers) == len(set(tickers))

    @pytest.mark.parametrize(
        "ticker, expected",
        [
            ("^BSESN", True),
            ("^NSEI", True),
            ("BEL.NS", False),
            ("AAPL", False),
        ],
    )
    def test_is_market_index(self, ticker: str, expected: bool) -> None:
        assert is_market_index(ticker) is expected

    def test_us_stocks_not_empty(self) -> None:
        assert len(US_STOCKS) >= 5
