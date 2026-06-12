"""Ticker symbols and helpers for US and Indian markets."""

from __future__ import annotations

US_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOG)": "GOOG",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
}

INDIAN_INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
}

INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Tata Motors": "TMPV.NS",
    "Wipro": "WIPRO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "BEL (Bharat Electronics)": "BEL.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Power": "ADANIPOWER.NS",
    "Zomato (Eternal)": "ETERNAL.NS",
    "Paytm (One97)": "PAYTM.NS",
}

INDIAN_SYMBOL_ALIASES = {
    "ADANI": "ADANIENT",
    "SBI": "SBIN",
    "HDFC": "HDFCBANK",
    "RELIANCE": "RELIANCE",
    "TATA": "TMPV",
    "TATAMOTORS": "TMPV",
    "ZOMATO": "ETERNAL",
}

MIN_TRAINING_DAYS = 730  # ~2 years of calendar days


def resolve_indian_ticker(symbol: str, exchange: str = "NSE") -> str:
    """Turn user input like BEL, ADANI, TCS into a Yahoo Finance ticker."""
    symbol = symbol.strip().upper().replace(" ", "")
    if not symbol:
        return ""

    if symbol.startswith("^"):
        return symbol

    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol

    base = INDIAN_SYMBOL_ALIASES.get(symbol, symbol)
    suffix = ".BO" if exchange.upper() == "BSE" else ".NS"
    return f"{base}{suffix}"


def is_market_index(ticker: str) -> bool:
    """Return True for Yahoo Finance index symbols (e.g. ^NSEI)."""
    return ticker.startswith("^")


def all_supported_tickers() -> list[str]:
    """Flat list of every pre-configured ticker in the app."""
    return list(US_STOCKS.values()) + list(INDIAN_INDICES.values()) + list(INDIAN_STOCKS.values())
