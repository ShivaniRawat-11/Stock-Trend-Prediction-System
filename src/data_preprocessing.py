import os
import logging
from datetime import timedelta

import pandas as pd
import yfinance as yf

from src.ticker_config import MIN_TRAINING_DAYS

logger = logging.getLogger("StockMarketTrendPrediction")

MIN_ROWS_REQUIRED = 100


def sanitize_ticker_for_cache(ticker: str) -> str:
    """Make ticker safe for filesystem cache keys (^ is problematic on some paths)."""
    return ticker.replace("^", "IDX_").replace("/", "_").replace("\\", "_")


def normalize_download_dates(start_date: str, end_date: str) -> tuple[str, str, str]:
    """
    Normalize user-provided dates for reliable Yahoo Finance downloads.

    yfinance treats ``end`` as exclusive, so we add one day to include the
    requested end date. If start >= end (common UI mistake), we expand the
    training window backwards automatically.

    Returns:
        (download_start, download_end_exclusive, cache_end_label)
    """
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()

    if start > end:
        logger.warning("Start date %s is after end date %s — swapping.", start.date(), end.date())
        start, end = end, start

    min_start = end - pd.Timedelta(days=MIN_TRAINING_DAYS)
    if start >= end:
        logger.warning(
            "Start and end are the same (%s). Expanding training window to %s.",
            end.date(),
            min_start.date(),
        )
        start = min_start
    elif start > min_start:
        logger.info(
            "Training window shorter than %d days — extending start from %s to %s.",
            MIN_TRAINING_DAYS,
            start.date(),
            min_start.date(),
        )
        start = min_start

    download_end = end + timedelta(days=1)
    return (
        start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _download_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download OHLCV data with fallbacks for indices and flaky responses."""
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    df = _flatten_yfinance_columns(df)
    if not df.empty:
        return df

    logger.warning("Primary yfinance download empty for '%s'. Trying Ticker.history fallback.", ticker)
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    hist = _flatten_yfinance_columns(hist)
    if not hist.empty:
        return hist

    if ticker.startswith("^") or ticker.startswith("IDX_"):
        logger.warning("Trying period='max' fallback for index '%s'.", ticker)
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        hist = _flatten_yfinance_columns(hist)
        if not hist.empty:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            filtered = hist[(hist.index >= start_dt) & (hist.index < end_dt)]
            if not filtered.empty:
                return filtered

    return pd.DataFrame()


def load_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    data_dir: str = "data",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Downloads historical stock market data from Yahoo Finance via yfinance,
    or loads it from a local cache if available and use_cache is True.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS', '^NSEI').
        start_date (str): Fetch start date in 'YYYY-MM-DD' format.
        end_date (str): Fetch end date in 'YYYY-MM-DD' format (inclusive).
        data_dir (str): Directory where CSV data is cached. Defaults to 'data'.
        use_cache (bool): Whether to look for and save to cache. Defaults to True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the historical stock data.
    """
    download_start, download_end, cache_end = normalize_download_dates(start_date, end_date)
    cache_ticker = sanitize_ticker_for_cache(ticker)
    file_path = os.path.join(data_dir, f"{cache_ticker}_{download_start}_{cache_end}.csv")

    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
        except OSError as e:
            logger.warning("Could not create data directory %s: %s. Caching will be disabled.", data_dir, e)

    if use_cache and os.path.exists(file_path):
        logger.info("Loading stock data for ticker '%s' from local cache: %s", ticker, file_path)
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            if len(df) >= MIN_ROWS_REQUIRED:
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                return df
            logger.warning("Cached file for '%s' has insufficient rows (%d). Re-downloading.", ticker, len(df))
        except Exception as e:
            logger.warning("Failed to load cached file. Proceeding with download. Error: %s", e)

    logger.info(
        "Downloading historical data for '%s' from %s to %s (exclusive end %s) via yfinance...",
        ticker,
        download_start,
        cache_end,
        download_end,
    )
    try:
        df = _download_ticker_data(ticker, download_start, download_end)
        if df.empty:
            raise ValueError(
                f"No data retrieved for ticker symbol '{ticker}' between {download_start} and {cache_end}. "
                "Check the symbol, exchange suffix (.NS/.BO), and date range."
            )

        df = _flatten_yfinance_columns(df)
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index)

        if use_cache:
            try:
                df.to_csv(file_path)
                logger.info("Data cached to %s", file_path)
            except OSError as e:
                logger.warning("Failed to save data cache to %s (running in memory without cache): %s", file_path, e)

        return df

    except Exception as e:
        logger.error("Error occurred while fetching stock data for '%s': %s", ticker, e)
        raise e


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans stock data by identifying, logging, and filling/dropping missing values,
    and verifying data integrity.

    Args:
        df (pd.DataFrame): Input raw stock data DataFrame.

    Returns:
        pd.DataFrame: Cleaned stock data DataFrame.
    """
    logger.info("Initializing data cleaning process...")
    cleaned_df = df.copy()

    logger.info("Initial dataset shape: %s", cleaned_df.shape)

    missing_counts = cleaned_df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        logger.warning("Found %d missing values in the raw dataset:\n%s", total_missing, missing_counts)
        cleaned_df = cleaned_df.ffill().bfill()

        remaining_missing = cleaned_df.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning("Remaining missing values after ffill/bfill: %d. Dropping residual NaNs.", remaining_missing)
            cleaned_df.dropna(inplace=True)
    else:
        logger.info("No missing values found in the dataset.")

    if cleaned_df.index.duplicated().any():
        logger.warning("Duplicate dates found in dataset index. Dropping duplicates, keeping the first occurrence.")
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="first")]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in cleaned_df.columns:
            lower_cols = {c.lower(): c for c in cleaned_df.columns}
            if col.lower() in lower_cols:
                logger.info("Renaming column '%s' to '%s'", lower_cols[col.lower()], col)
                cleaned_df.rename(columns={lower_cols[col.lower()]: col}, inplace=True)
            else:
                raise ValueError(f"Required column '{col}' is missing from the stock dataset.")

    for col in required_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    nans_after_cast = cleaned_df[required_cols].isnull().sum().sum()
    if nans_after_cast > 0:
        logger.warning("Dropping %d rows due to non-numeric casting issues.", nans_after_cast)
        cleaned_df.dropna(subset=required_cols, inplace=True)

    if len(cleaned_df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Insufficient data after cleaning ({len(cleaned_df)} rows). "
            f"Need at least {MIN_ROWS_REQUIRED} trading days for model training."
        )

    logger.info("Data cleaning completed successfully. Final shape: %s", cleaned_df.shape)
    return cleaned_df
