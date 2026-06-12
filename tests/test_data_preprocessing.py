"""Unit tests for data download, date normalization, and cleaning."""

from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_preprocessing import (
    clean_data,
    load_stock_data,
    normalize_download_dates,
    sanitize_ticker_for_cache,
)


class TestSanitizeTickerForCache:
    @pytest.mark.parametrize(
        "ticker, expected",
        [
            ("^BSESN", "IDX_BSESN"),
            ("^NSEI", "IDX_NSEI"),
            ("BEL.NS", "BEL.NS"),
            ("AAPL", "AAPL"),
        ],
    )
    def test_sanitize(self, ticker: str, expected: str) -> None:
        assert sanitize_ticker_for_cache(ticker) == expected


class TestNormalizeDownloadDates:
    def test_same_day_range_expands_training_window(self) -> None:
        start, end_exclusive, cache_end = normalize_download_dates("2026-06-12", "2026-06-12")
        assert cache_end == "2026-06-12"
        assert end_exclusive == "2026-06-13"
        assert start < "2026-06-12"

    def test_end_date_is_exclusive_for_yfinance(self) -> None:
        _, end_exclusive, cache_end = normalize_download_dates("2020-01-01", "2026-06-12")
        assert cache_end == "2026-06-12"
        assert end_exclusive == "2026-06-13"

    def test_swapped_dates_are_corrected(self) -> None:
        start, end_exclusive, cache_end = normalize_download_dates("2026-06-12", "2020-01-01")
        assert start < cache_end
        assert end_exclusive > cache_end

    def test_short_window_is_extended(self) -> None:
        start, _, cache_end = normalize_download_dates("2026-01-01", "2026-06-12")
        assert start < "2026-01-01"
        assert cache_end == "2026-06-12"


class TestLoadStockData:
    def test_loads_from_cache_with_sanitized_index_filename(self, tmp_path, sample_ohlcv_df) -> None:
        start, _, cache_end = normalize_download_dates("2020-01-01", "2026-06-12")
        cache_file = tmp_path / f"IDX_BSESN_{start}_{cache_end}.csv"
        sample_ohlcv_df.to_csv(cache_file)

        df = load_stock_data(
            "^BSESN",
            start_date="2020-01-01",
            end_date="2026-06-12",
            data_dir=str(tmp_path),
            use_cache=True,
        )
        assert len(df) >= 100
        assert "Close" in df.columns

    @patch("src.data_preprocessing._download_ticker_data")
    def test_downloads_when_cache_missing(self, mock_download, tmp_path, sample_ohlcv_df) -> None:
        mock_download.return_value = sample_ohlcv_df

        df = load_stock_data(
            "BEL.NS",
            start_date="2020-01-01",
            end_date="2026-06-12",
            data_dir=str(tmp_path),
            use_cache=True,
        )

        assert mock_download.called
        assert len(df) == len(sample_ohlcv_df)
        cache_files = list(tmp_path.glob("BEL.NS_*.csv"))
        assert len(cache_files) == 1

    @patch("src.data_preprocessing._download_ticker_data")
    def test_same_day_user_input_still_downloads(self, mock_download, tmp_path, sample_ohlcv_df) -> None:
        mock_download.return_value = sample_ohlcv_df

        df = load_stock_data(
            "^BSESN",
            start_date="2026-06-12",
            end_date="2026-06-12",
            data_dir=str(tmp_path),
            use_cache=False,
        )

        assert len(df) >= 100
        called_start, called_end = mock_download.call_args[0][1:]
        assert called_start < called_end

    @patch("src.data_preprocessing._download_ticker_data")
    def test_raises_clear_error_when_all_downloads_fail(self, mock_download, tmp_path) -> None:
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data retrieved for ticker symbol '\\^BSESN'"):
            load_stock_data(
                "^BSESN",
                start_date="2026-06-12",
                end_date="2026-06-12",
                data_dir=str(tmp_path),
                use_cache=False,
            )


class TestCleanData:
    def test_clean_valid_data(self, sample_ohlcv_df) -> None:
        cleaned = clean_data(sample_ohlcv_df)
        assert len(cleaned) >= 100
        assert cleaned[["Open", "High", "Low", "Close", "Volume"]].notnull().all().all()

    def test_rejects_insufficient_rows(self, sample_ohlcv_df) -> None:
        tiny = sample_ohlcv_df.iloc[:20]
        with pytest.raises(ValueError, match="Insufficient data after cleaning"):
            clean_data(tiny)

    def test_fills_missing_values(self, sample_ohlcv_df) -> None:
        dirty = sample_ohlcv_df.copy()
        dirty.iloc[10, dirty.columns.get_loc("Close")] = pd.NA
        cleaned = clean_data(dirty)
        assert cleaned["Close"].isnull().sum() == 0
