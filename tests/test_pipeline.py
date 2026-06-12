"""Integration tests for the ML pipeline across markets and tickers."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_preprocessing import clean_data, load_stock_data
from src.feature_engineering import create_features_and_target
from src.model_training import (
    get_models,
    scale_features,
    split_time_series_data,
    train_and_compare_models,
)
from src.ticker_config import INDIAN_INDICES, INDIAN_STOCKS, US_STOCKS, all_supported_tickers

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_14", "SMA_50", "EMA_14", "EMA_50",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "Daily_Return", "Volatility",
]
TARGET_COL = "Target_Close"


def _run_ml_core(cleaned_df: pd.DataFrame) -> dict:
    engineered_df = create_features_and_target(cleaned_df)
    X_train, X_test, y_train, y_test = split_time_series_data(
        df=engineered_df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        test_size=0.2,
    )
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, models_dir="models")
    models = get_models()
    results = train_and_compare_models(X_train_sc, X_test_sc, y_train, y_test, models)
    best_name = min(results, key=lambda name: results[name]["test_rmse"])
    last_row = engineered_df[FEATURE_COLS].iloc[[-1]]
    forecast = float(results[best_name]["model"].predict(scaler.transform(last_row))[0])
    return {
        "best_name": best_name,
        "forecast": forecast,
        "rows": len(engineered_df),
        "results": results,
    }


class TestPipelineWithSyntheticData:
    def test_us_stock_pipeline(self, sample_ohlcv_df) -> None:
        result = _run_ml_core(clean_data(sample_ohlcv_df))
        assert result["rows"] > 100
        assert result["forecast"] > 0
        assert result["best_name"] in result["results"]

    def test_indian_index_pipeline(self, sample_ohlcv_df) -> None:
        index_df = sample_ohlcv_df.copy()
        index_df["Close"] = index_df["Close"] * 400
        result = _run_ml_core(clean_data(index_df))
        assert result["forecast"] > 0


@pytest.mark.integration
@pytest.mark.parametrize("ticker", all_supported_tickers())
def test_live_download_all_configured_tickers(ticker: str) -> None:
    """Hit Yahoo Finance for every pre-configured ticker (run with: pytest -m integration)."""
    df = load_stock_data(
        ticker,
        start_date="2020-01-01",
        end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        data_dir="data",
        use_cache=True,
    )
    cleaned = clean_data(df)
    result = _run_ml_core(cleaned)
    assert result["forecast"] > 0


@pytest.mark.integration
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("BEL", "BEL.NS"),
        ("ADANI", "ADANIENT.NS"),
        ("^BSESN", "^BSESN"),
    ],
)
def test_live_same_day_user_mistake_still_works(raw: str, expected: str) -> None:
    """Regression: same-day start/end must not crash for Indian market usage."""
    from src.ticker_config import resolve_indian_ticker

    ticker = expected if raw.startswith("^") else resolve_indian_ticker(raw, "NSE")
    df = load_stock_data(
        ticker,
        start_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        data_dir="data",
        use_cache=True,
    )
    assert len(clean_data(df)) >= 100


class TestPipelineCatalogCoverage:
    def test_every_indian_index_symbol_is_caret_prefixed(self) -> None:
        for name, symbol in INDIAN_INDICES.items():
            assert symbol.startswith("^"), f"{name} should be an index symbol"

    def test_every_indian_stock_has_exchange_suffix(self) -> None:
        for name, symbol in INDIAN_STOCKS.items():
            assert symbol.endswith(".NS"), f"{name} should use NSE suffix"

    def test_us_symbols_have_no_suffix(self) -> None:
        for symbol in US_STOCKS.values():
            assert "." not in symbol

    @patch("src.data_preprocessing._download_ticker_data")
    def test_sensex_same_day_regression_with_mock(self, mock_download, tmp_path, sample_ohlcv_df) -> None:
        mock_download.return_value = sample_ohlcv_df
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        df = load_stock_data("^BSESN", today, today, data_dir=str(tmp_path), use_cache=False)
        result = _run_ml_core(clean_data(df))
        assert result["forecast"] > 0
