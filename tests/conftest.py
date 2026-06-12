"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Synthetic OHLCV history large enough for feature engineering."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=len(dates)))
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, len(dates)),
            "High": close + rng.uniform(0.5, 2.0, len(dates)),
            "Low": close - rng.uniform(0.5, 2.0, len(dates)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, len(dates)),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
