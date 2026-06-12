# 📈 Stock Market Trend Prediction System

A complete, production-quality Machine Learning project that predicts next-day stock closing prices using historical data from Yahoo Finance, technical indicators, and multiple regression models.

---

## Overview

**Problem Statement:** Predicting short-term stock price movements is a fundamental challenge in computational finance. This project builds an end-to-end ML pipeline that ingests historical OHLCV data, engineers domain-specific technical indicators, and trains four regression models to forecast the next trading day's closing price.

**Motivation:** Demonstrate a reproducible, modular ML workflow suitable for portfolio projects, ML internship applications, and Amazon ML School submissions.

---

## Features

- **Automated Data Collection** — Downloads historical stock data from Yahoo Finance via `yfinance` with local CSV caching.
- **Robust Data Cleaning** — Forward/backward fill for missing values, duplicate detection, column validation, and type enforcement.
- **6 Technical Indicators** — SMA, EMA, RSI, MACD (with Signal & Histogram), Daily Returns, and rolling Volatility.
- **Chronological Train/Test Split** — Prevents look-ahead bias (no random shuffling of time-series data).
- **4 ML Models Compared** — Linear Regression, Random Forest, Gradient Boosting, and XGBoost Regressors.
- **Automatic Best Model Selection** — Selects the model with the lowest test RMSE and serializes it with `joblib`.
- **Publication-Quality Visualizations** — Closing price trends, correlation heatmaps, actual vs predicted plots, and feature importance charts.
- **Structured Logging** — Dual-sink logging (console + file) for full pipeline traceability.
- **CLI Interface** — Customizable ticker, start date, and end date via command-line arguments.

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| Language | Python 3.11+ |
| Data | Pandas, NumPy, yfinance |
| ML Models | Scikit-Learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |
| Notebooks | Jupyter |

---

## Project Architecture

```
Stock-Market-Trend-Prediction-System/
│
├── data/                      # Cached stock market CSV datasets
├── logs/                      # Application runtime logs
├── models/                    # Serialized scaler, best model, and metrics
├── notebooks/                 # Jupyter notebook for EDA
├── plots/                     # High-resolution visualization plots
│
├── src/                       # Source modules
│   ├── __init__.py            # Package marker
│   ├── data_preprocessing.py  # Data fetching, cleaning, and validation
│   ├── feature_engineering.py # Technical indicator computation
│   ├── model_training.py      # Train-test split, scaling, training, comparison
│   ├── evaluation.py          # Performance metrics (RMSE, MAE, R², MAPE)
│   ├── visualization.py       # Plot generation (trends, heatmap, predictions)
│   └── utils.py               # Logging setup and directory helpers
│
├── main.py                    # CLI entry point — runs the full pipeline
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions for data, models, plots, logs
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Dataset Source

Historical stock price data is fetched dynamically from the **Yahoo Finance API** using the `yfinance` library.
By default, the pipeline downloads 5 years of daily OHLCV data for **Apple Inc. (AAPL)**. You can specify any other ticker symbol (e.g., `MSFT`, `GOOGL`, `TSLA`) or custom date ranges via command-line arguments.

---

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   git clone https://github.com/yourusername/Stock-Market-Trend-Prediction-System.git
   cd Stock-Market-Trend-Prediction-System
   ```

2. **Create and activate a virtual environment** *(recommended)*:
   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the complete ML pipeline with default settings (AAPL, 2020-01-01 to 2025-01-01):
```bash
python main.py
```

Custom ticker and date range:
```bash
python main.py --ticker MSFT --start 2018-01-01 --end 2024-12-31
```

### What happens when you run `main.py`:
1. Downloads (or loads cached) historical stock data.
2. Cleans and validates the dataset.
3. Computes 6 technical indicators + target variable (next-day close).
4. Splits data chronologically (80% train / 20% test).
5. Scales features using StandardScaler (fitted on train only).
6. Trains 4 regression models and compares performance.
7. Selects and saves the best model to `models/best_model.joblib`.
8. Generates 4 visualization plots in `plots/`.
9. Logs everything to `logs/system.log`.

---

## Results

### Model Comparison (AAPL — Test Set)

| Model | RMSE ($) | MAE ($) | R² Score | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** ✅ | **3.03** | **2.30** | **0.986** | **1.11%** |
| Random Forest Regressor | 27.54 | 20.76 | -0.159 | 9.16% |
| Gradient Boosting Regressor | 27.54 | 20.86 | -0.159 | 9.21% |
| XGBoost Regressor | 32.24 | 24.72 | -0.589 | 10.94% |

**Best Model:** Linear Regression (RMSE = $3.03, R² = 0.986)

### Key ML Insight
Tree-based regressors (Random Forest, Gradient Boosting, XGBoost) **cannot extrapolate** beyond the target value range seen during training. Since AAPL's stock price trended upward into new territory in the test period, tree models systematically underpredicted prices. Linear Regression succeeded because it can project linear trends beyond the training bounds.

---

## Sample Visualizations

All plots are saved in the `plots/` directory at 300 DPI:

| Plot | Description |
| :--- | :--- |
| `stock_close_trend.png` | Closing price with SMA-14, SMA-50, and EMA-14 overlays |
| `correlation_heatmap.png` | Feature correlation matrix (diverging color palette) |
| `actual_vs_predicted.png` | All 4 models' predictions vs actual test prices |
| `feature_importances.png` | Feature importance scores from Random Forest |

---

## Future Improvements

1. **Hyperparameter Tuning** — Integrate Optuna or `GridSearchCV` for automated hyperparameter optimization.
2. **Sentiment Analysis** — Incorporate financial news sentiment scores as additional features.
3. **Deep Learning** — Add LSTM/GRU models (PyTorch) to capture non-linear sequential dependencies.
4. **API Deployment** — Serve predictions via a FastAPI REST endpoint using the serialized model.
5. **Predict Returns Instead of Price** — Train tree models on daily returns to avoid the extrapolation limitation.

---

## Author

Built as a portfolio project for ML internship applications and the Amazon ML School.

---

## License

This project is licensed under the [MIT License](LICENSE).
