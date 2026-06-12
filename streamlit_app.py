"""
Stock Market Trend Prediction System — Premium Web Dashboard
=============================================================
A Streamlit-based interactive UI with dark glassmorphism aesthetics,
interactive Plotly charts, model performance leaderboard, and
next-day price forecast display.

Run with:
    streamlit run streamlit_app.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Pipeline Imports (reuse existing project modules)
# ---------------------------------------------------------------------------
from src.utils import setup_logging, ensure_directories
from src.data_preprocessing import load_stock_data, clean_data, normalize_download_dates
from src.ticker_config import (
    US_STOCKS,
    INDIAN_INDICES,
    INDIAN_STOCKS,
    resolve_indian_ticker,
    MIN_TRAINING_DAYS,
)
from src.feature_engineering import (
    create_features_and_target,
    add_moving_averages,
    add_rsi,
    add_macd,
    add_returns_and_volatility,
)
from src.model_training import (
    split_time_series_data,
    scale_features,
    get_models,
    train_and_compare_models,
    select_and_save_best_model,
)
from src.evaluation import evaluate_models

# ---------------------------------------------------------------------------
# Page Config — must be the FIRST Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Market Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium Custom CSS — dark glassmorphism + neon accents
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ---- Root variables ---- */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --border-glass: rgba(99, 102, 241, 0.15);
    --accent-primary: #6366f1;
    --accent-secondary: #8b5cf6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-cyan: #06b6d4;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --glow-primary: 0 0 20px rgba(99, 102, 241, 0.3);
    --glow-green: 0 0 20px rgba(16, 185, 129, 0.3);
}

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 50%) !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown label {
    color: var(--text-secondary) !important;
}

/* ---- Headers ---- */
h1, h2, h3 {
    color: var(--text-primary) !important;
}

p, span, label, .stMarkdown {
    color: var(--text-secondary) !important;
}

/* ---- Metric Cards ---- */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: var(--glow-primary) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.5) !important;
}

[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6) !important;
}

/* ---- DataFrames / Tables ---- */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
    color: white !important;
    border: none !important;
}

/* ---- Select boxes / Inputs ---- */
[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] > div > div,
.stTextInput > div > div > input {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 10px !important;
}

/* ---- Divider ---- */
hr {
    border-color: var(--border-glass) !important;
}

/* ---- Spinner ---- */
.stSpinner > div {
    border-top-color: var(--accent-primary) !important;
}

/* ---- Glass Card helper ---- */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--glow-primary);
}

.glass-card h3 {
    margin-top: 0;
    font-size: 1.1rem;
    color: var(--text-primary);
}

/* ---- Hero Section ---- */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 4px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-bottom: 24px;
}

/* ---- Forecast Card ---- */
.forecast-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.10));
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: var(--glow-green);
}

.forecast-price {
    font-size: 3rem;
    font-weight: 800;
    color: #10b981;
    margin: 8px 0;
}

.forecast-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* ---- Best Model Badge ---- */
.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent-green), #059669);
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly Template — consistent dark theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.5)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    title_font=dict(size=18, color="#f1f5f9"),
    legend=dict(
        bgcolor="rgba(17,24,39,0.6)",
        bordercolor="rgba(99,102,241,0.15)",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
    margin=dict(l=40, r=40, t=60, b=40),
)

ACCENT_COLORS = ["#6366f1", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#10b981"]

# ---------------------------------------------------------------------------
# Feature columns (must match main.py)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_14", "SMA_50", "EMA_14", "EMA_50",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "Daily_Return", "Volatility",
]
TARGET_COL = "Target_Close"

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
LOGS_DIR = "logs"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Control Panel")
    st.markdown("---")

    market = st.radio(
        "Market",
        options=["🇺🇸 US Stocks", "🇮🇳 Indian Market / Stocks"],
        index=1,
        help="Indian section supports NIFTY/SENSEX indices and any NSE/BSE stock.",
    )

    display_name = ""

    if market == "🇺🇸 US Stocks":
        stock_label = st.selectbox("Select Stock", options=list(US_STOCKS.keys()), index=0)
        ticker = US_STOCKS[stock_label]
        display_name = stock_label
    else:
        india_mode = st.radio(
            "Predict For",
            options=["📊 Indian Market Index", "🏢 Indian Stock"],
            horizontal=True,
            help="Choose a whole market index (NIFTY/SENSEX) or a single Indian company stock.",
        )

        if india_mode == "📊 Indian Market Index":
            index_label = st.selectbox(
                "Select Market Index",
                options=list(INDIAN_INDICES.keys()),
                index=0,
            )
            ticker = INDIAN_INDICES[index_label]
            display_name = index_label
            st.info(f"Predicting trend for **{index_label}** Indian market index.")

        else:
            stock_input_mode = st.radio(
                "How to pick stock?",
                options=["📋 Pick from List", "✏️ Enter Stock Name"],
                horizontal=True,
            )

            if stock_input_mode == "📋 Pick from List":
                stock_label = st.selectbox(
                    "Select Indian Stock",
                    options=list(INDIAN_STOCKS.keys()),
                    index=0,
                )
                ticker = INDIAN_STOCKS[stock_label]
                display_name = stock_label
            else:
                st.markdown("**Enter any Indian stock symbol**")
                custom_symbol = st.text_input(
                    "Stock Symbol",
                    placeholder="e.g. BEL, TCS, ADANI, INFY, SBIN",
                    help="Type only the symbol — .NS or .BO is added automatically.",
                )
                exchange = st.selectbox("Exchange", options=["NSE", "BSE"], index=0)

                ticker = resolve_indian_ticker(custom_symbol, exchange)
                display_name = custom_symbol.strip().upper() if custom_symbol else ""

                if ticker:
                    st.success(f"Yahoo Finance ticker: **{ticker}**")
                elif custom_symbol:
                    st.warning("Please enter a valid stock symbol (e.g. BEL, ADANIENT).")

    st.markdown("")
    today = datetime.date.today()
    max_forecast_date = today + datetime.timedelta(days=365)

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input(
            "Training Start Date",
            value=datetime.date(2020, 1, 1),
            min_value=datetime.date(2010, 1, 1),
            max_value=today,
            help="Historical data from this date is used to train the ML models.",
        )
    with col_e:
        end_date = st.date_input(
            "Forecast Target Date",
            value=today + datetime.timedelta(days=30),
            min_value=datetime.date(2010, 1, 1),
            max_value=max_forecast_date,
            help="Pick a future date (e.g. 20 Jun 2026) to see AI price predictions up to that day.",
        )

    if end_date < start_date:
        st.warning("Forecast target date should be on or after the training start date.")

    training_days = (min(today, end_date) - start_date).days
    if training_days < MIN_TRAINING_DAYS:
        st.info(
            f"Training window is short ({training_days} days). "
            f"The system will auto-expand to at least {MIN_TRAINING_DAYS} days of history."
        )

    st.markdown("")
    run_pipeline_btn = st.button("🚀  Run Prediction Pipeline", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:0.75rem; color:#64748b;'>"
        "Built with Streamlit · Plotly · Scikit-Learn · XGBoost"
        "</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="hero-title">Stock Market Trend Predictor</p>'
    '<p class="hero-subtitle">AI-powered future closing price forecasts — select a target date ahead to predict stock trends using multiple ML models & technical indicators</p>',
    unsafe_allow_html=True,
)

def compute_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators without shifting/dropping target row."""
    feat_df = df.copy()
    feat_df = add_moving_averages(feat_df, windows=[14, 50])
    feat_df = add_rsi(feat_df, window=14)
    feat_df = add_macd(feat_df, fast_span=12, slow_span=26, signal_span=9)
    feat_df = add_returns_and_volatility(feat_df, window=14)
    return feat_df

def generate_future_forecast(cleaned_df: pd.DataFrame, best_model, scaler, feature_cols, target_date_str: str):
    """
    Recursively forecasts stock closing prices from the day after the last historical date
    up to the target_date_str (business days only).
    """
    last_historical_date = cleaned_df.index[-1]
    target_date = pd.to_datetime(target_date_str)
    
    # Check if target date is in the future
    if target_date <= last_historical_date:
        return cleaned_df.copy(), []
        
    # Generate business days range
    future_dates = pd.bdate_range(start=last_historical_date + pd.Timedelta(days=1), end=target_date)
    if len(future_dates) == 0:
        return cleaned_df.copy(), []
        
    df_temp = cleaned_df.copy()
    future_predictions = []
    
    for current_date in future_dates:
        # Compute features for df_temp
        feat_df = compute_features_only(df_temp)
        
        # We need the last row's features (which correspond to the last date in df_temp)
        last_row_features = feat_df[feature_cols].iloc[[-1]]
        
        # Scale
        last_row_scaled = scaler.transform(last_row_features)
        
        # Predict Close
        pred_close = float(best_model.predict(last_row_scaled)[0])
        
        # Estimate Open, High, Low, Volume
        prev_close = df_temp["Close"].iloc[-1]
        pred_open = prev_close
        pred_high = max(pred_open, pred_close)
        pred_low = min(pred_open, pred_close)
        pred_volume = df_temp["Volume"].iloc[-14:].mean() if len(df_temp) >= 14 else df_temp["Volume"].mean()
        
        # Append as new row
        new_row = pd.DataFrame({
            "Open": [pred_open],
            "High": [pred_high],
            "Low": [pred_low],
            "Close": [pred_close],
            "Volume": [pred_volume]
        }, index=[current_date])
        new_row.index.name = "Date"
        
        df_temp = pd.concat([df_temp, new_row])
        
        future_predictions.append({
            "Date": current_date.strftime("%Y-%m-%d"),
            "Predicted_Close": pred_close
        })
        
    return df_temp, future_predictions

def run_full_pipeline(ticker_sym: str, start: str, end: str):
    """Run the entire ML pipeline and return all artefacts needed by the UI."""
    ensure_directories([DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR])

    # Historical data: cap at today; future target dates are handled by generate_future_forecast
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    fetch_end = min(end, today_str)
    normalize_download_dates(start, fetch_end)  # validates and logs auto-expansion rules

    # 1 — Data
    raw_df = load_stock_data(
        ticker=ticker_sym,
        start_date=start,
        end_date=fetch_end,
        data_dir=DATA_DIR,
        use_cache=True,
    )
    cleaned_df = clean_data(raw_df)

    # 2 — Features
    engineered_df = create_features_and_target(cleaned_df)

    # 3 — Split & Scale
    X_train, X_test, y_train, y_test = split_time_series_data(
        df=engineered_df, feature_cols=FEATURE_COLS, target_col=TARGET_COL, test_size=0.2
    )
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, models_dir=MODELS_DIR)

    # 4 — Train
    models = get_models()
    results = train_and_compare_models(X_train_sc, X_test_sc, y_train, y_test, models)

    # 5 — Evaluate
    metrics_df = evaluate_models(y_test=y_test, results=results)
    metrics_path = os.path.join(MODELS_DIR, "comparison_metrics.csv")
    try:
        metrics_df.to_csv(metrics_path, index=False)
    except OSError as e:
        # Import system for warning fallback print
        import sys
        sys.stderr.write(f"WARNING: Failed to save comparison metrics CSV: {e}\n")

    # 6 — Best model
    best_name, best_model = select_and_save_best_model(results=results, models_dir=MODELS_DIR)

    # 7 — Next-day forecast using last row of engineered data
    last_row = engineered_df[FEATURE_COLS].iloc[[-1]]
    last_row_sc = scaler.transform(last_row)
    forecast_price = float(best_model.predict(last_row_sc)[0])

    # 8 — Future recursive predictions up to end
    future_df, future_preds = generate_future_forecast(
        cleaned_df=cleaned_df,
        best_model=best_model,
        scaler=scaler,
        feature_cols=FEATURE_COLS,
        target_date_str=end
    )

    return {
        "engineered_df": engineered_df,
        "y_test": y_test,
        "results": results,
        "metrics_df": metrics_df,
        "best_name": best_name,
        "forecast_price": forecast_price,
        "last_close": float(engineered_df["Close"].iloc[-1]),
        "future_preds": future_preds,
        "future_df": future_df,
    }


# Session state
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = None

if run_pipeline_btn:
    if not ticker:
        st.error("❌  Please enter or select a valid stock ticker.")
    elif end_date < start_date:
        st.error("❌  Forecast target date must be on or after the training start date.")
    else:
        with st.spinner("⏳  Running ML pipeline… downloading data, engineering features, training models…"):
            try:
                data = run_full_pipeline(ticker, str(start_date), str(end_date))
                st.session_state.pipeline_data = data
                st.session_state.pipeline_ticker = display_name or ticker
                st.session_state.pipeline_symbol = ticker
                future_count = len(data.get("future_preds", []))
                if future_count > 0:
                    st.success(
                        f"✅  Pipeline completed! Best model: **{data['best_name']}** — "
                        f"forecast generated for **{future_count}** future trading day(s) up to **{end_date}**."
                    )
                else:
                    st.success(f"✅  Pipeline completed! Best model: **{data['best_name']}**")
            except Exception as exc:
                st.error(f"❌  Pipeline failed: {exc}")

# ---------------------------------------------------------------------------
# Dashboard (only show when data exists)
# ---------------------------------------------------------------------------
pipeline = st.session_state.pipeline_data

if pipeline is None:
    # Show a nice placeholder
    st.markdown("")
    st.markdown(
        '<div class="glass-card" style="text-align:center; padding:60px 40px;">'
        '<p style="font-size:3rem; margin-bottom:8px;">🚀</p>'
        '<h3 style="color:#f1f5f9; margin-bottom:8px;">Ready to Predict</h3>'
        '<p style="color:#64748b;">Choose <strong>Indian Market / Stocks</strong> for NIFTY/SENSEX or stocks like BEL, ADANI — '
        'set a future <strong>Forecast Target Date</strong>, then click '
        '<strong style="color:#6366f1;">Run Prediction Pipeline</strong>.</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Unpack
eng_df = pipeline["engineered_df"]
y_test = pipeline["y_test"]
results = pipeline["results"]
metrics_df = pipeline["metrics_df"]
best_name = pipeline["best_name"]
forecast_price = pipeline["forecast_price"]
last_close = pipeline["last_close"]
active_ticker = st.session_state.get("pipeline_ticker", "AAPL")

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
st.markdown("---")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

best_metrics = metrics_df[metrics_df["Model"] == best_name].iloc[0]
price_delta = forecast_price - last_close
pct_delta = (price_delta / last_close) * 100

with kpi1:
    st.metric("Best Model", best_name)
with kpi2:
    st.metric("Test RMSE", f"${best_metrics['RMSE']:.2f}")
with kpi3:
    st.metric("R² Score", f"{best_metrics['R2 Score']:.4f}")
with kpi4:
    st.metric("MAPE", f"{best_metrics['MAPE (%)']:.2f}%")

# ---------------------------------------------------------------------------
# Forecast Card
# ---------------------------------------------------------------------------
st.markdown("")
fc1, fc2 = st.columns([1, 2])

with fc1:
    direction = "📈" if price_delta >= 0 else "📉"
    delta_color = "#10b981" if price_delta >= 0 else "#ef4444"
    st.markdown(
        f'<div class="forecast-card">'
        f'<p class="forecast-label">Next Trading Day Forecast — {active_ticker}</p>'
        f'<p class="forecast-price">${forecast_price:,.2f}</p>'
        f'<p style="font-size:1.1rem; color:{delta_color}; font-weight:700;">'
        f'{direction} {"+" if price_delta >= 0 else ""}{price_delta:,.2f} ({pct_delta:+.2f}%)</p>'
        f'<p style="font-size:0.8rem; color:#64748b; margin-top:8px;">'
        f'Last Close: ${last_close:,.2f} · Model: {best_name}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    future_preds = pipeline.get("future_preds", [])
    if future_preds:
        target_forecast = future_preds[-1]["Predicted_Close"]
        target_date_label = future_preds[-1]["Date"]
        target_price_delta = target_forecast - last_close
        target_pct_delta = (target_price_delta / last_close) * 100
        target_direction = "📈" if target_price_delta >= 0 else "📉"
        target_delta_color = "#10b981" if target_price_delta >= 0 else "#ef4444"

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="forecast-card" style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.12), rgba(139, 92, 246, 0.10)); border-color: rgba(6, 182, 212, 0.3); box-shadow: 0 0 20px rgba(6, 182, 212, 0.3);">'
            f'<p class="forecast-label">Target Date Forecast ({target_date_label})</p>'
            f'<p class="forecast-price" style="color: #06b6d4;">${target_forecast:,.2f}</p>'
            f'<p style="font-size:1.1rem; color:{target_delta_color}; font-weight:700;">'
            f'{target_direction} {"+" if target_price_delta >= 0 else ""}{target_price_delta:,.2f} ({target_pct_delta:+.2f}%)</p>'
            f'<p style="font-size:0.8rem; color:#64748b; margin-top:8px;">'
            f'Cumulative trend prediction over {len(future_preds)} business days</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:0.75rem; color:#ef4444; margin-top:8px; text-align:center; font-style:italic; font-weight:500;">'
            "⚠️ Disclaimer: Recursive forecasts accumulate prediction error over time."
            "</p>",
            unsafe_allow_html=True,
        )

with fc2:
    # Mini sparkline of recent close prices
    recent = eng_df["Close"].iloc[-60:]
    fig_spark = go.Figure()
    fig_spark.add_trace(go.Scatter(
        x=recent.index, y=recent.values,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#6366f1", width=2),
        fillcolor="rgba(99,102,241,0.1)",
        hovertemplate="Date: %{x}<br>Close: $%{y:,.2f}<extra></extra>",
    ))
    fig_spark.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{active_ticker} — Last 60 Trading Days",
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_spark, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabs — Charts
# ---------------------------------------------------------------------------
st.markdown("---")

tab_trend, tab_pred, tab_corr, tab_imp, tab_table = st.tabs([
    "📊 Stock Trend", "🎯 Actual vs Predicted", "🔥 Correlation Heatmap",
    "🏆 Feature Importance", "📋 Metrics Table",
])

# ---- Tab 1: Stock Trend ----
with tab_trend:
    plot_df = eng_df.copy() if len(eng_df) <= 250 else eng_df.iloc[-250:]
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["Close"],
        name="Historical Close", line=dict(color="#6366f1", width=2.5),
        hovertemplate="$%{y:,.2f}<extra>Close</extra>",
    ))
    fig_trend.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["SMA_14"],
        name="SMA 14", line=dict(color="#f59e0b", width=1.5, dash="dash"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["SMA_50"],
        name="SMA 50", line=dict(color="#10b981", width=1.5, dash="dot"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["EMA_14"],
        name="EMA 14", line=dict(color="#8b5cf6", width=1.5, dash="dashdot"),
    ))

    # Add Future Forecast trace if available
    future_preds = pipeline.get("future_preds", [])
    if future_preds:
        future_dates = [plot_df.index[-1]] + [pd.to_datetime(p["Date"]) for p in future_preds]
        future_prices = [plot_df["Close"].iloc[-1]] + [p["Predicted_Close"] for p in future_preds]
        fig_trend.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            name="Future Forecast", line=dict(color="#06b6d4", width=2.5, dash="dash"),
            hovertemplate="$%{y:,.2f}<extra>Forecast</extra>",
        ))

    fig_trend.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{active_ticker} Closing Price & Moving Averages (with Future Forecast)",
        yaxis_title="Price ($)",
        height=500,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ---- Tab 2: Actual vs Predicted ----
with tab_pred:
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test.index, y=y_test.values,
        name="Actual", line=dict(color="#f1f5f9", width=2.5),
        hovertemplate="$%{y:,.2f}<extra>Actual</extra>",
    ))
    for i, (name, data) in enumerate(results.items()):
        fig_pred.add_trace(go.Scatter(
            x=y_test.index, y=data["test_preds"],
            name=name, line=dict(color=ACCENT_COLORS[i % len(ACCENT_COLORS)], width=1.5, dash="dash"),
            hovertemplate="$%{y:,.2f}<extra>" + name + "</extra>",
        ))
    fig_pred.update_layout(
        **PLOTLY_LAYOUT,
        title="Model Comparison: Actual vs Predicted Next-Day Close",
        yaxis_title="Price ($)",
        height=500,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# ---- Tab 3: Correlation Heatmap ----
with tab_corr:
    cols_corr = FEATURE_COLS + [TARGET_COL]
    corr = eng_df[cols_corr].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig_corr.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Correlation Matrix",
        height=650,
    )
    fig_corr.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_corr, use_container_width=True)

# ---- Tab 4: Feature Importance ----
with tab_imp:
    tree_model = None
    tree_name = None
    for mname in ["Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor"]:
        if mname in results and hasattr(results[mname]["model"], "feature_importances_"):
            tree_model = results[mname]["model"]
            tree_name = mname
            break

    if tree_model is not None:
        imp_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": tree_model.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker=dict(
                color=imp_df["Importance"],
                colorscale=[[0, "#6366f1"], [0.5, "#8b5cf6"], [1, "#06b6d4"]],
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Feature Importance — {tree_name}",
            xaxis_title="Importance Score",
            height=500,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No tree-based model available for feature importance display.")

# ---- Tab 5: Metrics Table ----
with tab_table:
    st.markdown(
        f'<p><span class="best-badge">🏆 Best Model: {best_name}</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Style the dataframe
    def highlight_best(row):
        is_best = row["Model"] == best_name
        return ["background-color: rgba(99,102,241,0.15); font-weight: 700;" if is_best else "" for _ in row]

    styled = (
        metrics_df.style
        .apply(highlight_best, axis=1)
        .format({
            "RMSE": "${:.4f}",
            "MAE": "${:.4f}",
            "R2 Score": "{:.6f}",
            "MAPE (%)": "{:.4f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar chart comparing RMSE
    fig_bar = go.Figure()
    colors_bar = ["#10b981" if m == best_name else "#6366f1" for m in metrics_df["Model"]]
    fig_bar.add_trace(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["RMSE"],
        marker_color=colors_bar,
        text=metrics_df["RMSE"].round(2),
        textposition="auto",
        hovertemplate="%{x}<br>RMSE: $%{y:.4f}<extra></extra>",
    ))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        title="Model RMSE Comparison (lower is better)",
        yaxis_title="RMSE ($)",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#64748b; font-size:0.8rem;">'
    "📈 Stock Market Trend Prediction System · Powered by Scikit-Learn, XGBoost & Plotly · "
    f"Data from Yahoo Finance · Last run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</p>",
    unsafe_allow_html=True,
)
