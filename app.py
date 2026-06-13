import os
import sys
import logging
from datetime import datetime
import pandas as pd
import gradio as gr

# Ensure the root directory is in the python path for importing src modules cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run_pipeline

# Configure a logger for app.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StockMarketTrendPredictionApp")

def predict_stock_trend(ticker: str, start_date: str, end_date: str):
    """
    Validates user inputs, runs the ML trend prediction pipeline,
    and retrieves the resulting metrics and plots for the Gradio UI.
    """
    # Clean inputs
    ticker = ticker.strip().upper()
    start_date = start_date.strip()
    end_date = end_date.strip()
    
    # 1. Validation
    if not ticker:
        return (
            "⚠️ Error: Stock ticker symbol cannot be empty.",
            None,
            None,
            None,
            None,
            None
        )
        
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return (
            "⚠️ Error: Dates must be in YYYY-MM-DD format (e.g., 2023-01-01).",
            None,
            None,
            None,
            None,
            None
        )
        
    if start_dt >= end_dt:
        return (
            "⚠️ Error: Start Date must be chronologically before End Date.",
            None,
            None,
            None,
            None,
            None
        )
        
    # 2. Execution
    try:
        logger.info(f"Running pipeline from Gradio Web UI for Ticker: {ticker}, Range: {start_date} to {end_date}")
        run_pipeline(ticker=ticker, start_date=start_date, end_date=end_date)
        
        # Load comparison metrics
        metrics_df = None
        metrics_path = os.path.join("models", "comparison_metrics.csv")
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            # Format decimals for better readability in UI
            for col in ["RMSE", "MAE", "R2 Score"]:
                if col in metrics_df.columns:
                    metrics_df[col] = metrics_df[col].round(4)
            if "MAPE (%)" in metrics_df.columns:
                metrics_df["MAPE (%)"] = metrics_df["MAPE (%)"].round(2).astype(str) + "%"
            
        # Read best model metadata
        best_model_info = "Linear Regression" # Default fallback
        meta_path = os.path.join("models", "model_meta.txt")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                lines = f.readlines()
                best_model_info = " | ".join([line.strip() for line in lines if line.strip()])
        elif metrics_df is not None and not metrics_df.empty:
            # Fallback if metadata text is missing
            best_model_info = f"Best Model Name: {metrics_df.iloc[0]['Model']} | Test RMSE: {metrics_df.iloc[0]['RMSE']}"

        status_msg = (
            f"✅ Success! ML Pipeline completed successfully for ticker '{ticker}'.\n"
            f"🎯 {best_model_info}\n"
            f"📅 Range: {start_date} to {end_date}"
        )
        
        # Verify plots exist before returning them
        close_trend_path = os.path.join("plots", "stock_close_trend.png")
        corr_heatmap_path = os.path.join("plots", "correlation_heatmap.png")
        actual_vs_pred_path = os.path.join("plots", "actual_vs_predicted.png")
        feat_importance_path = os.path.join("plots", "feature_importances.png")
        
        close_trend = close_trend_path if os.path.exists(close_trend_path) else None
        corr_heatmap = corr_heatmap_path if os.path.exists(corr_heatmap_path) else None
        actual_vs_pred = actual_vs_pred_path if os.path.exists(actual_vs_pred_path) else None
        feat_importance = feat_importance_path if os.path.exists(feat_importance_path) else None
        
        return (
            status_msg,
            metrics_df,
            actual_vs_pred,
            close_trend,
            feat_importance,
            corr_heatmap
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Gradio Web UI execution exception: {str(e)}")
        user_error_msg = (
            f"❌ Pipeline Execution Failed!\n\n"
            f"Error Details: {str(e)}\n\n"
            f"Please ensure the stock ticker symbol '{ticker}' is valid and historical data is available via yfinance for the requested date range."
        )
        return user_error_msg, None, None, None, None, None

# Define custom premium styling/theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

# Custom CSS for modern dashboard styling
custom_css = """
.title-header {
    text-align: center;
    background: linear-gradient(135deg, #1e3a8a 0%, #10b981 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.title-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 8px;
    color: white !important;
}
.title-header p {
    font-size: 1.1rem;
    opacity: 0.9;
}
.run-btn {
    font-weight: bold !important;
    background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
}
.run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4) !important;
}
"""

with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.HTML(
        """
        <div class="title-header">
            <h1>📈 Stock Market Trend Prediction System</h1>
            <p>An End-to-End Machine Learning Pipeline comparing Linear Regression, Random Forest, Gradient Boosting, and XGBoost to predict next-day closing prices.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Pipeline Configuration")
            ticker_input = gr.Textbox(
                label="Stock Ticker Symbol",
                value="AAPL",
                placeholder="e.g., AAPL, MSFT, GOOGL, TSLA",
                max_lines=1
            )
            start_date_input = gr.Textbox(
                label="Start Date (YYYY-MM-DD)",
                value="2023-01-01",
                placeholder="YYYY-MM-DD",
                max_lines=1
            )
            end_date_input = gr.Textbox(
                label="End Date (YYYY-MM-DD)",
                value="2024-12-31",
                placeholder="YYYY-MM-DD",
                max_lines=1
            )
            run_btn = gr.Button("🚀 Run Prediction Pipeline", elem_classes=["run-btn"])
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Execution Status & Best Model Summary")
            status_box = gr.Textbox(
                label="System Output / Logs",
                value="Gradio interface initialized. Enter configuration and click 'Run Prediction Pipeline' to begin.",
                interactive=False,
                lines=5
            )
            
    with gr.Tabs():
        with gr.Tab("🏆 Model Performance & Forecasts"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📈 Model Comparison Table")
                    metrics_table = gr.Dataframe(
                        headers=["Model", "RMSE", "MAE", "R2 Score", "MAPE (%)"],
                        interactive=False,
                        wrap=True
                    )
                with gr.Column():
                    gr.Markdown("#### 📉 Actual vs Predicted Close Price")
                    plot_actual_vs_pred = gr.Image(
                        label="Actual vs Predicted Line Chart",
                        type="filepath",
                        show_download_button=True
                    )
                    
        with gr.Tab("💹 Stock Close Trend & Indicators"):
            gr.Markdown("#### 🔍 Historical Trend & Moving Averages Overlay")
            plot_close_trend = gr.Image(
                label="Moving Averages Trend Plot",
                type="filepath",
                show_download_button=True
            )
            
        with gr.Tab("🧠 Feature Importance & Correlation"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 Feature Importance (Best Tree Model)")
                    plot_feat_importance = gr.Image(
                        label="Feature Importance Scores Chart",
                        type="filepath",
                        show_download_button=True
                    )
                with gr.Column():
                    gr.Markdown("#### 🌡️ Feature Correlation Matrix")
                    plot_corr_heatmap = gr.Image(
                        label="Correlation Heatmap",
                        type="filepath",
                        show_download_button=True
                    )

    # Wire up the execution trigger
    run_btn.click(
        fn=predict_stock_trend,
        inputs=[ticker_input, start_date_input, end_date_input],
        outputs=[
            status_box,
            metrics_table,
            plot_actual_vs_pred,
            plot_close_trend,
            plot_feat_importance,
            plot_corr_heatmap
        ]
    )

if __name__ == "__main__":
    # Hugging Face Spaces automatically listens on port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
