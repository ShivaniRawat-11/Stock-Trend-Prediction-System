import os
import argparse
from src.utils import setup_logging, ensure_directories
from src.data_preprocessing import load_stock_data, clean_data
from src.feature_engineering import create_features_and_target
from src.model_training import (
    split_time_series_data,
    scale_features,
    get_models,
    train_and_compare_models,
    select_and_save_best_model
)
from src.evaluation import evaluate_models
from src.visualization import (
    plot_stock_trend,
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_feature_importance
)

def run_pipeline(ticker: str, start_date: str, end_date: str) -> None:
    """
    Executes the entire end-to-end Machine Learning pipeline:
    1. Setup directories and logging
    2. Data acquisition & cleaning
    3. Technical indicator calculation & target creation
    4. Time-series chronological split
    5. Feature scaling
    6. Training & comparing multiple ML models
    7. Evaluating metrics (RMSE, MAE, R², MAPE)
    8. Selecting & saving the best model
    9. Generating & saving publication-quality plots

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data download.
        end_date (str): End date for data download.
    """
    # 1. Initialize directories and logging
    log_dir = "logs"
    data_dir = "data"
    plots_dir = "plots"
    models_dir = "models"
    
    ensure_directories([log_dir, data_dir, plots_dir, models_dir])
    logger = setup_logging(log_dir=log_dir, log_filename="system.log")
    
    logger.info("=" * 60)
    logger.info("Starting Stock Market Trend Prediction Pipeline")
    logger.info("Ticker: %s | Start Date: %s | End Date: %s", ticker, start_date, end_date)
    logger.info("=" * 60)
    
    try:
        # 2. Data Acquisition
        raw_df = load_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            use_cache=True
        )
        
        # 3. Data Cleaning
        cleaned_df = clean_data(raw_df)
        
        # 4. Feature Engineering
        engineered_df = create_features_and_target(cleaned_df)
        
        # Define Features & Target columns
        target_col = "Target_Close"
        feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_14", "SMA_50", "EMA_14", "EMA_50",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "Daily_Return", "Volatility"
        ]
        
        # Verify columns exist
        missing_feats = [col for col in feature_cols if col not in engineered_df.columns]
        if missing_feats:
            raise ValueError(f"Feature engineering failed to generate expected columns: {missing_feats}")
            
        # 5. Chronological Split
        X_train, X_test, y_train, y_test = split_time_series_data(
            df=engineered_df,
            feature_cols=feature_cols,
            target_col=target_col,
            test_size=0.2
        )
        
        # 6. Feature Scaling
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train=X_train,
            X_test=X_test,
            models_dir=models_dir
        )
        
        # 7. Model Training & Selection
        models = get_models()
        results = train_and_compare_models(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            models=models
        )
        
        # 8. Model Evaluation & Best Model Choice
        metrics_df = evaluate_models(y_test=y_test, results=results)
        
        # Log and save model metrics comparison table
        logger.info("\n--- Model Evaluation Summary (Sorted by RMSE) ---\n%s\n", metrics_df.to_string(index=False))
        metrics_path = os.path.join(models_dir, "comparison_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Saved evaluation metrics table to %s", metrics_path)
        
        best_name, best_model = select_and_save_best_model(results=results, models_dir=models_dir)
        
        # 9. Visualization plots
        plot_stock_trend(df=engineered_df, ticker=ticker, plots_dir=plots_dir)
        plot_correlation_heatmap(df=engineered_df, feature_cols=feature_cols, target_col=target_col, plots_dir=plots_dir)
        plot_actual_vs_predicted(y_test=y_test, results=results, plots_dir=plots_dir)
        plot_feature_importance(results=results, feature_cols=feature_cols, plots_dir=plots_dir)
        
        logger.info("=" * 60)
        logger.info("Pipeline executed successfully! Best Model: %s", best_name)
        logger.info("All visualization plots saved in: %s/", plots_dir)
        logger.info("Best model and scaler saved in: %s/", models_dir)
        logger.info("=" * 60)
        
    except Exception as e:
        logger.critical("Critical pipeline failure: %s", e, exc_info=True)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Market Trend Prediction Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date YYYY-MM-DD (default: 2025-01-01)")
    
    args = parser.parse_args()
    run_pipeline(ticker=args.ticker, start_date=args.start, end_date=args.end)
