import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("StockMarketTrendPrediction")

def evaluate_models(
    y_test: pd.Series, 
    results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Calculates evaluation metrics (RMSE, MAE, R2 Score, MAPE) for all trained models.

    Args:
        y_test (pd.Series): Actual test target values.
        results (Dict[str, Dict[str, Any]]): Dictionary containing model names and predictions.

    Returns:
        pd.DataFrame: A comparison dataframe of all model performance metrics.
    """
    logger.info("Evaluating all model predictions...")
    metrics_list = []
    
    for name, data in results.items():
        test_preds = data["test_preds"]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
        
        logger.info("%s Metrics - RMSE: %.4f | MAE: %.4f | R²: %.4f | MAPE: %.2f%%", name, rmse, mae, r2, mape)
        
        metrics_list.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2 Score": r2,
            "MAPE (%)": mape
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    # Sort by RMSE ascending
    metrics_df.sort_values("RMSE", inplace=True)
    return metrics_df
