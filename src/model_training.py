import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger("StockMarketTrendPrediction")

# Try to import XGBoost, handle gracefully if not available
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
    logger.info("XGBoost is available for model training.")
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost could not be imported. It will be excluded from the training comparison.")

def split_time_series_data(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str, 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits stock market time-series data chronologically into train and test sets
    to avoid data leakage (look-ahead bias).

    Args:
        df (pd.DataFrame): Dataframe containing features and target.
        feature_cols (List[str]): List of column names to use as features.
        target_col (str): Column name of target variable.
        test_size (float): Proportion of data to include in test split. Defaults to 0.2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
    """
    logger.info("Splitting dataset chronologically: train = %d%%, test = %d%%", int((1-test_size)*100), int(test_size*100))
    
    split_idx = int(len(df) * (1 - test_size))
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    logger.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    models_dir: str = "models"
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits a StandardScaler on the training data and transforms both train and test data.
    Saves the fitted scaler for future inference.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        models_dir (str): Directory where the scaler will be saved. Defaults to 'models'.

    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: Scaled X_train, scaled X_test, and scaler.
    """
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    # Fit on training set only to prevent leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler (gracefully skip if directory/file is read-only)
    try:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info("Fitted scaler saved successfully to %s", scaler_path)
    except OSError as e:
        logger.warning("Failed to save scaler to directory %s (proceeding in-memory): %s", models_dir, e)
    
    return X_train_scaled, X_test_scaled, scaler

def get_models() -> Dict[str, Any]:
    """
    Defines the dictionary of models to be trained and evaluated.

    Returns:
        Dict[str, Any]: Dictionary mapping model names to model instances.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    if XGB_AVAILABLE:
        models["XGBoost Regressor"] = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
        
    return models

def train_and_compare_models(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: pd.Series, 
    y_test: pd.Series,
    models: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Trains each model, makes predictions, and calculates preliminary RMSE to compare performance.

    Args:
        X_train (np.ndarray): Scaled training features.
        X_test (np.ndarray): Scaled testing features.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Testing target values.
        models (Dict[str, Any]): Dict of models to train.

    Returns:
        Dict[str, Dict[str, Any]]: Detailed training results containing models and predictions.
    """
    logger.info("Starting training of %d models...", len(models))
    results = {}
    
    for name, model in models.items():
        logger.info("Training %s...", name)
        try:
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            # Evaluate basic metric for comparison
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            
            logger.info("%s - Train RMSE: %.4f | Test RMSE: %.4f", name, train_rmse, test_rmse)
            
            results[name] = {
                "model": model,
                "train_preds": train_preds,
                "test_preds": test_preds,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse
            }
        except Exception as e:
            logger.error("Failed to train model %s: %s", name, e)
            
    return results

def select_and_save_best_model(
    results: Dict[str, Dict[str, Any]], 
    models_dir: str = "models"
) -> Tuple[str, Any]:
    """
    Identifies the best model based on the lowest test RMSE, logs the selection,
    and saves the model object to disk.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of model training results.
        models_dir (str): Directory where best model will be saved. Defaults to 'models'.

    Returns:
        Tuple[str, Any]: Name of the best model and the best model instance.
    """
    best_model_name = None
    min_rmse = float("inf")
    best_model_data = None
    
    for name, data in results.items():
        if data["test_rmse"] < min_rmse:
            min_rmse = data["test_rmse"]
            best_model_name = name
            best_model_data = data
            
    if not best_model_name:
        raise RuntimeError("No models were successfully trained. Check the logs for errors.")
        
    logger.info(">>> Best model selected: %s with Test RMSE of %.4f <<<", best_model_name, min_rmse)
    
    # Save the model and metadata (gracefully skip if directory/file is read-only)
    try:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        best_model_path = os.path.join(models_dir, "best_model.joblib")
        joblib.dump(best_model_data["model"], best_model_path)
        logger.info("Saved best model '%s' successfully to %s", best_model_name, best_model_path)
        
        # Save metadata info about which model was chosen
        meta_path = os.path.join(models_dir, "model_meta.txt")
        with open(meta_path, "w") as f:
            f.write(f"Best Model Name: {best_model_name}\n")
            f.write(f"Test RMSE: {min_rmse:.6f}\n")
    except OSError as e:
        logger.warning("Failed to save best model and metadata to directory %s (proceeding in-memory): %s", models_dir, e)
        
    return best_model_name, best_model_data["model"]
