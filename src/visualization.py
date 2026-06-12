import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid tkinter thread errors
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

logger = logging.getLogger("StockMarketTrendPrediction")

def set_premium_plot_style() -> None:
    """
    Sets a cohesive, high-quality dark/modern theme for all matplotlib/seaborn plots
    to ensure professional visual aesthetics.
    """
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 200,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.alpha": 0.3,
        "legend.fontsize": 10,
        "legend.frameon": True
    })

def plot_stock_trend(
    df: pd.DataFrame, 
    ticker: str, 
    plots_dir: str = "plots"
) -> None:
    """
    Generates and saves a trend plot showing the stock's closing price
    along with moving averages (SMA_14, SMA_50, EMA_14, EMA_50).

    Args:
        df (pd.DataFrame): Dataframe containing the original and engineered price columns.
        ticker (str): The stock ticker symbol.
        plots_dir (str): Directory where the plot will be saved. Defaults to 'plots'.
    """
    logger.info("Generating stock closing price and indicator trend plot...")
    set_premium_plot_style()
    
    # Corrected the df.suffix("") bug to df.copy() to ensure code safety on datasets <= 250 rows.
    plot_df = df.copy() if len(df) <= 250 else df.iloc[-250:]
    
    plt.figure()
    plt.plot(plot_df.index, plot_df["Close"], label="Close Price", color="#1f77b4", linewidth=2.0)
    plt.plot(plot_df.index, plot_df["SMA_14"], label="SMA 14", color="#ff7f0e", linestyle="--", linewidth=1.2)
    plt.plot(plot_df.index, plot_df["SMA_50"], label="SMA 50", color="#2ca02c", linestyle=":", linewidth=1.5)
    plt.plot(plot_df.index, plot_df["EMA_14"], label="EMA 14", color="#9467bd", linestyle="-.", linewidth=1.2)
    
    plt.title(f"{ticker} Stock Closing Price & Moving Averages (Last 250 Trading Days)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "stock_close_trend.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved stock trend plot to %s", plot_path)

def plot_correlation_heatmap(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str, 
    plots_dir: str = "plots"
) -> None:
    """
    Generates and saves a correlation heatmap for the features and target variable.

    Args:
        df (pd.DataFrame): The engineered DataFrame.
        feature_cols (List[str]): List of feature column names.
        target_col (str): Target column name.
        plots_dir (str): Directory where the plot will be saved. Defaults to 'plots'.
    """
    logger.info("Generating feature correlation heatmap...")
    set_premium_plot_style()
    
    cols_to_corr = feature_cols + [target_col]
    corr_matrix = df[cols_to_corr].corr()
    
    plt.figure(figsize=(10, 8))
    # Custom color palette: clean coolwarm
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    
    plt.title("Correlation Matrix: Features & Target Variable", pad=20)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved correlation heatmap to %s", plot_path)

def plot_actual_vs_predicted(
    y_test: pd.Series, 
    results: Dict[str, Dict[str, Any]], 
    plots_dir: str = "plots"
) -> None:
    """
    Generates and saves a comparison line plot of actual test prices vs. 
    the predicted prices of all trained models.

    Args:
        y_test (pd.Series): Actual test target values.
        results (Dict[str, Dict[str, Any]]): Model training results containing test predictions.
        plots_dir (str): Directory where the plot will be saved. Defaults to 'plots'.
    """
    logger.info("Generating actual vs predicted prices plot...")
    set_premium_plot_style()
    
    plt.figure()
    
    # Plot actual values (we use index directly for chronological ordering)
    plt.plot(y_test.index, y_test.values, label="Actual Next Day Close", color="black", linewidth=2.0, alpha=0.8)
    
    # Plot predicted values for each model
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        plt.plot(y_test.index, data["test_preds"], label=f"{name} Predicted", color=color, linestyle="--", linewidth=1.2)
        
    plt.title("Model Comparison: Actual vs Predicted Next Day Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved actual vs predicted prices plot to %s", plot_path)

def plot_feature_importance(
    results: Dict[str, Dict[str, Any]], 
    feature_cols: List[str], 
    plots_dir: str = "plots"
) -> None:
    """
    Generates and saves a horizontal bar chart showing feature importances 
    from the best available tree-based model (Random Forest, Gradient Boosting, or XGBoost).

    Args:
        results (Dict[str, Dict[str, Any]]): Model training results.
        feature_cols (List[str]): List of feature names corresponding to the model inputs.
        plots_dir (str): Directory where the plot will be saved. Defaults to 'plots'.
    """
    logger.info("Checking for tree-based models to plot feature importance...")
    set_premium_plot_style()
    
    # Find a model with feature_importances_ attribute
    tree_model_name = None
    tree_model = None
    
    # Prefer Random Forest or Gradient Boosting if available
    for name in ["Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor"]:
        if name in results:
            model_obj = results[name]["model"]
            if hasattr(model_obj, "feature_importances_"):
                tree_model_name = name
                tree_model = model_obj
                break
                
    if tree_model is None:
        logger.warning("No tree-based model with feature importances was found. Skipping feature importance plot.")
        return
        
    logger.info("Generating feature importance plot using model: %s", tree_model_name)
    importances = tree_model.feature_importances_
    
    # Create a DataFrame for sorting
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance", 
        y="Feature", 
        data=importance_df, 
        palette="viridis",
        hue="Feature",
        legend=False
    )
    
    plt.title(f"Feature Importance ({tree_model_name})")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "feature_importances.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved feature importance plot to %s", plot_path)
