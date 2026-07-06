import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Root Mean Squared Percentage Error (RMSPE).
    Filters out actual zeros to prevent division by zero.
    
    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    
    Returns:
    float: RMSPE value in percent (e.g., 12.34 means 12.34%).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out actual values that are zero
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    percentage_error = (y_true_filtered - y_pred_filtered) / y_true_filtered
    rmspe = np.sqrt(np.mean(percentage_error ** 2)) * 100
    
    return rmspe

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates key evaluation metrics: MAE, RMSE, R2, and RMSPE.
    
    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    
    Returns:
    dict: Dictionary containing the calculated metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rmspe = calculate_rmspe(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'RMSPE': rmspe
    }
    
    return metrics

def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """
    Utility to print metrics cleanly.
    """
    print(f"=== {title} ===")
    print(f"  RMSE:      {metrics['RMSE']:.2f}")
    print(f"  RMSPE (%): {metrics['RMSPE']:.2f}%")
    print(f"  MAE:       {metrics['MAE']:.2f}")
    print(f"  R2 Score:  {metrics['R2']:.4f}")
    print("=" * (len(title) + 8))
