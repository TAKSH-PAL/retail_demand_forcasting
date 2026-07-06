import time
import argparse
import sys
import os
from xgboost import XGBRegressor

# Append source directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_data
from evaluate import evaluate_predictions, print_metrics

def train_and_evaluate(stage: int = 0):
    """
    Runs the end-to-end training and evaluation pipeline for a given stage.
    
    Parameters:
    stage (int): Feature engineering stage (0 = Baseline, 1 = Date/Holiday Features).
    """
    train_path = "data/raw/train.csv"
    store_path = "data/raw/store.csv"
    
    print(f"\n==========================================")
    print(f"Starting Training for Stage {stage}")
    print(f"==========================================")
    
    # 1. Prepare data
    X_train, y_train, X_val, y_val = prepare_data(train_path, store_path, stage=stage)
    
    # 2. Define model
    # Use tree_method='hist' for extremely fast training on 1M rows.
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
    # 3. Train model
    start_time = time.time()
    print("Training XGBoost Regressor (on log1p target)...")
    model.fit(X_train, np.log1p(y_train))
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    # 4. Predict
    print("Generating validation predictions (expm1 transformed)...")
    y_pred_val = np.expm1(model.predict(X_val))
    # Clip negative predictions to zero (sales cannot be negative)
    y_pred_val = np.clip(y_pred_val, 0, None)
    
    # 5. Evaluate
    metrics = evaluate_predictions(y_val, y_pred_val)
    print_metrics(metrics, title=f"Stage {stage} Validation Metrics")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    
    return metrics, elapsed_time

if __name__ == "__main__":
    import numpy as np # import here to prevent any missing ref in scope
    
    parser = argparse.ArgumentParser(description="Train and evaluate retail demand forecasting model.")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Stage to run (0 = Baseline, 1 = Date & Holiday, 2 = Business, 3 = Time-Series, 4 = Advanced)")
    args = parser.parse_args()
    
    train_and_evaluate(args.stage)
