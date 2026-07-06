import pandas as pd
import numpy as np
import time
import sys
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Append source directory to path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_data
from evaluate import evaluate_predictions, print_metrics

def main():
    train_path = "data/raw/train.csv"
    store_path = "data/raw/store.csv"
    
    print("\n==========================================")
    print("Starting Model Comparison (Stage 4 Features)")
    print("==========================================")
    
    # 1. Prepare Stage 4 data
    X_train, y_train, X_val, y_val = prepare_data(train_path, store_path, stage=4)
    
    # Apply target log-transformation
    y_train_log = np.log1p(y_train)
    
    # 2. Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            thread_count=-1,
            verbose=False
        )
    }
    
    results = []
    
    # 3. Train and Evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        model.fit(X_train, y_train_log)
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds.")
        
        # Predict & Inverse Transform
        print(f"Generating predictions for {name}...")
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.clip(y_pred, 0, None)
        
        # Evaluate
        metrics = evaluate_predictions(y_val, y_pred)
        print_metrics(metrics, title=f"{name} Validation")
        
        results.append({
            "Model": name,
            "RMSE": metrics["RMSE"],
            "RMSPE": metrics["RMSPE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "Train Time (s)": round(elapsed_time, 2)
        })
        
    # 4. Print Summary Table
    df_results = pd.DataFrame(results)
    print("\n==========================================")
    print("Comparison Summary Table:")
    print("==========================================")
    print(df_results.to_string(index=False))
    
if __name__ == "__main__":
    main()
