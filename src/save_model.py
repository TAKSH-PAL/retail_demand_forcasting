import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from datetime import datetime
from xgboost import XGBRegressor

# Append source directory to path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_data
from evaluate import evaluate_predictions

def main():
    train_path = "data/raw/train.csv"
    store_path = "data/raw/store.csv"
    models_dir = "models"
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Prepare Stage 4 data
    print("Preparing Stage 4 data...")
    X_train, y_train, X_val, y_val = prepare_data(train_path, store_path, stage=4)
    
    # Apply log-transformation to target
    y_train_log = np.log1p(y_train)
    
    # 2. Train the winning XGBoost Regressor
    print("Training winning XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    model.fit(X_train, y_train_log)
    
    # Evaluate validation metrics
    print("Evaluating validation metrics...")
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)
    metrics = evaluate_predictions(y_val, y_pred)
    
    # 3. Save serialized model
    model_pkl_path = os.path.join(models_dir, "xgboost_model.pkl")
    print(f"Saving serialized model to {model_pkl_path}...")
    joblib.dump(model, model_pkl_path)
    
    # 4. Save target encoder map (calculated strictly on train mask)
    # We reload train.csv and compute target encoder to map Store -> Store_AvgSales
    train = pd.read_csv(train_path, low_memory=False)
    train_active = train[(train['Open'] == 1) & (train['Sales'] > 0)].copy()
    train_active['Date'] = pd.to_datetime(train_active['Date'])
    
    split_date = pd.to_datetime('2015-06-15')
    train_split = train_active[train_active['Date'] < split_date]
    
    store_means = train_split.groupby('Store')['Sales'].mean()
    # Fill missing stores with global training mean
    global_mean = train_split['Sales'].mean()
    
    target_encoder_dict = {
        "store_means": {int(k): float(v) for k, v in store_means.items()},
        "global_mean": float(global_mean)
    }
    
    target_encoder_json_path = os.path.join(models_dir, "target_encoder.json")
    print(f"Saving target encoder map to {target_encoder_json_path}...")
    with open(target_encoder_json_path, 'w') as f:
        json.dump(target_encoder_dict, f, indent=4)
        
    # 5. Save Feature Order
    feature_order_json_path = os.path.join(models_dir, "feature_order.json")
    feature_order_list = list(X_train.columns)
    print(f"Saving feature order list to {feature_order_json_path}...")
    with open(feature_order_json_path, 'w') as f:
        json.dump(feature_order_list, f, indent=4)
        
    # 6. Save Model Info & Metrics
    metrics_json_path = os.path.join(models_dir, "metrics.json")
    model_info = {
        "name": "XGBoost",
        "rmse": int(round(metrics["RMSE"])),
        "r2": float(round(metrics["R2"], 4)),
        "trained": datetime.now().strftime("%Y-%m-%d")
    }
    print(f"Saving model info to {metrics_json_path}...")
    with open(metrics_json_path, 'w') as f:
        json.dump(model_info, f, indent=4)
        
    print("Serialization completed successfully!")

if __name__ == "__main__":
    main()
