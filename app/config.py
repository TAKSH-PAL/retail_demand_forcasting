import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model & Encoder paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "models", "target_encoder.json")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "models", "feature_order.json")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# Data paths for feature engineering on-the-fly
STORE_METADATA_PATH = os.path.join(BASE_DIR, "data", "raw", "store.csv")
SALES_HISTORY_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
