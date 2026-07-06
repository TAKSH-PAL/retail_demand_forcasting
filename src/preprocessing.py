import pandas as pd
import numpy as np
import sys
import os

# Append source directory to path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import extract_date_features, extract_holiday_features, extract_business_features, extract_time_series_features, extract_advanced_features

def prepare_data(train_path: str, store_path: str, stage: int = 0):
    """
    Loads, cleans, processes, and splits the retail forecasting dataset.
    Includes Target Encoding for Store ID to stabilize tree splits and prevent scale leakage.
    
    Parameters:
    train_path (str): Path to train.csv.
    store_path (str): Path to store.csv.
    stage (int): Modeling stage (0 = Baseline, 1 = Date & Holiday, 2 = Business).
    
    Returns:
    tuple: (X_train, y_train, X_val, y_val)
    """
    # Load data
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path)
    
    # Pre-filter: drop closed days and days with zero sales as discussed in EDA
    train = train[(train['Open'] == 1) & (train['Sales'] > 0)].copy()
    
    # Merge with store metadata
    df = pd.merge(train, store, on='Store', how='left')
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Store and Date to keep chronological records aligned
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    # Chronological Split Boundary
    split_date = pd.to_datetime('2015-06-15')
    train_mask = df['Date'] < split_date
    val_mask = df['Date'] >= split_date
    
    # 1. Target Encoding for Store ID (computed strictly on training split to avoid leakage)
    store_means = df.loc[train_mask].groupby('Store')['Sales'].mean()
    df['Store_AvgSales'] = df['Store'].map(store_means)
    # Fill any missing values with global train mean
    global_train_mean = df.loc[train_mask, 'Sales'].mean()
    df['Store_AvgSales'] = df['Store_AvgSales'].fillna(global_train_mean)
    
    # Stage-specific feature engineering first (so they see the raw string columns)
    if stage >= 1:
        # Extract calendar and holiday features
        df = extract_date_features(df, date_col='Date')
        df = extract_holiday_features(df, state_col='StateHoliday', school_col='SchoolHoliday')
    if stage >= 2:
        # Extract store, competition, and promotion tenure features
        df = extract_business_features(df)
    if stage >= 3:
        # Extract time-series lag and rolling features
        df = extract_time_series_features(df)
    if stage >= 4:
        # Extract cyclical encodings
        df = extract_advanced_features(df)

    # Baseline Preprocessing (Common to all stages, run after feature engineering)
    # Fill missing values
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0).astype(int)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0).astype(int)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    
    # Normalize StateHoliday to standard string representation
    df['StateHoliday'] = df['StateHoliday'].astype(str).str.strip().replace({'0.0': '0', '0': '0'})
    
    # Categorical encoding (mapping to integers)
    store_type_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    assortment_map = {'a': 0, 'b': 1, 'c': 2}
    state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3, '1': 1, '2': 2, '3': 3}
    promo_interval_map = {'None': 0, 'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3}
    
    df['StoreType'] = df['StoreType'].map(store_type_map)
    df['Assortment'] = df['Assortment'].map(assortment_map)
    df['StateHoliday'] = df['StateHoliday'].map(state_holiday_map).fillna(0).astype(int)
    df['PromoInterval'] = df['PromoInterval'].map(promo_interval_map)
    
    # Features to exclude from X
    # Note: Customers is dropped because it is NOT available in the test set.
    # Open is dropped because we only train on open days (Open == 1).
    # Store ID is dropped because it is replaced by the stationary Store_AvgSales target encoding.
    # Non-stationary open days columns (CompetitionOpenDays, Promo2OpenDays) are excluded 
    # to maintain stationarity across stages.
    exclude_cols = [
        'Sales', 'Customers', 'Open', 'Date', 'Store',
        # Non-stationary / raw absolute temporal features
        'Year', 'Quarter', 'CompetitionOpenDays', 'Promo2OpenDays',
        # Redundant Holiday columns (retaining StateHoliday category and IsHoliday flag)
        'SchoolHoliday', 'IsSchoolHoliday', 'IsStateHoliday',
        # Redundant Competition and Promotion parameters
        'Promo2', 'PromoInterval',
        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
        'Promo2SinceWeek', 'Promo2SinceYear'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, 'Sales'].copy()
    
    X_val = df.loc[val_mask, feature_cols].copy()
    y_val = df.loc[val_mask, 'Sales'].copy()
    
    print(f"Stage {stage} Prep Complete:")
    print(f"  Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val set:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Features used: {list(X_train.columns)}")
    
    return X_train, y_train, X_val, y_val
