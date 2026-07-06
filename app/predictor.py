import os
import json
import joblib
import pandas as pd
import numpy as np
import calendar
from datetime import date
from app.config import MODEL_PATH, TARGET_ENCODER_PATH, FEATURE_ORDER_PATH, STORE_METADATA_PATH, SALES_HISTORY_PATH

class SalesPredictor:
    def __init__(self):
        # 1. Load serialized model assets
        print("Loading serialized model...")
        self.model = joblib.load(MODEL_PATH)
        
        print("Loading target encoder...")
        with open(TARGET_ENCODER_PATH, 'r') as f:
            encoder_data = json.load(f)
            self.store_means = encoder_data["store_means"]
            self.global_mean = encoder_data["global_mean"]
            
        print("Loading feature order...")
        with open(FEATURE_ORDER_PATH, 'r') as f:
            self.feature_order = json.load(f)
            
        # 2. Load and index store metadata for fast O(1) metadata mapping
        print("Loading store metadata...")
        store_df = pd.read_csv(STORE_METADATA_PATH)
        
        # Normalize and fill missing values matching preprocessing.py
        store_df['CompetitionDistance'] = store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median())
        store_df['CompetitionOpenSinceMonth'] = store_df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
        store_df['CompetitionOpenSinceYear'] = store_df['CompetitionOpenSinceYear'].fillna(0).astype(int)
        store_df['Promo2SinceWeek'] = store_df['Promo2SinceWeek'].fillna(0).astype(int)
        store_df['Promo2SinceYear'] = store_df['Promo2SinceYear'].fillna(0).astype(int)
        store_df['PromoInterval'] = store_df['PromoInterval'].fillna('None')
        
        # Map categories
        store_type_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        assortment_map = {'a': 0, 'b': 1, 'c': 2}
        store_df['StoreType'] = store_df['StoreType'].map(store_type_map)
        store_df['Assortment'] = store_df['Assortment'].map(assortment_map)
        
        self.store_metadata = store_df.set_index('Store').to_dict(orient='index')
        
        # 3. Load lightweight sales history into memory for on-the-fly lag/rolling features
        print("Loading historical sales database...")
        # Since train.csv has 1M rows, we optimize by loading only the columns we need,
        # filtering to active days, and keeping only the latest ~40 days to save memory and lookup time.
        train_df = pd.read_csv(SALES_HISTORY_PATH, usecols=['Store', 'Date', 'Sales', 'Open', 'Promo'], low_memory=False)
        train_df = train_df[(train_df['Open'] == 1) & (train_df['Sales'] > 0)].copy()
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        
        # Keep latest historical window (validation period starts 2015-06-15, we keep from 2015-05-15)
        # to ensure sufficient historical sequence for 30-day lag and 14-day rolling windows.
        history_start_date = pd.to_datetime('2015-05-15')
        train_df = train_df[train_df['Date'] >= history_start_date].copy()
        
        # Convert date to string format to speed up lookups
        train_df['Date_str'] = train_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Store as a dictionary mapping Store ID to its historical list of records
        self.sales_history = {}
        for store_id, group in train_df.groupby('Store'):
            # Sort chronologically
            sorted_group = group.sort_values('Date')
            self.sales_history[int(store_id)] = sorted_group[['Date_str', 'Sales', 'Promo']].to_dict(orient='records')
            
        print("ML Service Predictor initialization completed!")

    def predict(self, request_data) -> dict:
        """
        Executes leak-free on-the-fly feature engineering and performs model inference.
        """
        store = request_data.store
        query_date = request_data.date
        promo = request_data.promo
        state_holiday = request_data.state_holiday
        school_holiday = request_data.school_holiday
        
        query_date_str = query_date.strftime('%Y-%m-%d')
        
        # 1. Fetch sales history for this store
        history = self.sales_history.get(store, [])
        
        # Construct temporary dataframe representing store history
        df_store = pd.DataFrame(history)
        
        # Append the query date row (with NaN Sales since it is in the future relative to history)
        query_row = pd.DataFrame([{
            'Date_str': query_date_str,
            'Sales': np.nan,
            'Promo': promo
        }])
        df_store = pd.concat([df_store, query_row], ignore_index=True)
        
        # Parse dates and sort
        df_store['Date'] = pd.to_datetime(df_store['Date_str'])
        df_store = df_store.sort_values('Date').reset_index(drop=True)
        
        # 2. Extract dynamic Store Average Sales (leak-free training encoding)
        store_avg = self.store_means.get(str(store), self.global_mean)
        df_store['Store_AvgSales'] = store_avg
        
        # 3. Compute time-series lag features on-the-fly
        # Lags
        df_store['Sales_Lag_1'] = df_store['Sales'].shift(1)
        df_store['Sales_Lag_7'] = df_store['Sales'].shift(7)
        df_store['Sales_Lag_14'] = df_store['Sales'].shift(14)
        df_store['Sales_Lag_30'] = df_store['Sales'].shift(30)
        
        # Indicators
        df_store['HasLag1'] = df_store['Sales_Lag_1'].notnull().astype(int)
        df_store['HasLag7'] = df_store['Sales_Lag_7'].notnull().astype(int)
        
        # Rolling Mean & Std (shift(1) avoids current day target leakage)
        df_store['Sales_RollingMean_7'] = df_store['Sales'].shift(1).rolling(window=7).mean()
        df_store['Sales_RollingStd_7'] = df_store['Sales'].shift(1).rolling(window=7).std()
        df_store['Sales_RollingMean_14'] = df_store['Sales'].shift(1).rolling(window=14).mean()
        
        # Rolling indicator
        df_store['HasRolling7'] = df_store['Sales_RollingMean_7'].notnull().astype(int)
        
        # Impute cold-starts using Store_AvgSales
        df_store['Sales_Lag_1'] = df_store['Sales_Lag_1'].fillna(store_avg)
        df_store['Sales_Lag_7'] = df_store['Sales_Lag_7'].fillna(store_avg)
        df_store['Sales_Lag_14'] = df_store['Sales_Lag_14'].fillna(store_avg)
        df_store['Sales_Lag_30'] = df_store['Sales_Lag_30'].fillna(store_avg)
        df_store['Sales_RollingMean_7'] = df_store['Sales_RollingMean_7'].fillna(store_avg)
        df_store['Sales_RollingStd_7'] = df_store['Sales_RollingStd_7'].fillna(0.0)
        df_store['Sales_RollingMean_14'] = df_store['Sales_RollingMean_14'].fillna(store_avg)
        
        # 4. Map Store metadata
        meta = self.store_metadata.get(store, {
            'StoreType': 0, 'Assortment': 0, 'CompetitionDistance': 2300.0,
            'Promo2': 0, 'PromoInterval': 'None'
        })
        
        df_store['StoreType'] = meta['StoreType']
        df_store['Assortment'] = meta['Assortment']
        df_store['CompetitionDistance'] = meta['CompetitionDistance']
        
        # State Holiday mapping
        state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3, '1': 1, '2': 2, '3': 3}
        df_store['StateHoliday'] = state_holiday_map.get(str(state_holiday), 0)
        
        # School holiday
        df_store['SchoolHoliday'] = int(school_holiday)
        
        # Holiday union
        df_store['IsHoliday'] = (df_store['StateHoliday'] > 0) | (df_store['SchoolHoliday'] == 1)
        df_store['IsHoliday'] = df_store['IsHoliday'].astype(int)
        
        # 5. Business / Promo features
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                     7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        month_abbr = month_map[query_date.month]
        
        # Promo2 Month check
        promo2_active_month = 0
        if meta['Promo2'] == 1 and meta['PromoInterval'] != 'None':
            if month_abbr in meta['PromoInterval']:
                promo2_active_month = 1
                
        df_store['IsPromotionRunning'] = ((df_store['Promo'] == 1) | (promo2_active_month == 1)).astype(int)
        
        # PromoDuration (Consecutive active Promo days)
        promo_diff = (df_store['Promo'] != df_store['Promo'].shift(1))
        promo_diff.iloc[0] = True
        df_store['PromoBlock'] = promo_diff.cumsum()
        df_store['PromoDuration'] = df_store.groupby('PromoBlock').cumcount() + 1
        df_store.loc[df_store['Promo'] == 0, 'PromoDuration'] = 0
        
        # 6. Cyclical Date features
        day_of_week = query_date.weekday() + 1
        month = query_date.month
        day = query_date.day
        week_of_year = query_date.isocalendar()[1]
        
        df_store['DayOfWeek'] = day_of_week
        df_store['Month'] = month
        df_store['Day'] = day
        df_store['WeekOfYear'] = week_of_year
        df_store['IsWeekend'] = 1 if day_of_week in [6, 7] else 0
        df_store['IsMonthStart'] = 1 if day == 1 else 0
        
        last_day = calendar.monthrange(query_date.year, month)[1]
        df_store['IsMonthEnd'] = 1 if day == last_day else 0
        
        df_store['DayOfWeek_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df_store['DayOfWeek_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        df_store['Month_sin'] = np.sin(2 * np.pi * month / 12)
        df_store['Month_cos'] = np.cos(2 * np.pi * month / 12)
        df_store['Day_sin'] = np.sin(2 * np.pi * day / 31)
        df_store['Day_cos'] = np.cos(2 * np.pi * day / 31)
        df_store['WeekOfYear_sin'] = np.sin(2 * np.pi * week_of_year / 52)
        df_store['WeekOfYear_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        
        # 7. Extract the query row
        query_df = df_store[df_store['Date_str'] == query_date_str].copy()
        
        # Align columns to exact feature order trained
        X_infer = query_df[self.feature_order]
        
        # 8. Model inference (log1p prediction -> inverse expm1 -> clip)
        pred_log = self.model.predict(X_infer)[0]
        predicted_sales = float(np.expm1(pred_log))
        predicted_sales = max(0.0, round(predicted_sales, 2))
        
        # 9. Business Recommendation Layer
        # Demand Level: Compare predicted sales to historical store average
        if predicted_sales > store_avg * 1.2:
            demand_level = "High"
            inventory_recommendation = "High demand expected: Increase stock levels by 15-20% immediately."
        elif predicted_sales < store_avg * 0.8:
            demand_level = "Low"
            inventory_recommendation = "Low demand expected: Reduce fresh stock delivery to prevent inventory waste."
        else:
            demand_level = "Normal"
            inventory_recommendation = "Standard demand expected: Maintain typical baseline stock levels."
            
        # Confidence: High if short-term standard deviation is below 25% of average sales, else Medium
        rolling_std_7_val = query_df['Sales_RollingStd_7'].values[0]
        if pd.isna(rolling_std_7_val) or rolling_std_7_val < store_avg * 0.25:
            confidence = "High"
        else:
            confidence = "Medium"
            
        store_type_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}.get(meta.get('StoreType', 0), 'A')
        comp_dist_val = float(meta.get('CompetitionDistance', 2300.0))
        
        return {
            "predicted_sales": predicted_sales,
            "confidence": confidence,
            "demand_level": demand_level,
            "inventory_recommendation": inventory_recommendation,
            "store_avg": float(store_avg),
            "competition_distance": comp_dist_val,
            "store_type": store_type_char
        }
