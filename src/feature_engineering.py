import pandas as pd
import numpy as np

def extract_date_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Extracts date-based features from a datetime column.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    date_col (str): Name of the date column.
    
    Returns:
    pd.DataFrame: Dataframe with added date features.
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    # Ensure DayOfWeek matches the existing column if present (1 = Monday, 7 = Sunday)
    df['DayOfWeek'] = df[date_col].dt.dayofweek + 1
    df['WeekOfYear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['Quarter'] = df[date_col].dt.quarter
    
    # Binary flags (0 or 1) for weekend and month/quarter markers
    df['IsWeekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    df['IsMonthStart'] = df[date_col].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df[date_col].dt.is_month_end.astype(int)
    
    return df

def extract_holiday_features(df: pd.DataFrame, state_col: str = 'StateHoliday', school_col: str = 'SchoolHoliday') -> pd.DataFrame:
    """
    Extracts holiday-based features by normalizing StateHoliday and SchoolHoliday,
    and creating a consolidated IsHoliday flag.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    state_col (str): Column name for StateHoliday.
    school_col (str): Column name for SchoolHoliday.
    
    Returns:
    pd.DataFrame: Dataframe with added holiday features.
    """
    df = df.copy()
    
    if state_col in df.columns:
        # Check if already numeric/pre-mapped
        if pd.api.types.is_numeric_dtype(df[state_col]):
            # If mapped (0 = none, 1 = a, 2 = b, 3 = c), anything > 0 is a holiday
            df['IsStateHoliday'] = (df[state_col] > 0).astype(int)
            df[state_col] = df[state_col].astype(int)
        else:
            # It is string/object. Normalize first.
            df[state_col] = df[state_col].astype(str).str.strip().replace({'0.0': '0', '0': '0'})
            df['IsStateHoliday'] = df[state_col].isin(['a', 'b', 'c']).astype(int)
            state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
            df[state_col] = df[state_col].map(state_holiday_map).fillna(0).astype(int)
    else:
        df['IsStateHoliday'] = 0
        
    if school_col in df.columns:
        df['IsSchoolHoliday'] = df[school_col].astype(int)
    else:
        df['IsSchoolHoliday'] = 0
        
    # Combined indicator: 1 if either is a holiday
    df['IsHoliday'] = ((df['IsStateHoliday'] == 1) | (df['IsSchoolHoliday'] == 1)).astype(int)
    
    return df

def extract_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts store, competition, and promotional tenure/duration features.
    
    Parameters:
    df (pd.DataFrame): Input dataframe (must be merged with store.csv metadata).
    
    Returns:
    pd.DataFrame: Dataframe with added business features.
    """
    df = df.copy()
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
        
    # Sort by Store and Date for duration group-bys
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    # 1. Competition Tenure (CompetitionOpenDays)
    # Filter valid competitor opening dates
    competitor_open = (df['CompetitionOpenSinceYear'] > 0) & (df['CompetitionOpenSinceMonth'] > 0)
    df['CompetitionOpenDays'] = 0
    if competitor_open.any():
        comp_date_str = (
            df.loc[competitor_open, 'CompetitionOpenSinceYear'].astype(int).astype(str) + '-' +
            df.loc[competitor_open, 'CompetitionOpenSinceMonth'].astype(int).astype(str).str.zfill(2) + '-15'
        )
        comp_dates = pd.to_datetime(comp_date_str, format='%Y-%m-%d', errors='coerce')
        df.loc[competitor_open, 'CompetitionOpenDays'] = (df.loc[competitor_open, 'Date'] - comp_dates).dt.days
        # Clip negative values (competition opens in the future) to 0
        df.loc[df['CompetitionOpenDays'] < 0, 'CompetitionOpenDays'] = 0
        
    # 2. Promo2 Tenure (Promo2OpenDays)
    promo2_active = (df['Promo2'] == 1) & (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0)
    df['Promo2OpenDays'] = 0
    if promo2_active.any():
        # ISO week date format: %Y-W%W-%w (where %w is day of week 1-7)
        promo2_date_str = (
            df.loc[promo2_active, 'Promo2SinceYear'].astype(int).astype(str) + '-W' +
            df.loc[promo2_active, 'Promo2SinceWeek'].astype(int).astype(str).str.zfill(2) + '-1'
        )
        promo2_dates = pd.to_datetime(promo2_date_str, format='%Y-W%W-%w', errors='coerce')
        df.loc[promo2_active, 'Promo2OpenDays'] = (df.loc[promo2_active, 'Date'] - promo2_dates).dt.days
        # Clip negative values to 0
        df.loc[df['Promo2OpenDays'] < 0, 'Promo2OpenDays'] = 0
        
    # 3. Dynamic Promo2 month indicator and IsPromotionRunning
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['Promo2ActiveMonth'] = 0
    for month_int, month_abbr in month_map.items():
        mask = (df['Date'].dt.month == month_int) & (df['Promo2'] == 1) & (df['PromoInterval'].str.contains(month_abbr, na=False))
        df.loc[mask, 'Promo2ActiveMonth'] = 1
        
    df['IsPromotionRunning'] = ((df['Promo'] == 1) | (df['Promo2ActiveMonth'] == 1)).astype(int)
    
    # 4. Short-term Promo Duration (Consecutive Promo Days)
    promo_diff = (df['Promo'] != df['Promo'].shift(1)) | (df['Store'] != df['Store'].shift(1))
    df['PromoBlock'] = promo_diff.cumsum()
    df['PromoDuration'] = df.groupby(['Store', 'PromoBlock']).cumcount() + 1
    # Only keep duration count on days when Promo is actually active (1)
    df.loc[df['Promo'] == 0, 'PromoDuration'] = 0
    
    # Clean up temporary columns
    df = df.drop(columns=['Promo2ActiveMonth', 'PromoBlock'])
    
    return df

def extract_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-series lag and rolling statistics per store.
    Uses shift(1) to avoid leaking current day's Sales target into features.
    Includes indicator columns for early NaN cold-starts.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with added time-series features.
    """
    df = df.copy()
    
    # Sort by Store and Date to maintain sequence
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    # 1. Lag features
    df['Sales_Lag_1'] = df.groupby('Store')['Sales'].shift(1)
    df['Sales_Lag_7'] = df.groupby('Store')['Sales'].shift(7)
    df['Sales_Lag_14'] = df.groupby('Store')['Sales'].shift(14)
    df['Sales_Lag_30'] = df.groupby('Store')['Sales'].shift(30)
    
    # 2. Missing indicators (before imputation)
    df['HasLag1'] = df['Sales_Lag_1'].notnull().astype(int)
    df['HasLag7'] = df['Sales_Lag_7'].notnull().astype(int)
    
    # 3. Rolling features (shift(1) ensures no target leakage of current day)
    # Exclude min_periods so that early sequence starts naturally output NaN
    df['Sales_RollingMean_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(window=7).mean())
    df['Sales_RollingStd_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(window=7).std())
    df['Sales_RollingMean_14'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(window=14).mean())
    
    # Missing indicator for rolling mean
    df['HasRolling7'] = df['Sales_RollingMean_7'].notnull().astype(int)
    
    # 4. Handle early NaN cold-starts using Store_AvgSales (imputed from leak-free training encoding)
    fill_col = 'Store_AvgSales' if 'Store_AvgSales' in df.columns else 0
    
    df['Sales_Lag_1'] = df['Sales_Lag_1'].fillna(df[fill_col])
    df['Sales_Lag_7'] = df['Sales_Lag_7'].fillna(df[fill_col])
    df['Sales_Lag_14'] = df['Sales_Lag_14'].fillna(df[fill_col])
    df['Sales_Lag_30'] = df['Sales_Lag_30'].fillna(df[fill_col])
    
    df['Sales_RollingMean_7'] = df['Sales_RollingMean_7'].fillna(df[fill_col])
    df['Sales_RollingStd_7'] = df['Sales_RollingStd_7'].fillna(0) # Standard deviation of early periods filled with 0
    df['Sales_RollingMean_14'] = df['Sales_RollingMean_14'].fillna(df[fill_col])
    
    return df

def extract_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts cyclical temporal encodings (sine and cosine components)
    for DayOfWeek, Month, Day, and WeekOfYear.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with cyclical encodings added.
    """
    df = df.copy()
    
    # 1. DayOfWeek (period = 7)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # 2. Month (period = 12)
    month_col = df['Date'].dt.month if 'Month' not in df.columns else df['Month']
    df['Month_sin'] = np.sin(2 * np.pi * month_col / 12)
    df['Month_cos'] = np.cos(2 * np.pi * month_col / 12)
    
    # 3. Day of month (period = 31)
    day_col = df['Date'].dt.day if 'Day' not in df.columns else df['Day']
    df['Day_sin'] = np.sin(2 * np.pi * day_col / 31)
    df['Day_cos'] = np.cos(2 * np.pi * day_col / 31)
    
    # 4. WeekOfYear (period = 52)
    week_col = df['Date'].dt.isocalendar().week if 'WeekOfYear' not in df.columns else df['WeekOfYear']
    df['WeekOfYear_sin'] = np.sin(2 * np.pi * week_col / 52)
    df['WeekOfYear_cos'] = np.cos(2 * np.pi * week_col / 52)
    
    return df


