# Feature Engineering Roadmap & Documentation

This document serves as a comprehensive reference for the features designed and implemented for the Retail Demand Forecasting & Inventory Optimization model. Features are organized into categories representing business domains, temporal behaviors, and signal smoothing techniques.

---

## Current Status: Stage 2 (Date and Holiday Features Active)
Currently, **Date-based Features** and **Holiday Features** (IsHoliday) are active. Subsequent modeling stages will incrementally introduce additional feature categories.

---

## Feature Importance and Description Table

| Feature Name | Category | Logic / Formula | Business Reason | Expected Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Year** | Date Features | `df['Date'].dt.year` | Captures long-term annual growth trends. | Medium |
| **Month** | Date Features | `df['Date'].dt.month` | Captures monthly seasonality patterns (e.g. December spikes). | High |
| **Day** | Date Features | `df['Date'].dt.day` | Captures intra-month behaviors (e.g., paydays). | Medium |
| **DayOfWeek** | Date Features | `df['Date'].dt.dayofweek + 1` | Captures weekly sales patterns (e.g., Sunday drop-offs). | High |
| **WeekOfYear** | Date Features | `df['Date'].dt.isocalendar().week` | Captures micro-seasonality and holiday alignment week-by-week. | High |
| **Quarter** | Date Features | `df['Date'].dt.quarter` | Captures quarterly sales cycles and promotions. | Low |
| **IsWeekend** | Date Features | `df['Date'].dt.dayofweek.isin([5, 6])` | Identifies weekends where consumer shopping behavior changes. | Medium |
| **IsMonthStart** | Date Features | `df['Date'].dt.is_month_start` | Captures salary-day shopping spikes. | Low |
| **IsMonthEnd** | Date Features | `df['Date'].dt.is_month_end` | Captures end-of-month promotional campaigns and shopping surges. | Low |
| **StoreType_Encoded** | Store Features | One-Hot / Label encoding of `StoreType` | Captures differences in core business/store models. | High |
| **Assortment_Encoded** | Store Features | One-Hot / Label encoding of `Assortment` | Captures impact of product selection complexity. | High |
| **CompetitionOpenDays** | Competition | Days since competitor opened | Competitor existence tenure influences market share erosion. | High |
| **IsPromotionRunning** | Promotion | `Promo` or `Promo2` indicator | Combines short-term and long-term promotion signals. | High |
| **PromoDuration** | Promotion | Tenure of active promotional campaign | Long promos affect buyer urgency compared to new ones. | High |
| **AverageBasketSize** | Customer Features | `Sales / Customers` | Measures average revenue generated per customer visit. | Medium |
| **Lag_1** | Lag Features | Sales of previous day | Captures immediate temporal autocorrelation. | Very High |
| **Lag_7** | Lag Features | Sales of same weekday last week | Captures weekly cyclical patterns. | Very High |
| **Lag_14** | Lag Features | Sales of same weekday two weeks ago | Captures longer term weekday correlation. | High |
| **Lag_30** | Lag Features | Sales of 30 days ago | Captures monthly autocorrelation. | High |
| **RollingMean_7** | Rolling Features | 7-day moving average of Sales | Smooths out daily noise to highlight weekly trend. | Very High |
| **RollingMean_14** | Rolling Features | 14-day moving average of Sales | Smooths short-term fluctuations. | High |
| **RollingMean_30** | Rolling Features | 30-day moving average of Sales | Identifies baseline sales trend. | High |
| **RollingStd_7** | Rolling Features | 7-day standard deviation of Sales | Captures local variance/volatility in demand. | Medium |
| **RollingMax_7** | Rolling Features | 7-day maximum Sales | Identifies recent peak demand capacity. | Medium |
| **RollingMin_7** | Rolling Features | 7-day minimum Sales | Identifies recent baseline/floor demand. | Low |
| **SalesGrowthRate** | Trend Features | `(Sales_today - Sales_yesterday) / Sales_yesterday` | Measures rapid acceleration or deceleration of sales. | Medium |
| **IsHoliday** | Holiday Features | Union of `StateHoliday` and `SchoolHoliday` | Captures standard and regional shopping closures/surges. | High |
| **MonthSin / MonthCos** | Cyclical Encoding | Sine/Cosine transformation of Month | Retains cyclical proximity (e.g. Dec (12) is close to Jan (1)). | Medium |
| **WeekdaySin / WeekdayCos** | Cyclical Encoding | Sine/Cosine transformation of Weekday | Retains cyclical proximity of days of the week. | Medium |

---

## Detailed Rationale for Key Features

### 1. Date Features
Time indicators allow tree-based models to partition data into structural calendar chunks.
* **IsWeekend**: Consumers show differing habits on weekends due to availability of free time.
* **IsMonthStart/End**: Many consumers receive paychecks on the 1st or last day of a month, causing a predictable lift in retail demand.

### 2. Time-Series (Lag & Rolling) Features
Forecasting relies heavily on historical trends.
* **Lag_7**: Retail operates on weekly loops. Yesterday's performance is important, but last Monday is a better predictor for today (Monday).
* **RollingMean_7**: Provides a noise-filtered view of whether a store's sales are generally increasing or decreasing.
