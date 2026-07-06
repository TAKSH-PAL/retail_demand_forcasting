# Experiment Log

This document records the performance of various machine learning model stages to evaluate the impact of feature engineering and design decisions on prediction accuracy.

---

## Model Tracking Table

| Stage | Description | Features Added | RMSE | RMSPE (%) | MAE | R² | Train Time | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 0** | Baseline | Raw features + Store Target Encoding | **1205.63** | **16.67%** | **832.88** | **0.8477** | **0.84s** | Replaced nominal Store ID with Store_AvgSales target encoding |
| **Stage 1** | Calendar & Holidays | Stage 0 + Date features + IsWeekend + IsHoliday | **1173.73** | **16.38%** | **822.80** | **0.8556** | **1.46s** | Excludes non-stationary Year and Quarter features |
| **Stage 2** | Business Features | Stage 1 + StoreType/Assortment + Competition + Promo | **1178.76** | **16.33%** | **824.19** | **0.8544** | **1.62s** | Excludes non-stationary Competitor/Promo Open Days; retains stationary PromoDuration |
| **Stage 3** | Time-Series | Stage 2 + Lag (1, 7, 14, 30) + Rolling Mean/Std (7, 14) + indicators | **952.43** | **13.21%** | **650.33** | **0.9049** | **2.03s** | Added missingness indicators (HasLag1, HasLag7, HasRolling7) and rolling std |
| **Stage 4** | Advanced | Stage 3 + Cyclical Encoding (Sin/Cos of time features) | **949.87** | **13.24%** | **649.07** | **0.9055** | **2.20s** | Cyclical encoding improves RMSE/MAE; R² reaches **0.9055** |

---

## Detailed Experiment Observations

### Stage 0: Baseline Performance
* **Date Conducted**: 2026-07-06
* **Model**: XGBoost Regressor
* **Key Observations**: 
  * Replaced the raw nominal `Store` ID with a leak-free target encoding (`Store_AvgSales`) which resolved tree split inefficiencies.
  * Log-scale transformation (`np.log1p(Sales)`) successfully aligned the model's MSE objective with the target validation RMSPE.

### Stage 1: Calendar & Holidays Performance
* **Date Conducted**: 2026-07-06
* **Model**: XGBoost Regressor
* **Key Observations**:
  * With absolute date boundaries (`Year`, `Quarter`) removed and Store target encoded, Stage 1 achieved validation gains. RMSE improved to **1173.73** and RMSPE dropped to **16.38%** (R² = `0.8556`).

### Stage 2: Business Features Performance
* **Date Conducted**: 2026-07-06
* **Model**: XGBoost Regressor
* **Key Observations**:
  * Excluded the strictly-increasing, non-stationary `CompetitionOpenDays` and `Promo2OpenDays` features. Added stationary business features: unified campaign promotions (`IsPromotionRunning`) and campaign duration (`PromoDuration`).
  * RMSPE successfully improved further to **16.33%**.

### Stage 3: Time-Series Performance
* **Date Conducted**: 2026-07-06
* **Model**: XGBoost Regressor
* **Key Observations**:
  * **Major Improvement**: Incorporating historical lags (`Sales_Lag_1`, `Sales_Lag_7`, `Sales_Lag_14`, `Sales_Lag_30`), rolling statistics (`Sales_RollingMean_7`, `Sales_RollingStd_7`, `Sales_RollingMean_14`), and missingness flags (`HasLag1`, `HasLag7`, `HasRolling7`) dropped the validation RMSPE to **13.21%** and boosted $R^2$ to **0.9049**.
  * **Explicit Uncertainty Mapping**: The introduction of binary missingness flags (`HasLag1`, `HasLag7`, `HasRolling7`) allows XGBoost to distinguish between an actual rolling sales average and a filled baseline average (`Store_AvgSales`) during early-sequence cold starts. This prevents decision splits from conflating true rolling means with filled statistics, leading to a significant **+3.9% reduction in RMSE** (from 991 to 952.43) and **1.18% drop in RMSPE** over simple imputation.

### Stage 4: Advanced Performance
* **Date Conducted**: 2026-07-06
* **Model**: XGBoost Regressor
* **Key Observations**:
  * **Cyclical Encoding Effectiveness**: Representing date periodicities (`DayOfWeek`, `Month`, `Day`, `WeekOfYear`) using sine and cosine components improved validation RMSE down to **949.87** and MAE to **649.07** ($R^2$ = `0.9055`).
  * **Generalization Gains**: Cyclical coordinates allow tree models to capture proximity across wrap-around boundaries (e.g. Saturday adjacent to Sunday, or December 31st adjacent to January 1st) in fewer partition splits, slightly boosting the model's overall accuracy.
