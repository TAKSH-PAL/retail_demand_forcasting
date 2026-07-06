# Error & Residual Analysis

This document provides a diagnostic evaluation of the winning XGBoost model to understand its predictions, residual behaviors, and identify structural areas of forecasting failures.

---

## Residual Analysis Visualization

Below are the diagnostic plots generated from predictions on the chronological validation set (June 15th, 2015 to July 31st, 2015).

![Residual Analysis Plots](/Users/takshpal/.gemini/antigravity-ide/brain/aed60d4a-30c5-4b23-817d-074038b821fd/residual_analysis.png)

---

## Key Diagnostic Statistics

* **Mean Residual**: `+104.25` (indicates a slight under-prediction bias overall)
* **Median Residual**: `+45.38`
* **Mean Absolute Error (MAE)**: `650.55`
* **Root Mean Squared Error (RMSE)**: `951.64`
* **Skewness of Residuals**: `1.9250` (heavily right-skewed; the model exhibits a tail of significant under-predictions where actual sales far exceed estimates)
* **Kurtosis of Residuals**: `36.5113` (highly leptokurtic, indicating heavy outlier error tails)

---

## Where Does the Model Fail?

### 1. The Saturday & Sunday shopping variance
Analyzing the Mean Absolute Percentage Error (MAPE) by day of the week shows a stark pattern:

| Day of Week | Mean APE (%) | Observation |
| :--- | :---: | :--- |
| **Monday (1)** | 10.44% | Spillover demand from Sunday store closures |
| Tuesday (2) | **7.51%** | Normal weekday (most stable/accurate) |
| Wednesday (3) | 8.46% | Normal weekday |
| Thursday (4) | 8.00% | Normal weekday |
| Friday (5) | 7.79% | Normal weekday |
| **Saturday (6)** | **15.22%** | **Worst weekday performance** (highly volatile volume) |
| **Sunday (7)** | **14.83%** | High variance due to limited open stores |

* **Analysis**: The model is highly accurate on standard weekdays (Tue-Fri APE is ~7.8%). However, the error doubles on **Saturdays** (15.22%) and **Sundays** (14.83%). Saturdays represent a massive retail spike day with volatile store-specific traffic, while Sundays represent a highly skewed day (only 1% of stores are open), making it difficult for the tree splits to capture the localized Sunday demand.

---

### 2. Specific Store Outliers (Store 909 and Store 876)
Looking at the top 5 worst predictions in validation:

* **Store 909 Under-predictions**:
  * On Monday 2015-06-22: Actual sales were **41,551**, predicted was **12,412** (Error: `+29,139`).
  * On Tuesday 2015-06-23: Actual sales were **30,038**, predicted was **12,170** (Error: `+17,868`).
* **Store 909 Over-predictions**:
  * On Monday 2015-06-29: Actual sales were **6,125**, predicted was **17,318** (Error: `-11,193`).
* **Analysis**: Store 909 undergoes massive, highly volatile swings (spiking up to 41k and dropping to 6k within a week) that do not align with its typical historical mean. This suggests localized stock clearances, promotions, or supply events that the global model cannot capture without store-specific indicators.

---

### 3. Non-Promo Days (Promo = 0)
* **Promo = 1 (Promoting)**: Mean APE is **`8.37%`**
* **Promo = 0 (No Promo)**: Mean APE is **`10.47%`**
* **Analysis**: Surprisingly, the model performs **better** when promotions are active. Promotional periods are highly structured and follow clean scheduling rules (Mon-Fri active weeks), meaning the model learns the promotional sales uplift very well. On non-promo days, baseline demand is driven by less visible factors (local weather, neighborhood foot traffic, competing events), leading to higher forecasting variance.
