# Model Selection & Comparison

This document presents a comprehensive evaluation of five different machine learning models trained on our Stage 4 advanced feature-engineered dataset.

---

## Performance Comparison Table

All models were trained on the `log1p(Sales)` target and evaluated on the out-of-time chronological validation set (June 15th, 2015 onwards).

| Model | RMSE | RMSPE (%) | MAE | R² | Train Time | Performance Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **951.64** | **13.24%** | **650.55** | **0.9051** | **1.75s** | **Winner (Best accuracy and speed)** |
| **LightGBM** | 975.04 | 13.56% | 664.41 | 0.9004 | 1.78s | Runner-up (Extremely close and fast) |
| **Random Forest** | 1009.65 | 14.47% | 688.87 | 0.8932 | 29.70s | Scalability bottleneck |
| **CatBoost** | 1020.66 | 14.23% | 696.04 | 0.8908 | 3.02s | Solid performance |
| **Linear Regression** | 1879.41 | 20.72% | 971.50 | 0.6299 | 0.44s | Baseline linear model |

---

## Detailed Model Analysis & Insights

### 1. Why the Gradient Boosting Frameworks Win
* **Sequential Error Correction**: Gradient boosting algorithms (XGBoost, LightGBM, CatBoost) fit shallow trees sequentially to minimize residual errors. This allows them to model complex, localized interactions (such as the combined impact of Promo, DayOfWeek, and Store-AvgSales) much more accurately than bagging or linear models.
* **Histogram-based splits**: Modern frameworks use histogram binning on continuous features (such as `CompetitionDistance` and lag averages), reducing split search time from $O(N \log N)$ to $O(N_{\text{bins}})$, letting them train on 800k rows in under 2 seconds.

### 2. Why XGBoost Outperforms LightGBM and CatBoost
* **Split Search Method**: XGBoost's level-wise growth with histogram binning (`tree_method='hist'`) matches the dense feature structures (lags + target encoded features) of this dataset.
* **LightGBM**'s leaf-wise growth is highly efficient but can occasionally overfit local noise on smaller validation windows (yielding a slightly higher RMSPE of `13.56%`).
* **CatBoost** specializes in handling raw categorical features natively. Since we pre-engineered the categorical store columns into numerical target encodings (`Store_AvgSales`), CatBoost's primary categorical strength was already materialized in the dataset, leading to similar performance as XGBoost but with slightly longer training time (3.02s).

### 3. Why Random Forest and Linear Regression Underperformed
* **Random Forest Scalability**: Random Forest builds deep, independent trees in parallel. While it achieves strong predictive capabilities ($R^2 = 0.8932$), it does not optimize sequentially, requiring more trees and deeper splits to match the boosting accuracy. This leads to a training bottleneck (29.70s), making it 15x slower than XGBoost.
* **Linear Regression Limitations**: Retail sales demand contains severe non-linear spikes (e.g., promo days, weekend closures, holidays). Linear Regression assumes linear, additive relationships and fails to capture these complex interactions, leading to a poor $R^2$ of `0.6299`.
