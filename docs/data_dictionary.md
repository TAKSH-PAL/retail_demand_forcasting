# Data Dictionary

This document provides a comprehensive description of the datasets used in the Retail Demand Forecasting & Inventory Optimization project. The datasets consist of historical sales records (`train.csv`), a test set for forecasting (`test.csv`), and store metadata (`store.csv`).

---

## Dataset Overview

| File | Path | Shape | Description |
| :--- | :--- | :--- | :--- |
| **Store Metadata** | `data/raw/store.csv` | 1,115 rows, 10 columns | Metadata detailing permanent characteristics of each store (type, assortment, competition, promotions). |
| **Training Data** | `data/raw/train.csv` | 1,017,209 rows, 9 columns | Historical daily sales data from 2013-01-01 to 2015-07-31. |
| **Test Data** | `data/raw/test.csv` | 41,088 rows, 8 columns | Out-of-sample data representing stores and dates for which sales must be predicted. |

---

## Store Metadata (`store.csv`)
This file contains metadata describing the attributes of each of the 1,115 stores.

| Column Name | Data Type | Null Count (%) | Range / Unique Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Store** | `int64` (Id) | 0 (0.0%) | 1 to 1,115 | A unique identifier for each retail store. |
| **StoreType** | `string` (Categorical) | 0 (0.0%) | 'a', 'b', 'c', 'd' | Differentiates between 4 different store models. |
| **Assortment** | `string` (Categorical) | 0 (0.0%) | 'a' (basic), 'b' (extra), 'c' (extended) | Describes the assortment level of products offered at the store. |
| **CompetitionDistance** | `float64` (Numeric) | 3 (0.27%) | 20 to 75,860 | Distance in meters to the nearest competitor store. |
| **CompetitionOpenSinceMonth** | `float64` (Categorical) | 354 (31.75%) | 1 to 12 | The approximate calendar month when the nearest competitor was opened. |
| **CompetitionOpenSinceYear** | `float64` (Categorical) | 354 (31.75%) | 1900 to 2015 | The approximate calendar year when the nearest competitor was opened. |
| **Promo2** | `int64` (Binary) | 0 (0.0%) | 0, 1 | Indicates whether the store participates in Promo2: `0` = Not participating, `1` = Participating. Promo2 is a continuing and consecutive promotion for some stores. |
| **Promo2SinceWeek** | `float64` (Numeric) | 544 (48.79%) | 1 to 50 | The calendar week index of the year when the store started participating in Promo2. |
| **Promo2SinceYear** | `float64` (Numeric) | 544 (48.79%) | 2009 to 2015 | The calendar year when the store started participating in Promo2. |
| **PromoInterval** | `string` (Categorical) | 544 (48.79%) | 'Jan,Apr,Jul,Oct'<br>'Feb,May,Aug,Nov'<br>'Mar,Jun,Sept,Dec' | Describes the consecutive calendar months when Promo2 restarts. E.g., `"Feb,May,Aug,Nov"` means each round starts in February, May, August, and November. |

---

## Training Dataset (`train.csv`)
Historical daily transaction records containing the target variable (`Sales`).

| Column Name | Data Type | Null Count (%) | Range / Unique Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Store** | `int64` (Id) | 0 (0.0%) | 1 to 1,115 | A unique identifier matching the store in `store.csv`. |
| **DayOfWeek** | `int64` (Categorical) | 0 (0.0%) | 1 (Monday) to 7 (Sunday) | Numeric index representing the day of the week. |
| **Date** | `string` (Date) | 0 (0.0%) | YYYY-MM-DD | The date of the transaction record. |
| **Sales** | `int64` (Numeric) | 0 (0.0%) | 0 to 41,551 | **Target Variable.** The daily turnover/sales revenue for the store. |
| **Customers** | `int64` (Numeric) | 0 (0.0%) | 0 to 7,388 | The number of customers visiting the store on that day. |
| **Open** | `int64` (Binary) | 0 (0.0%) | 0 (Closed), 1 (Open) | An indicator showing whether the store was open. |
| **Promo** | `int64` (Binary) | 0 (0.0%) | 0, 1 | Indicates whether the store was running a promotion on that day. |
| **StateHoliday** | `string/int` (Categorical) | 0 (0.0%) | '0' or 0 (None), 'a' (Public), 'b' (Easter), 'c' (Christmas) | Indicates a state holiday. Stores are generally closed on state holidays (with exceptions). |
| **SchoolHoliday** | `int64` (Binary) | 0 (0.0%) | 0, 1 | Indicates if the store and date combination was affected by the closure of public schools. |

---

## Test Dataset (`test.csv`)
Out-of-sample data representing the period for which daily sales must be predicted.

| Column Name | Data Type | Null Count (%) | Range / Unique Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Id** | `int64` (Id) | 0 (0.0%) | 1 to 41,088 | A unique identifier representing a (Store, Date) pair within the test set. |
| **Store** | `int64` (Id) | 0 (0.0%) | 1 to 1,115 (856 unique) | Unique identifier matching the store in `store.csv`. |
| **DayOfWeek** | `int64` (Categorical) | 0 (0.0%) | 1 (Monday) to 7 (Sunday) | Numeric index representing the day of the week. |
| **Date** | `string` (Date) | 0 (0.0%) | YYYY-MM-DD | The date for which predictions are required. |
| **Open** | `float64` (Binary) | 11 (0.03%) | 0 (Closed), 1 (Open) | An indicator showing whether the store was open. *(Note: Contains 11 missing values that will require imputation during preprocessing).* |
| **Promo** | `int64` (Binary) | 0 (0.0%) | 0, 1 | Indicates whether the store is running a promotion on that day. |
| **StateHoliday** | `string` (Categorical) | 0 (0.0%) | '0' (None), 'a' (Public) | Indicates a state holiday on that day. |
| **SchoolHoliday** | `int64` (Binary) | 0 (0.0%) | 0, 1 | Indicates if the store and date combination is affected by the closure of public schools. |
