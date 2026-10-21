# 🏪 Cerebal Kaggle — Complete ML Strategy Guide
### Solari Hub Daily Order Volume Forecasting (RMSLE)

---

## 📊 1. EDA — State of the Data

### Dataset Overview

| Dataset | Rows | Columns | Size |
|---|---|---|---|
| `orders_train.csv` | **970,379** | 9 | 32 MB |
| `orders_test.csv` | 46,830 | 8 | 1.4 MB |
| `hub_metadata.csv` | 1,115 | 10 | 47 KB |

- **1,115 unique hubs** — all appear in both train and test
- **Train period**: `2013-01-01` → `2015-06-19` (~2.5 years)
- **Test period**: `2015-06-20` → `2015-07-31` (~6 weeks in the future)

> [!IMPORTANT]
> This is a **temporal holdout** — not a random split. The test set is a future period. Any validation strategy must respect time order (no leakage).

---

### 🎯 Target Variable: `OrderVolume`

| Stat | Value |
|---|---|
| Min | 0 |
| Max | 38,722 |
| Mean | 5,762.77 |
| Median | 5,736 |
| Std | 3,855.46 |
| **Zeros** | **166,323 / 970,379 = 17.14%** |

**Key finding:** All zero-OrderVolume rows correspond exactly to `IsOpen = 0` (hub closed). When `IsOpen = 0`, OrderVolume is **always 0** — no exceptions. This means the zero-inflation is perfectly explained by closure status.

```
IsOpen=0: count=166,269, mean=0, max=0      ← always zero
IsOpen=1: count=804,110, mean=6,954, max=38,722 ← real signal here
```

> [!NOTE]
> You can use a **two-stage model**: (1) predict if hub is open (trivially given by `IsOpen` flag in test), then (2) predict volume for open days only.

---

### 🔍 Missing Values

**Train (`orders_train.csv`):** ✅ **Zero missing values** — perfectly clean.

**Meta (`hub_metadata.csv`):**

| Column | Missing |
|---|---|
| `CompetitorDistance` | 3 (0.3%) — trivial |
| `CompetitorOpenSinceMonth` | 354 (31.7%) — no competitor nearby |
| `CompetitorOpenSinceYear` | 354 (31.7%) — no competitor nearby |
| `LoyaltyProgramSinceWeek` | 544 (48.8%) — hub not in loyalty program |
| `LoyaltyProgramSinceYear` | 544 (48.8%) — hub not in loyalty program |
| `LoyaltyProgramInterval` | 544 (48.8%) — hub not in loyalty program |

> [!NOTE]
> Most "missing" values are **structurally missing** (e.g., no competitor = no competitor date). They aren't random gaps — they carry information. Treat them as a separate category/flag rather than imputing with mean/median.

---

### 📈 Feature Distributions

**Operational flags (train):**

| Feature | Values |
|---|---|
| `IsOpen` | 1: 804K (82.9%), 0: 166K (17.1%) |
| `PromoActive` | 0: 599K (61.7%), 1: 371K (38.3%) |
| `RegionalHoliday` | 0: 939K, 1: 20K, 2: 6.7K, 3: 4.1K |
| `SchoolClosureFlag` | 0: 802K, 1: 168K |
| `Weekday` | Fairly balanced 1–7 (~138K each) |

**Hub metadata:**

| Feature | Distribution |
|---|---|
| `HubFormat` | 1: 602, 4: 348, 3: 148, 2: 17 (imbalanced) |
| `AssortmentTier` | 1: 593, 3: 513, 2: 9 (nearly binary) |
| `LoyaltyProgram` | 1: 571, 0: 544 (balanced) |
| `CompetitorDistance` | Mean: 5,405m, Std: 7,663m, Min: 20m, Max: 75,860m |

**Critical correlation:**
```
AppSessions ↔ OrderVolume: r = 0.896  ← extremely strong signal
```
`AppSessions` is your **single most powerful feature** (it's a same-day demand proxy). However — in the test set, `AppSessions` is **missing** (test has no `AppSessions` or `OrderVolume`). You'll need lagged/historical AppSessions as features.

---

## 🔧 2. Data Preprocessing

### Step 1 — Handle the Zero/Closed Hub Problem
```python
# STRATEGY: Only model open-hub rows
# IsOpen=0 → predict 0 directly (100% accurate, always true in train)
train_open = train[train['IsOpen'] == 1].copy()
```

### Step 2 — Merge Metadata
```python
df = train_open.merge(meta, on='HubID', how='left')
```

### Step 3 — Handle Structural Missing Values
```python
# CompetitorDistance — 3 missing: impute with median
df['CompetitorDistance'].fillna(df['CompetitorDistance'].median(), inplace=True)

# Competitor date — missing = no competitor → binary flag
df['HasCompetitor'] = df['CompetitorOpenSinceYear'].notna().astype(int)

# Loyalty Program date — missing = not participating → already captured by LoyaltyProgram=0
# LoyaltyProgramSinceWeek/Year: fill 0 if LoyaltyProgram=0
df['LoyaltyProgramSinceYear'] = df['LoyaltyProgramSinceYear'].fillna(0)
df['LoyaltyProgramSinceWeek'] = df['LoyaltyProgramSinceWeek'].fillna(0)

# LoyaltyProgramInterval: encode as # of programs per year
def count_intervals(s):
    if pd.isna(s): return 0
    return len(s.split(','))
df['LoyaltyIntervalCount'] = df['LoyaltyProgramInterval'].apply(count_intervals)
```

### Step 4 — Parse Dates
```python
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfMonth'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Quarter'] = df['Date'].dt.quarter
```

### Step 5 — Encode Categoricals
```python
# HubFormat: 4 types — treat as categorical (one-hot or ordinal)
# AssortmentTier: ordinal (1=basic, 2=extended, 3=premium)
# RegionalHoliday: ordinal (0,1,2,3)
# LoyaltyProgramInterval: parse into frequency (done above)
```

---

## ⚙️ 3. Feature Engineering

This is the most impactful phase. The key idea: since `AppSessions` isn't in the test set, you must create **lag features and rolling statistics** from historical data.

### 🕒 Time-Series Lag Features (Per Hub)
```python
# Sort by Hub + Date first
df = df.sort_values(['HubID', 'Date'])

# Short-term memory
df['lag_7']  = df.groupby('HubID')['OrderVolume'].shift(7)   # same weekday last week
df['lag_14'] = df.groupby('HubID')['OrderVolume'].shift(14)
df['lag_28'] = df.groupby('HubID')['OrderVolume'].shift(28)  # 4 weeks ago same weekday

# Rolling statistics
df['roll_7_mean']  = df.groupby('HubID')['OrderVolume'].shift(1).rolling(7).mean()
df['roll_14_mean'] = df.groupby('HubID')['OrderVolume'].shift(1).rolling(14).mean()
df['roll_28_mean'] = df.groupby('HubID')['OrderVolume'].shift(1).rolling(28).mean()
df['roll_7_std']   = df.groupby('HubID')['OrderVolume'].shift(1).rolling(7).std()
df['roll_7_max']   = df.groupby('HubID')['OrderVolume'].shift(1).rolling(7).max()
df['roll_7_min']   = df.groupby('HubID')['OrderVolume'].shift(1).rolling(7).min()

# Expanding mean (hub lifetime average up to that point)
df['expanding_mean'] = df.groupby('HubID')['OrderVolume'].shift(1).expanding().mean()
```

### 📅 Calendar & Cyclical Features
```python
import numpy as np
# Cyclical encoding for weekday, month (avoids 1→7 discontinuity)
df['weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
df['month_sin']   = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos']   = np.cos(2 * np.pi * df['Month'] / 12)
df['doy_sin']     = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['doy_cos']     = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# Is weekend flag
df['IsWeekend'] = (df['Weekday'].isin([6, 7])).astype(int)
```

### 🏪 Hub-Level Aggregate Features (Target Encoding Style)
```python
# Hub's overall mean/std volume (on open days only, computed on train only)
hub_stats = df.groupby('HubID')['OrderVolume'].agg(['mean','std','median']).reset_index()
hub_stats.columns = ['HubID','hub_mean_vol','hub_std_vol','hub_median_vol']
df = df.merge(hub_stats, on='HubID', how='left')

# Weekday-level means per hub
hub_weekday = df.groupby(['HubID','Weekday'])['OrderVolume'].mean().reset_index()
hub_weekday.columns = ['HubID','Weekday','hub_weekday_mean']
df = df.merge(hub_weekday, on=['HubID','Weekday'], how='left')

# Promo effect per hub
hub_promo = df.groupby(['HubID','PromoActive'])['OrderVolume'].mean().unstack()
hub_promo['promo_lift'] = hub_promo[1] / hub_promo[0].clip(lower=1)
```

> [!CAUTION]
> Target encoding MUST be done with **out-of-fold** encoding to prevent leakage. Use `KFold` with time-aware splits.

### 🏁 Competitor & Loyalty Features
```python
# Competitor tenure (how long has competitor been open?)
# e.g., competitor opened 2008-09 → as of 2015 = ~6.75 years
df['CompetitorAge_months'] = (
    (df['Year'] - df['CompetitorOpenSinceYear'].fillna(df['Year'])) * 12 +
    (df['Month'] - df['CompetitorOpenSinceMonth'].fillna(df['Month']))
).clip(lower=0)

# Loyalty program tenure
df['LoyaltyAge_weeks'] = (
    (df['Year'] - df['LoyaltyProgramSinceYear'].fillna(df['Year'])) * 52 +
    (df['Week'] - df['LoyaltyProgramSinceWeek'].fillna(df['Week']))
).clip(lower=0)
```

### 📉 Promotion & Holiday Interaction Features
```python
df['Promo_x_Weekday']   = df['PromoActive'] * df['Weekday']
df['Promo_x_Holiday']   = df['PromoActive'] * (df['RegionalHoliday'] > 0).astype(int)
df['Promo_x_School']    = df['PromoActive'] * df['SchoolClosureFlag']
df['Holiday_x_Weekend'] = (df['RegionalHoliday'] > 0).astype(int) * df['IsWeekend']
```

### 🔁 AppSessions Lags (Key since it's missing in test!)
```python
# AppSessions is available in train — use its lags as proxy
df['appsess_lag_7']    = df.groupby('HubID')['AppSessions'].shift(7)
df['appsess_lag_14']   = df.groupby('HubID')['AppSessions'].shift(14)
df['appsess_roll7_mean'] = df.groupby('HubID')['AppSessions'].shift(1).rolling(7).mean()
```

---

## ✅ 4. Validation Strategy

> [!IMPORTANT]
> **Never use random K-Fold.** This is a time-series problem with a future holdout.

Use **TimeSeriesSplit** or a manual walk-forward split:

```python
# Manual time-based split
CUTOFF = '2015-04-01'   # last ~2.5 months as local val
train_cv = df[df['Date'] < CUTOFF]
val_cv   = df[df['Date'] >= CUTOFF]

# Or for CV: sliding window
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

**Metric to optimize locally:**
```python
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
```

---

## 🤖 5. ML Models — Ranked by Expected RMSLE Performance

### 🥇 Tier 1 — Best Expected RMSLE

#### 1. **LightGBM** ⭐ (Primary Recommendation)
- Native handling of categoricals, very fast on 970K rows
- Best baseline for tabular time-series with lag features
- Built-in `tweedie` or `huber` loss; can use custom RMSLE objective
- Handles heterogeneous hubs naturally via hub-level features

```python
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
}
# Train on log1p(OrderVolume), predict, then expm1
```

#### 2. **XGBoost**
- Slightly slower than LightGBM but similarly powerful
- Use `tree_method='hist'` for speed
- Ensemble with LightGBM for extra gain (~0.005 RMSLE improvement)

#### 3. **CatBoost**
- Excellent for categorical features (HubFormat, AssortmentTier, Weekday)
- Slower to train but often on-par or better with categoricals
- Built-in ordered target encoding avoids leakage

---

### 🥈 Tier 2 — Good Supplements / Blending

#### 4. **Prophet (per-hub)**
- Meta's time-series model — excellent for seasonality + holiday effects
- Run one Prophet model **per hub** → computationally heavy but pure TS
- Use as a **blending component**, not a standalone

#### 5. **Linear Models with Rich Features**
- `Ridge` / `Lasso` on log1p target with polynomial features
- Very fast, interpretable, good regularization baseline
- Useful for feature selection insights

#### 6. **Random Forest**
- Good for feature importance analysis
- Slower and usually worse than LightGBM on this scale
- Use as a diversity model in ensembles

---

### 🥉 Tier 3 — Experimental / Advanced

#### 7. **Neural Networks (MLP / TabNet)**
- Harder to tune, but can capture complex interactions
- TabNet gives feature importance + attention mechanism
- Useful for an ensemble component if LightGBM is already tuned

#### 8. **N-BEATS / N-HiTS / TFT (Temporal Fusion Transformer)**
- Deep learning pure time-series models
- Very powerful but require significant setup (Darts / PyTorch Forecasting)
- Diminishing returns vs. well-tuned LightGBM for this problem size

---

### 🏆 Winning Strategy: Stacked Ensemble

```
Stage 1: LightGBM + XGBoost + CatBoost (with lag features)
Stage 2: Linear blender (Ridge) on OOF predictions
Final:   Weighted average or Optuna-tuned blend
```

Expected RMSLE: **~0.08–0.15** depending on feature engineering quality.

---

## 📋 6. Full ML Pipeline Summary

```
1. Load data & merge metadata
         ↓
2. Filter IsOpen=1 rows for modeling (predict 0 for IsOpen=0)
         ↓
3. Parse dates, engineer calendar features
         ↓
4. Create lag features (7, 14, 28 days) per hub
         ↓
5. Create rolling stats (7, 14, 28 day windows) per hub
         ↓
6. Encode structurally missing values as features (HasCompetitor, etc.)
         ↓
7. Create hub-level aggregates (OOF target encoding)
         ↓
8. Create interaction features (Promo × Holiday, etc.)
         ↓
9. Log1p transform the target
         ↓
10. TimeSeriesSplit cross-validation
         ↓
11. Train LightGBM (primary) + XGBoost + CatBoost
         ↓
12. OOF blending with Ridge
         ↓
13. expm1 back-transform predictions
         ↓
14. Clip negatives to 0 (RMSLE undefined for negatives)
         ↓
15. Fill IsOpen=0 rows with 0
         ↓
16. Submit
```

---

## ⚠️ 7. Key Gotchas & Tips

> [!WARNING]
> **Train-Test Leakage Risk:** Don't use future data when computing rolling/lag features. Always `.shift(1)` before `.rolling()`.

> [!WARNING]
> **AppSessions is not in test set.** Only use AppSessions lags from train data. Do not use raw AppSessions as a feature (it won't exist at prediction time).

> [!TIP]
> **Log1p transform:** Since metric is RMSLE, training on `log1p(OrderVolume)` and minimizing RMSE is equivalent to minimizing RMSLE — this is the standard trick.

> [!TIP]
> **Hub-specific models:** For hubs with very different patterns (new vs. mature, Format 1 vs. 4), consider training separate models or using hub-embedding features.

> [!TIP]
> **Promo effect is huge:** `PromoActive=1` days have significantly higher volume. Interaction of PromoActive × Weekday × HubFormat is likely one of the top features.

> [!TIP]
> **CompetitorDistance as log:** The distribution is right-skewed (mean 5.4km, max 75.8km). Log-transform it: `log1p(CompetitorDistance)`.

> [!NOTE]
> **AssortmentTier=2 has only 9 hubs** — very rare. Consider merging with Tier 1 or treating as an outlier group.

---

## 🗂️ Recommended Kaggle Notebook Structure

1. **Section 1** — Data Loading & Overview
2. **Section 2** — EDA (distributions, time series plots, correlation heatmap)
3. **Section 3** — Feature Engineering Pipeline
4. **Section 4** — Model Training (LightGBM baseline → CV loop)
5. **Section 5** — Feature Importance Analysis
6. **Section 6** — Ensemble & Blending
7. **Section 7** — Submission Generation
