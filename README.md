# 🏪 Solari Enterprise Demand Forecasting Engine
### Multi-Hub Quick-Commerce Optimization Platform & Grandmaster Architecture

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Metric: RMSLE](https://img.shields.io/badge/Metric-RMSLE-brightgreen.svg)](#-evaluation-metric)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI%20%7C%20REST-009688.svg)](#-production-rest-api-fastapi-microservice)
[![Container: Docker](https://img.shields.io/badge/Container-Docker%20%7C%20Compose-2496ED.svg)](#-docker-containerization)
[![CI: GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](#-testing-suite--mathematical-invariants)
[![Tests: 13 Passed](https://img.shields.io/badge/Tests-13%20Passed-success.svg)](#-testing-suite--mathematical-invariants)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

**Company_X (Solari)** operates a nationwide ultra-fast fulfillment logistics network of **1,115 micro-fulfillment dark stores (hubs)** guaranteeing grocery and household delivery within 15–30 minutes. Due to tight delivery Service Level Agreements (SLAs) and constrained hub storage footprints, demand forecasting accuracy directly governs network profitability:
* **Over-forecasting** exhausts refrigerated capacity and causes perishable spoilage.
* **Under-forecasting** triggers rider stockouts, delivery queue overflows, and customer churn.

This repository implements a **production-grade, modular demand forecasting engine** designed to predict daily order volume (`OrderVolume`) across all 1,115 hubs over a future 6-week horizon (**June 20, 2015 to July 31, 2015**).

The system adapts and extends the winning methodology of **Gert Jacobusse** (1st place winner of Kaggle's *Rossmann Store Sales* competition), combining a **60+ feature temporal engineering engine**, a **4-model heterogeneous gradient boosting suite**, a **Weighted Harmonic Mean aggregator**, a mathematically calibrated **Year-over-Year (YoY) anchor shrinkage layer**, and a **low-latency FastAPI serving microservice**.

---

## 🏛️ Software Architecture & Design Patterns

The codebase is built according to production software engineering standards, cleanly separated into an object-oriented Python package (`src/`), a REST API service (`api/`), configuration dataclasses, unit tests, and research notebooks:

```text
├── .github/
│   └── workflows/
│       └── ci.yml                    # Automated GitHub Actions CI workflow (matrix: 3.10, 3.11)
│
├── api/                              # 🌐 FastAPI Serving Microservice
│   ├── __init__.py
│   ├── schemas.py                    # Strictly-typed Pydantic request & response contracts
│   └── main.py                       # REST endpoints (/health, /predict, /predict/batch, /hubs)
│
├── Dataset/                          # Raw operational data
│   ├── orders_train.csv              # 970,379 rows: Daily records (2013-01-01 to 2015-06-19)
│   ├── orders_test.csv               # 46,830 rows: Future test forecast period (2015-06-20 to 2015-07-31)
│   ├── hub_metadata.csv              # 1,115 rows: Structural hub attributes, competitor, & loyalty specs
│   └── sample_submission.csv         # Submission template with Id and OrderVolume
│
├── docs/                             # 📚 Competition Documentation & Research
│   ├── Rossmann_nr1_doc.pdf          # Winning 1st-place documentation by Gert Jacobusse
│   ├── cerebal_kaggle_analysis.md    # 430+ line ML roadmap, EDA stats, & loss trade-offs
│   ├── cells_summary.txt             # Cell-by-cell architectural mapping to Rossmann paper
│   ├── Overview.md                   # Quick-commerce business overview & metric specifications
│   ├── data.md                       # Field-by-field dataset definitions & schema
│   └── image.png                     # RMSLE mathematical equation visual
│
├── notebooks/                        # 📓 Research & Exploration Archive
│   ├── final_championship_notebook.ipynb # Interactive master championship notebook
│   ├── notebook0ee4452879(1).ipynb       # Executed Kaggle notebook with outputs & diagnostics
│   ├── cerebal_solari_forecasting.ipynb  # Rossmann 1st-place solution pipeline template
│   ├── cerebal_solari_forecastingv1.ipynb# EDA, correlation heatmaps, and baseline models
│   └── cerebal_rossmann_champion.ipynb   # Two-stage customer/session prototype
│
├── src/                              # 📦 Core Production Package
│   ├── __init__.py
│   ├── config.py                     # Immutable typed configuration dataclasses
│   ├── pipeline.py                   # End-to-end CLI orchestrator & execution entrypoint
│   │
│   ├── data/                         # Data layer
│   │   ├── __init__.py
│   │   ├── loader.py                 # Schema validation & memory dtype downcasting
│   │   └── preprocessor.py           # Calendar decomposition & structural missingness
│   │
│   ├── features/                     # Feature Engineering Engine (Transformer Protocol)
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseFeatureTransformer abstract base class
│   │   ├── ratios.py                 # Hub elasticity & promotional/holiday ratios
│   │   ├── trend.py                  # Per-hub Ridge regression trend slope estimator
│   │   ├── distance.py               # Bidirectional promo day counters & reopening surge
│   │   ├── lags.py                   # YoY cyclic lags (364/365/371) & causal rolling windows
│   │   ├── target_encoder.py         # TimeSeriesSplit out-of-fold target encoder
│   │   └── pipeline.py               # Composite FeaturePipeline coordinator
│   │
│   └── models/                       # Model Zoo & Ensembling
│       ├── __init__.py
│       ├── base.py                   # BaseForecaster ABC & mathematical RMSLE implementation
│       ├── lgbm_model.py             # High-capacity leaf-wise LightGBM regressor
│       ├── xgb_model.py              # Histogram depth-wise XGBoost regressor
│       ├── catboost_model.py         # Categorical-aware symmetric oblivious CatBoost
│       ├── summer_specialist.py      # Seasonal domain specialist (May–September sub-window)
│       └── ensemble.py               # Weighted Harmonic Mean & YoY-Anchored Calibrator
│
├── tests/                            # 🧪 Comprehensive Test Suite (13 Tests Passing)
│   ├── __init__.py
│   ├── test_api.py                   # FastAPI endpoint status, schemas, & invariant tests
│   ├── test_metrics.py               # RMSLE & Jensen's inequality (Harmonic <= Arithmetic Mean)
│   ├── test_features.py              # Shift-1 causality guards & 364-day phase alignment
│   └── test_data.py                  # Memory compression & numeric type downcasting
│
├── Dockerfile                        # Multi-stage production container definition
├── docker-compose.yml                # Microservice container orchestration
├── Makefile                          # Standard developer command automation
├── requirements.txt                  # Production dependencies
├── .gitignore                        # Git hygiene & artifact exclusions
└── README.md                         # Technical architectural documentation (this file)
```

### Key Design Patterns Implemented
1. **Transformer Pattern (`fit`/`transform`)**: All feature generators inherit from [`BaseFeatureTransformer`](file:///a:/Github/Cerebal%20Kaggle/src/features/base.py) to guarantee strict separation between training state estimation and test projection.
2. **Composite Pattern**: [`FeaturePipeline`](file:///a:/Github/Cerebal%20Kaggle/src/features/pipeline.py) composes individual domain transformers into a single atomic transformation step.
3. **Strategy Pattern**: [`BaseForecaster`](file:///a:/Github/Cerebal%20Kaggle/src/models/base.py) establishes an interchangeable interface for LightGBM, XGBoost, and CatBoost.
4. **Inversion of Control & Invariant Guards**: State predicates are verified before and after operations, and hyperparameters are injected via immutable dataclasses in [`src/config.py`](file:///a:/Github/Cerebal%20Kaggle/src/config.py).

---

## 🧠 Computer Science Fundamentals & Architectural Decisions

This system was designed from the ground up with deep algorithmic, mathematical, and systems considerations:

### 1. Computational Complexity & Vectorization
* **Time Complexity**: Feature extraction across 1 million rows avoids slow nested Python loops. Grouped lag generation, rolling statistics, and cumulative distance counters are implemented via **vectorized Pandas/NumPy operations** executing in $\mathcal{O}(N \log N)$ (due to grouping and sorting) or $\mathcal{O}(N)$ sequential scans.
* **Closed-Form Ridge Regression**: Trailing sales trends are computed per hub using Ridge regression:
  $$\hat{\beta} = \left(X^T X + \alpha I\right)^{-1} X^T y$$
  With $p=3$ features (`[DayNumber, Weekday, PromoActive]`) and $T \le 365$ days per hub, matrix inversion is $\mathcal{O}(p^3) = \mathcal{O}(27)$. For all $H = 1,115$ hubs, the entire network trend slope is estimated in under **1.8 seconds**.

### 2. Memory Footprint Optimization (Cache Locality)
Default Pandas ingestion loads numerical columns as 64-bit IEEE floats and 64-bit signed integers (8 bytes/value). With 60+ engineered features across 1 million rows, uncompressed DataFrames exceed 500MB per copy, saturating L3 cache and slowing decision tree quantile binning.
* [`DataLoader.downcast_dtypes`](file:///a:/Github/Cerebal%20Kaggle/src/data/loader.py) dynamically downcasts integers to `int8`/`int16`/`int32` and floats to IEEE `float32` (4 bytes/value).
* **Result**: **46.9% reduction in RAM usage** (1.91MB vs 3.61MB per 50k rows), doubling throughput during gradient boosting tree construction.

### 3. Leak-Free Partition Splitting via Indicator Flags
A subtle bug in time-series implementations occurs when concatenating train and test partitions to compute running lags:
```python
# ANTI-PATTERN: iloc[:train_len] after sorting by [HubID, Date]
full = pd.concat([train_df, test_df]).sort_values(["HubID", "Date"])
train_out = full.iloc[:train_len]  # BUG! Test rows are now interleaved!
```
When sorted by `[HubID, Date]`, future test rows get sorted into the top partition, causing silent forward leakage and corrupting training targets with NaNs.
* **Our Solution**: [`_partition_is_test`](file:///a:/Github/Cerebal%20Kaggle/src/features/lags.py) partition indicator flags are injected before concatenation and filtered explicitly post-transformation, ensuring mathematical partition integrity.

### 4. Mathematical Equivalence: Minimizing RMSLE via MSE on Log Space
The competition evaluates submissions on Root Mean Squared Logarithmic Error:
$$\text{RMSLE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^n \left(\log(\hat{y}_i + 1) - \log(y_i + 1)\right)^2}$$

Let $z_i = \log(y_i + 1)$ and $\hat{z}_i = \log(\hat{y}_i + 1)$. Standard Mean Squared Error (MSE) in log space is:
$$\text{MSE}(z, \hat{z}) = \frac{1}{n} \sum_{i=1}^n \left(\hat{z}_i - z_i\right)^2 = \frac{1}{n} \sum_{i=1}^n \left(\log(\hat{y}_i + 1) - \log(y_i + 1)\right)^2$$
Thus:
$$\text{RMSLE}(y, \hat{y}) = \sqrt{\text{MSE}(z, \hat{z})}$$
Because the square root is a strictly monotonic transformation on $\mathbb{R}_{\ge 0}$, the parameter vector $\theta^*$ that minimizes log-space MSE is **identical** to the parameter vector that minimizes RMSLE:
$$\arg\min_{\theta} \text{RMSLE}(y, f_\theta(X)) \equiv \arg\min_{\theta} \text{MSE}\left(\log(1 + y), \log(1 + f_\theta(X))\right)$$
This mathematical equivalence allows us to train high-speed GBDT histogram engines (LightGBM/XGBoost) natively on standard RMSE loss without custom second-order Hessian approximations.

### 5. Jensen's Inequality & Harmonic Mean Aggregation
Let $x_1, x_2, \dots, x_M > 0$ be predictions from $M$ diverse models with convex weights $w_i \ge 0$ ($\sum w_i = 1$). The Weighted Harmonic Mean is:
$$H(x, w) = \frac{1}{\sum_{i=1}^M \frac{w_i}{x_i}}$$

Because the function $g(t) = \frac{1}{t}$ is strictly convex on $(0, \infty)$, **Jensen's Inequality** states:
$$g\left(\sum_{i=1}^M w_i x_i\right) \le \sum_{i=1}^M w_i g(x_i) \implies \frac{1}{\sum_{i=1}^M w_i x_i} \le \sum_{i=1}^M \frac{w_i}{x_i}$$
Inverting both sides:
$$H(x, w) = \frac{1}{\sum_{i=1}^M \frac{w_i}{x_i}} \le \sum_{i=1}^M w_i x_i = A(x, w)$$
with equality if and only if $x_1 = x_2 = \dots = x_M$.

**Why this matters under RMSLE loss**:
The RMSLE loss function penalizes over-predictions $\hat{y} \gg y$ exponentially harder than under-predictions $\hat{y} \ll y$. The Harmonic Mean acts as a conservative regularizer, pulling predictions toward the lower bound when models disagree and preventing costly right-tail loss spikes.

### 6. Recursive Horizon Attenuation & The 80/20 YoY Anchor
When predicting 42 days into the future, recursive short-term autoregressive features (`lag_7`, `lag_14`, rolling means) suffer variance attenuation because ground-truth values are unobserved.
In [`notebook0ee4452879(1).ipynb`](file:///a:/Github/Cerebal%20Kaggle/notebooks/notebook0ee4452879(1).ipynb), inspection of raw test predictions revealed a mean open-day prediction of **~12,586** (over-predicted vs the historical summer open-day average of **~7,074–7,200**).

**The Bayesian-Inspired Solution**:
We formulate the final forecast as a convex combination of an empirical seasonal prior (YoY Anchor) and the daily model likelihood:
$$\hat{y}_{\text{final}} = \alpha \cdot \left(y_{t-364} \times g_{\text{YoY}}\right) + (1 - \alpha) \cdot \hat{y}_{\text{ensemble}}$$
* $y_{t-364}$ is the exact day-of-week aligned demand from 52 weeks prior ($364 = 52 \times 7$, phase-locking calendar weekdays).
* $g_{\text{YoY}} = 1.013$ is the empirical macro growth rate observed in training data.
* $\alpha = 0.80$ anchors overall hub scale to true summer demand (~7,200 orders), while the 20% ensemble provides intra-week promo elasticity, weekday shifts, and hub trend slopes.
* **Operational Invariant Filter**: All predictions where `IsOpen == 0` are hardcoded to `0.0` in $\mathcal{O}(1)$ time.

---

## 🌐 Production REST API (FastAPI Microservice)

The service includes a production-ready **FastAPI inference microservice** in [`api/main.py`](file:///a:/Github/Cerebal%20Kaggle/api/main.py) with interactive OpenAPI/Swagger documentation at `http://localhost:8000/docs`.

### Key Endpoints
| HTTP Method | Route | Description |
|---|---|---|
| `GET` | `/health` | System health check, active model suite, and registered hub count |
| `GET` | `/api/v1/hubs/{hub_id}` | Retrieves structural metadata & competitor distances for a hub |
| `POST` | `/api/v1/forecast` | Real-time single-hub order volume prediction |
| `POST` | `/api/v1/forecast/batch` | Batch multi-hub demand scheduling for fulfillment planning |

### Sample cURL Request: Single-Hub Demand Forecast
```bash
curl -X POST "http://localhost:8000/api/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "hub_id": 1,
    "target_date": "2015-07-01",
    "is_open": 1,
    "promo_active": 1,
    "regional_holiday": 0,
    "school_closure_flag": 0
  }'
```

### Sample Response:
```json
{
  "hub_id": 1,
  "target_date": "2015-07-01",
  "predicted_order_volume": 6885.34,
  "status": "OPERATIONAL",
  "is_open": 1,
  "promo_active": 1,
  "model_version": "1.0.0"
}
```

---

## 🐳 Docker Containerization

The service is fully containerized using a multi-stage Docker build:

### 1. Build & Run with Docker Compose
```bash
# Build and launch microservice in detached mode
docker-compose up --build -d

# Check service logs
docker-compose logs -f

# Verify container health
curl http://localhost:8000/health
```

### 2. Standalone Docker Run
```bash
docker build -t solari-demand-forecasting:latest .
docker run -p 8000:8000 solari-demand-forecasting:latest
```

---

## 🛠️ Developer Makefile

The repository includes a [`Makefile`](file:///a:/Github/Cerebal%20Kaggle/Makefile) automating standard development workflows:

```bash
make help               # Display all available commands
make install            # Install Python dependencies from requirements.txt
make test               # Run all 13 unit tests across metrics, features, & API
make pipeline-test      # Run fast end-to-end integration pipeline on 2% data sample
make pipeline-predict   # Run complete production pipeline across all 1,115 hubs
make run-api            # Launch local FastAPI dev server with hot reload
make docker-build       # Build production Docker container
make docker-up          # Spin up containerized service with Docker Compose
make clean              # Purge temporary cache files and compile artifacts
```

---

## 🧪 Testing Suite & Mathematical Invariants

The project includes **13 unit tests** in [`tests/`](file:///a:/Github/Cerebal%20Kaggle/tests/) verifying mathematical invariants, data integrity, and API schemas:

```bash
python -m unittest discover tests -v
```

```text
test_batch_forecast_endpoint (test_api.TestForecastingAPI) ... OK
test_health_check (test_api.TestForecastingAPI) ... OK
test_predict_closed_hub_invariant (test_api.TestForecastingAPI) ... OK
test_predict_open_hub_returns_positive_volume (test_api.TestForecastingAPI) ... OK
test_root_endpoint (test_api.TestForecastingAPI) ... OK
test_downcast_numeric_dtypes (test_data.TestDataLoader) ... OK
test_day_counter_promo_distance (test_features.TestFeatureEngineering) ... OK
test_rolling_feature_causality_guard (test_features.TestFeatureEngineering) ... OK
test_yoy_lag_weekday_phase_alignment (test_features.TestFeatureEngineering) ... OK
test_jensen_inequality_harmonic_le_arithmetic (test_metrics.TestEvaluationMetrics) ... OK
test_rmsle_clips_negative_predictions (test_metrics.TestEvaluationMetrics) ... OK
test_rmsle_zero_on_exact_match (test_metrics.TestEvaluationMetrics) ... OK
test_yoy_calibrator_enforces_closed_hub_zero (test_metrics.TestEvaluationMetrics) ... OK

Ran 13 tests in 0.914s — OK
```

---

## 📊 Experimental Results & Validation Benchmark

Evaluated using a 5-fold walk-forward **`TimeSeriesSplit`** on operational hub records (`IsOpen == 1`):

| Model Configuration | OOF RMSLE | Error Reduction vs Baseline |
|---|---|---|
| Naive Historical Mean Baseline | `0.44026` | — |
| CatBoost (`rsm = 0.30`) | `0.08538` | -0.35488 |
| May–September Summer Specialist LightGBM | `0.07683` | -0.36343 |
| XGBoost (`colsample_bytree = 0.30`) | `0.07193` | -0.36833 |
| LightGBM (`feature_fraction = 0.30`) | `0.07101` | -0.36925 |
| **Arithmetic Mean Ensemble** | `0.07144` | -0.36882 |
| **Weighted Harmonic Mean Ensemble** | `0.07049` | -0.36977 |
| **Harmonic Ensemble + Log-Bias Multiplier (×1.008)** | **`0.07004`** | **-0.37022 (84.1% reduction)** |

---

## 📚 References & Acknowledgments

* **Gert Jacobusse (Rogatio)**: 1st Place Solution, *Kaggle Rossmann Store Sales Competition* ([Documentation PDF](file:///a:/Github/Cerebal%20Kaggle/docs/Rossmann_nr1_doc.pdf)).
* **Kaggle Rossmann Store Sales Competition**: [Competition Page & Discussion](https://www.kaggle.com/c/rossmann-store-sales).
* **Chen & Guestrin (2016)**: *XGBoost: A Scalable Tree Boosting System*, ACM KDD.
* **Ke et al. (2017)**: *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS.
* **Prokhorenkova et al. (2018)**: *CatBoost: Unbiased boosting with categorical features*, NeurIPS.
