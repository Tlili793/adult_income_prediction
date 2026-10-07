# Adult Census Income — Income Prediction (binary classification)

This project builds a machine-learning pipeline to predict whether an individual's
annual income exceeds **$50,000** based on demographic, professional, and financial
features from the **UCI Adult dataset**. The primary artifact is the Jupyter
notebook `adult (7).ipynb`, which walks through the full workflow from data
understanding to model training, evaluation, and saving.

## Machine Learning Problem

Binary classification. Given a person's features (age, education, occupation,
hours worked per week, marital status, and so on), the model predicts the target
`income`:

- `<=50K` — annual income **≤** $50,000
- `>50K` — annual income **>** $50,000

### Project Objectives

- **Predictive Modeling** — Build and evaluate classification algorithms capable of
  accurately forecasting income levels.
- **Explanatory Analysis** — Highlight the major socio-economic factors that heavily
  impact remuneration levels.
- **Data Quality Assurance** — Rigorously handle anomalies, missing values (such as
  `?` markers), and text inconsistencies to ensure a reliable foundation for modeling.

### Key Questions Addressed During EDA

1. **Target distribution** — What is the degree of class imbalance between `<=50K`
   and `>50K`?
2. **Impact of education and age** — What role do `education` / `education-num` and
   `age` play in accessing higher income brackets?
3. **Role of work hours** — Is `hours-per-week` a decisive factor in crossing the
   $50K threshold?
4. **Demographic and societal factors** — Are there income disparities based on
   gender (`sex`), `marital-status`, or `native-country`?
5. **Collinearity and redundancy** — Do some features measure similar information
   (e.g., text `education` labels vs. `education-num`) that should be streamlined?

## Dataset

The data is loaded with `ucimlrepo` (UCI ID `2`) and assembled into a single
DataFrame by concatenating features and the target.

- **Rows:** 48,842
- **Columns:** 15 (14 features + target `income`)

| # | Feature         | Type   | Description                                         |
|---|-----------------|--------|-----------------------------------------------------|
| 0 | `age`           | int64  | Age of the individual                                |
| 1 | `workclass`     | object | Employment type (Private, State-gov, ...)            |
| 2 | `fnlwgt`        | int64  | Final sampling weight                                |
| 3 | `education`     | object | Highest education level (HS-grad, Bachelors, ...)    |
| 4 | `education-num` | int64  | Years / level of education (numeric)                 |
| 5 | `marital-status`| object | Marital status                                       |
| 6 | `occupation`    | object | Job category                                         |
| 7 | `relationship`  | object | Relation to household (Husband, Wife, ...)           |
| 8 | `race`          | object | Race                                                 |
| 9 | `sex`           | object | Gender                                              |
| 10| `capital-gain`  | int64  | Capital gains (highly right-skewed)                  |
| 11| `capital-loss`  | int64  | Capital losses (highly right-skewed)                 |
| 12| `hours-per-week`| int64  | Weekly working hours                                 |
| 13| `native-country`| object | Country of origin (42 unique values, mostly US)      |
| 14| `income`        | object | **Target** — `<=50K` / `>50K`                        |

**Notable points from data understanding:**

- 6 numerical (int64) columns and 9 categorical (object) columns.
- `workclass`, `occupation`, and `native-country` contain `NaN` counts that hide
  `?` placeholder tokens.
- `native-country` has 42 unique values — a broad international representation that
  needs grouping / encoding to avoid sparse one-hot columns.
- The `income` target initially shows inconsistent formatting (e.g., `>50K.` vs
  `>50K`).

## Workflow

The notebook follows a clear pipeline:

1. **Data Understanding** — dimensions, dtypes, missing values, descriptive
   statistics, unique-value audit.
2. **Data Quality Analysis** — duplicate removal, placeholder detection (`?`),
   anomaly and range audits.
3. **Target Analysis** — class distribution and imbalance assessment.
4. **Univariate Analysis** — distributions, box plots, outliers, skewness/kurtosis
   (numerical), concentration/entropy (categorical).
5. **Bivariate Analysis** — Pearson correlation (numerical-numerical),
   Cramér's V (categorical-categorical), Eta (numerical-categorical), and a
   feature-removal plan based on redundancy findings.
6. **Cleaning** — placeholder → `NaN`, whitespace stripping, target normalization.
7. **Feature Engineering** — net-capital signed log transformation and engineered
   feature sets.
8. **Training** — supervised classification + unsupervised segmentation (KMeans)
   + a PCA-based dimension-reduction branch.
9. **Evaluation & Saving** — cross-validation comparison, feature-variant tests,
   and serializing the best model.

## Key EDA Findings

- **Class imbalance:** `<=50K` has 37,113 records (76.1%); `>50K` has 11,681
  records (23.9%). (Raw data contains 48,842 rows total.)
- **Age** is roughly right-skewed and unimodal with peak density in the working-age
  range.
- **`capital-gain` / `capital-loss`** are extremely right-skewed with severe
  outliers — handled with a signed-log (`net-capital-log`) transformation.
- **Collinearity:** `education` vs `education-num` capture the same information;
  categorical association analysis (Cramér's V) finds strong links between
  `relationship` and `marital-status` (and `sex`), motivating the redundancy
  evaluation.
- **Hours-per-week** centers around 40 hours (median 40).

## Data Preparation

### Cleaning

`clean_adult_data()`:

1. Standardizes missing-value placeholders (`' ?'` / `'?'`) to `NaN`.
2. Strips whitespace from all string/object columns.
3. Cleans the target variable (removes inconsistent formatting).

### Feature Engineering

`engineer_adult_features()`:

1. **Net Capital & Signed Log Transformation** — builds `net-capital-log`
   (`sign(net) * log1p(|net|)`), then drops the raw `capital-gain`/`capital-loss`
   columns.
2. Additional engineered variants (e.g., `capital-gain-log`) are used across
   feature-set experiments.

## Modeling

### Preprocessing & Workflow

Scikit-learn `Pipeline` + `ColumnTransformer`:

- Numeric features → `StandardScaler`
- Categorical features → `OneHotEncoder`
- Classifiers trained with `StratifiedKFold` cross-validation
- Metrics: **ROC-AUC**, **Accuracy**, **F1-score**

### Models Evaluated

- Logistic Regression (`class_weight='balanced'`)
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- **XGBoost** (`eval_metric='logloss'`, `enable_categorical=True`)

### Model Comparison (mean CV scores)

| Model               | Mean ROC-AUC | Mean Accuracy | Mean F1 |
|:--------------------|-------------:|--------------:|--------:|
| **XGBoost**         |     **0.9292** |      **0.8737** |  **0.7093** |
| Gradient Boosting   |       0.9217 |        0.8677 |   0.6870 |
| Logistic Regression |       0.8937 |        0.7956 |   0.6658 |
| Random Forest       |       0.8888 |        0.8375 |   0.6547 |

> A reference hold-out Random Forest run reported ROC-AUC ≈ 0.8886 with ~84%
> accuracy.

### Feature Redundancy Evaluation (XGBoost)

Variants test dropping highly-associated features:

| Variant                         | Num Features | Mean ROC-AUC | Mean Accuracy | Mean F1 |
|:--------------------------------|-------------:|-------------:|--------------:|--------:|
| Model A (All Features)          |   13         |  **0.9292** |    **0.8737** |  **0.7093** |
| Model B (Remove relationship)   |      12      |       0.9291 |          0.8743 |   0.7101 |
| Model D (Remove sex)            |      12      |       0.9288 |          0.8735 |   0.7083 |
| Model C (Remove marital-status) |      12      |       0.9285 |          0.8733 |   0.7082 |

Keeping all features (**Model A**) performs best while remaining near-identical to
the most competitive reduced variant (Model B).

### Unsupervised & PCA Branches

- **KMeans clustering** is used to build a segmentation-based feature set
  (validated with silhouette score) that is fed into downstream classifiers.
- **PCA** is applied to the pure continuous features (`age`, `education-num`,
  `hours-per-week`, capital features) to test a dimension-reduced modeling branch.

## Results & Artifacts

- **Best model:** XGBoost on the full feature set — ROC-AUC ≈ **0.93**, accuracy
  ≈ **0.87**, F1 ≈ **0.71**.
- **Saved artifacts:**
  - `best_adult_xgboost_model.pkl` — trained XGBoost pipeline (Model A, all
    features).
  - `best_adult_model.pkl` — additional saved pipeline.
- Models are serialized with `joblib` and can be reloaded for inference.

## Requirements

- Python 3.x
- `ucimlrepo`
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`, `seaborn`
- `scipy`
- `joblib`

Install with:

```bash
pip install ucimlrepo pandas numpy scikit-learn xgboost matplotlib seaborn scipy joblib
```

## How to Run

Open `adult (7).ipynb` in Jupyter (or Colab) and run cells top-to-bottom:

1. The `ucimlrepo` fetch loads the dataset (an internet connection is needed).
2. Executing the notebook reproduces the EDA, preprocessing, feature engineering,
   model comparison, and model-saving steps.

## Repository Structure

```
jupyter/
├── adult (7).ipynb              # Main project notebook (EDA → modeling)
├── adult_3_clear_structure.ipynb# Restructured variant following KDD phases
├── best_adult_model.pkl         # Saved pipeline
├── best_adult_xgboost_model.pkl # Best XGBoost pipeline (Model A)
├── garbage/                     # Archived/scratch notebooks & CSV exports
└── pca/                         # Separate PCA (ACP) cybersecurity notebook
```