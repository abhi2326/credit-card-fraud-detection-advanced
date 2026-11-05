# Credit Card Fraud Detection & Transaction Analytics

An end-to-end fraud analytics project designed to demonstrate practical machine learning, imbalanced classification, model evaluation, explainability, SQL analytics, and lightweight deployment.

> **Data integrity note:** the included demo dataset is synthetic and is intended only to validate the workflow and dashboard. It is not the original card-fraud benchmark and must not be used to claim benchmark performance. For realistic evaluation, download the public `creditcard.csv` dataset described below and run the same pipeline.

## Project capabilities

- Data validation and exploratory analysis
- Stratified train / validation / test workflow
- Leakage-conscious preprocessing
- Logistic Regression, Random Forest and XGBoost comparison
- Optional SMOTE inside the training pipeline only
- Threshold selection using validation-set F1 or a recall-first policy
- Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix and classification report
- SHAP explainability (optional dependency)
- Persisted model, threshold and evaluation report
- Streamlit fraud monitoring dashboard
- FastAPI prediction endpoint
- SQL investigation and monitoring queries
- Unit tests and reproducible demo data generator

## Repository structure

```text
credit-card-fraud-detection-advanced/
├── app.py                      # Streamlit analytics dashboard
├── api.py                      # FastAPI prediction service
├── requirements.txt
├── config.yaml
├── README.md
├── data/
│   ├── raw/                     # Place creditcard.csv here (not committed)
│   └── processed/
├── models/                      # Trained artifacts (generated locally)
├── reports/                     # Metrics and plots (generated locally)
├── notebooks/
│   └── analysis_guide.md        # Guided EDA and modelling workflow
├── sql/
│   ├── 01_fraud_overview.sql
│   ├── 02_high_risk_transactions.sql
│   └── 03_hourly_monitoring.sql
├── src/
│   ├── data_validation.py
│   ├── generate_demo_data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
└── tests/
    ├── test_data_validation.py
    └── test_features.py
```

## 1. Environment setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Get the benchmark dataset

Use the public Kaggle dataset commonly titled **Credit Card Fraud Detection** (European cardholder transactions, September 2013). Download `creditcard.csv` from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Put the file at:

```text
data/raw/creditcard.csv
```

The expected columns are `Time`, `Amount`, `V1` ... `V28`, and `Class` (0 = legitimate, 1 = fraud). The dataset is highly imbalanced. Review the dataset's current licence and terms on its source page before redistribution. Do not commit the raw dataset to GitHub.

## 3. Run with synthetic demo data

```bash
python -m src.generate_demo_data --rows 30000 --output data/raw/demo_transactions.csv
python -m src.train --data data/raw/demo_transactions.csv --target Class --output-dir models --report-dir reports
```

The demo generator creates a deliberately imbalanced, artificial dataset with the same broad schema. It is for smoke-testing only; results are not comparable to real-world fraud detection.

## 4. Train on the benchmark data

```bash
python -m src.train --data data/raw/creditcard.csv --target Class --output-dir models --report-dir reports
```

The training workflow:
1. validates schema and target values;
2. separates train, validation and test sets using stratification;
3. fits preprocessing and optional resampling only on training data;
4. compares Logistic Regression, Random Forest and XGBoost (when installed);
5. chooses a threshold using the validation set;
6. evaluates the selected model once on the held-out test set;
7. saves `fraud_model.joblib`, `metadata.json`, `metrics.json`, and plots.

SMOTE is enabled by default when `imbalanced-learn` is installed. It is applied only inside the training pipeline. For a class-weight-only baseline:

```bash
python -m src.train --data data/raw/creditcard.csv --no-smote
```

For a high-recall operating point, choose `--threshold-objective recall_at_precision` and set `--min-precision 0.80`. Threshold choice is a business decision: higher recall generally increases false positives and investigation workload.

## 5. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard reads the trained artifacts and evaluation report. If no model is trained yet, it displays setup guidance and demo data context.

## 6. Launch the prediction API

```bash
uvicorn api:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 10000, "Amount": 149.99, "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0}'
```

The API returns a probability, decision, and threshold. It accepts either a flat feature object or a `{"features": {...}}` wrapper. This example is only a schema demonstration; the PCA-derived V features are not meaningful when manually set to zero.

## 7. Explainability with SHAP

After training, generate a global feature-attribution summary (the V1–V28 features are anonymised):

```bash
python -m src.explain --data data/raw/creditcard.csv --sample-size 200
```

SHAP explanations can be computationally expensive. Start with a small sample and describe the anonymised components cautiously.

## 8. SQL analytics

The SQL files use a generic `transactions` table with `transaction_id`, `transaction_time`, `amount`, `is_fraud`, `risk_score`, and `merchant_category`. Adapt names and timestamp functions for your warehouse (PostgreSQL syntax is used).

## 9. Testing

```bash
pytest -q
```

## 10. Important modelling considerations

- **Accuracy is misleading** for highly imbalanced fraud data. A model predicting every transaction as legitimate can achieve high accuracy while detecting no fraud.
- **PR-AUC** is especially informative when the positive class is rare.
- **Threshold 0.50 is not sacred.** Select an operating threshold based on fraud loss, false-decline cost, analyst capacity and customer experience.
- **SMOTE must never be applied before splitting.** Doing so leaks information from validation/test observations into training.
- **Random stratified splits** reproduce a common benchmark setup, but production evaluation should use a time-based split to reflect future transactions and concept drift.
- The dataset's `Time` is elapsed seconds in the source dataset, not a wall-clock timestamp. Do not interpret it as a real date/time without an external mapping.
- The V1–V28 features are anonymised PCA components. Their business meaning is not available; feature-level explanations should be described carefully.
- This project is a portfolio demonstration, not a deployable banking decision system. Real deployment requires privacy, security, fairness, model governance, monitoring, human review, and regulatory controls.

## Suggested resume bullet (use only after running and verifying)

"Built an end-to-end credit-card fraud detection pipeline on a highly imbalanced transaction dataset, comparing Logistic Regression, Random Forest and XGBoost with leakage-safe resampling; evaluated precision, recall and PR-AUC, optimised the decision threshold, and deployed risk scoring through a Streamlit dashboard and FastAPI."

Do not claim a specific precision/recall figure until you have reproduced it on the held-out test set and recorded the dataset version and split strategy.
