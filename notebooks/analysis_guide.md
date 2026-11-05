# Analysis guide (notebook companion)

Use this sequence in a Jupyter notebook when presenting the project.

## 1. Business framing
A card issuer wants to identify fraudulent transactions early while limiting false declines and investigator workload. Fraud is the positive class.

## 2. Data quality
- Inspect shape, dtypes, nulls, duplicates and target values.
- Check `Amount` distribution and fraud prevalence.
- Confirm the data source and document its limitations.

## 3. Exploratory analysis
- Compare transaction amount distributions by class (use log scale).
- Plot class counts on a log axis.
- Inspect correlations, while remembering V1–V28 are anonymised PCA components.
- Compare transaction time patterns cautiously: source `Time` is elapsed seconds, not calendar time.

## 4. Modelling
- Establish a Logistic Regression baseline.
- Compare Random Forest and XGBoost.
- Keep resampling inside the training pipeline.
- Select a threshold on validation data; do not tune it on the test set.

## 5. Evaluation
Report confusion matrix, precision, recall, F1, ROC-AUC and PR-AUC. Explain the cost of false positives and false negatives. Report the test-set fraud prevalence so PR-AUC can be interpreted against the baseline.

## 6. Production discussion
Discuss delayed labels, concept drift, model monitoring, champion/challenger deployment, human review, privacy and auditability.
