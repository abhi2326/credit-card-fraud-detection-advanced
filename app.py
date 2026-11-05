from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Fraud Intelligence", page_icon="🛡️", layout="wide")
st.title("Credit Card Fraud — Intelligence & Monitoring")
st.caption("Portfolio demonstration | Use benchmark data for meaningful model evaluation.")

metrics_path = Path("reports/metrics.json")
metadata_path = Path("models/metadata.json")
model_path = Path("models/fraud_model.joblib")

if metrics_path.exists():
    report = json.loads(metrics_path.read_text())
    m = report["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{m['precision']:.1%}")
    c2.metric("Recall", f"{m['recall']:.1%}")
    c3.metric("F1 score", f"{m['f1']:.3f}")
    c4.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    st.subheader("Held-out test evaluation")
    cm = pd.DataFrame(m["confusion_matrix"],
                      index=["Actual legitimate", "Actual fraud"],
                      columns=["Predicted legitimate", "Predicted fraud"])
    st.dataframe(cm, use_container_width=True)
    st.caption(f"Selected model: {report['best_model']} | Test fraud prevalence: {report['test_fraud_prevalence']:.3%}")
    st.subheader("Validation model comparison")
    comp = pd.DataFrame(report["validation_comparison"]).T.reset_index(names="Model")
    st.dataframe(comp[["Model", "precision", "recall", "f1", "pr_auc", "roc_auc"]], use_container_width=True)
    if Path("reports/confusion_matrix.png").exists():
        st.image("reports/confusion_matrix.png", caption="Held-out test confusion matrix")
else:
    st.info("No trained evaluation found. Follow README steps to generate demo data and train models.")

st.divider()
st.subheader("Transaction risk scoring")
if model_path.exists():
    try:
        import joblib
        meta = json.loads(metadata_path.read_text())
        uploaded = st.file_uploader("Upload a CSV with model feature columns (no target required)", type=["csv"])
        if uploaded:
            from src.predict import score_transactions
            data = pd.read_csv(uploaded)
            scored = score_transactions(data)
            st.dataframe(pd.concat([data.reset_index(drop=True), scored], axis=1), use_container_width=True)
            st.download_button("Download scored transactions", scored.to_csv(index=False), "fraud_scores.csv", "text/csv")
    except Exception as exc:
        st.error(f"Scoring unavailable: {exc}")
else:
    st.caption("Train a model first to enable transaction scoring.")
