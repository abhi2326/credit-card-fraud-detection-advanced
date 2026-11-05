from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for a trained fraud model.")
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--output", default="reports/shap_summary.png")
    args = parser.parse_args()

    try:
        import shap
    except ImportError as exc:
        raise SystemExit("Install SHAP first: pip install shap") from exc

    model = joblib.load("models/fraud_model.joblib")
    meta = __import__("json").loads(Path("models/metadata.json").read_text())
    df = pd.read_csv(args.data)
    X = df[meta["feature_columns"]].sample(min(args.sample_size, len(df)), random_state=42)

    # Model-agnostic explainer works with the full preprocessing + classifier pipeline.
    # Keep sample size modest because Kernel/Permutation-style explanations can be costly.
    background = X.sample(min(50, len(X)), random_state=7)
    explainer = shap.Explainer(model.predict_proba, background)
    values = explainer(X)
    # Explain the positive (fraud) class for binary classifiers.
    if len(values.values.shape) == 3:
        values = shap.Explanation(
            values=values.values[:, :, 1],
            base_values=values.base_values[:, 1] if len(values.base_values.shape) > 1 else values.base_values,
            data=values.data,
            feature_names=values.feature_names
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"Saved SHAP summary to {args.output}")

if __name__ == "__main__":
    main()
