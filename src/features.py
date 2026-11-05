from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def split_features_target(df: pd.DataFrame, target: str = "Class"):
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale continuous fields; pass through the remaining numeric features."""
    scale_cols = [c for c in ["Time", "Amount"] if c in X.columns]
    other_cols = [c for c in X.columns if c not in scale_cols]
    transformers = []
    if scale_cols:
        transformers.append(("scaled", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), scale_cols))
    if other_cols:
        transformers.append(("other", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), other_cols))
    return ColumnTransformer(transformers, remainder="drop")
