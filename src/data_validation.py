from __future__ import annotations
import pandas as pd

REQUIRED_BASE = {"Time", "Amount", "Class"}

def validate_transactions(df: pd.DataFrame, target: str = "Class") -> None:
    """Raise ValueError for schema or data-quality problems."""
    if df.empty:
        raise ValueError("Dataset is empty.")
    missing = REQUIRED_BASE.union({target}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[target].isna().any():
        raise ValueError(f"Target column '{target}' contains null values.")
    values = set(pd.Series(df[target]).dropna().unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1; found {values}.")
    if df["Amount"].isna().any() or (df["Amount"] < 0).any():
        raise ValueError("Amount must be non-null and non-negative.")
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric feature columns found.")
    if numeric.isna().any().any():
        raise ValueError("Numeric columns contain missing values.")
