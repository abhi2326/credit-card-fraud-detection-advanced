from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def generate_demo(rows: int = 30000, fraud_rate: float = 0.012, seed: int = 42):
    """Generate synthetic schema-compatible data for workflow testing only."""
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < fraud_rate).astype(int)
    time = np.sort(rng.integers(0, 172800, rows))
    amount = np.where(y == 1,
                      rng.lognormal(mean=4.0, sigma=1.0, size=rows),
                      rng.lognormal(mean=3.2, sigma=1.1, size=rows))
    data = {"Time": time.astype(float), "Amount": np.round(amount, 2)}
    for i in range(1, 29):
        # Artificial latent shifts; these are not PCA features from the real dataset.
        base = rng.normal(0, 1, rows)
        shift = rng.normal(0.0, 1.8, rows) * y
        data[f"V{i}"] = base + shift
    data["Class"] = y
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--output", default="data/raw/demo_transactions.csv")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_demo(args.rows, args.fraud_rate)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} synthetic rows to {out}; fraud rate={df.Class.mean():.3%}")
