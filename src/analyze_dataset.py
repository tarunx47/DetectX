
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Synthetic_Financial_datasets_log.csv"
MODEL = ROOT / "models" / "best_model_pipeline.joblib"
OUTPUT = ROOT / "reports" / "dataset_analysis.json"

FEATURES = [
    "step", "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]

def main():
    print("Loading model...")
    model = joblib.load(MODEL)

    stats = {
        "total_transactions": 0,
        "actual_fraud": 0,
        "actual_legitimate": 0,
        "predicted_fraud": 0,
        "predicted_legitimate": 0,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "by_type": {},
    }

    print("Processing CSV in batches...")
    for number, df in enumerate(
        pd.read_csv(DATA, chunksize=50000), start=1
    ):
        required = FEATURES + ["isFraud"]
        df = df.dropna(subset=required)
        if df.empty:
            continue

        y = df["isFraud"].astype(int).to_numpy()
        pred = model.predict(df[FEATURES]).astype(int)

        actual_fraud = y == 1
        predicted_fraud = pred == 1

        stats["total_transactions"] += len(df)
        stats["actual_fraud"] += int(actual_fraud.sum())
        stats["actual_legitimate"] += int((~actual_fraud).sum())
        stats["predicted_fraud"] += int(predicted_fraud.sum())
        stats["predicted_legitimate"] += int((~predicted_fraud).sum())
        stats["true_positive"] += int((actual_fraud & predicted_fraud).sum())
        stats["false_positive"] += int((~actual_fraud & predicted_fraud).sum())
        stats["true_negative"] += int((~actual_fraud & ~predicted_fraud).sum())
        stats["false_negative"] += int((actual_fraud & ~predicted_fraud).sum())

        for kind, group in df.assign(prediction=pred).groupby("type"):
            item = stats["by_type"].setdefault(
                str(kind),
                {"transactions": 0, "actual_fraud": 0, "predicted_fraud": 0},
            )
            item["transactions"] += len(group)
            item["actual_fraud"] += int(group["isFraud"].sum())
            item["predicted_fraud"] += int((group["prediction"] == 1).sum())

        print(f"Batch {number}: {stats['total_transactions']:,} rows")

    n = stats["total_transactions"]
    if not n:
        raise RuntimeError("No valid dataset rows were processed.")

    tp = stats["true_positive"]
    fp = stats["false_positive"]
    fn = stats["false_negative"]
    tn = stats["true_negative"]

    stats["actual_fraud_rate"] = round(
        stats["actual_fraud"] / n * 100, 3
    )
    stats["predicted_fraud_rate"] = round(
        stats["predicted_fraud"] / n * 100, 3
    )
    stats["evaluation"] = {
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "accuracy": (tp + tn) / n,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nAnalysis complete!")
    print(f"Transactions: {n:,}")
    print(f"Actual fraud: {stats['actual_fraud']:,}")
    print(f"Predicted fraud: {stats['predicted_fraud']:,}")
    print(f"Report: {OUTPUT}")

if __name__ == "__main__":
    main()
