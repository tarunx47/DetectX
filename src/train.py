"""
DetectX Model Training and Evaluation Pipeline.

Orchestrates:
1. Low-memory loading of transaction data with strict typing.
2. Stratified train/test split (80/20) preserving untouched natural class prevalence in the test set.
3. Representative training sampling: retains 100% of training frauds while sampling legitimate transactions (20:1 ratio).
4. Preprocessing pipelines (OneHotEncoder + StandardScaler).
5. Fitting 4 models: DummyClassifier, LogisticRegression, DecisionTree, RandomForest.
6. Evaluating metrics: PR-AUC, F1-Score, Recall, Precision, ROC-AUC, Confusion Matrix.
7. Saving comparison artifacts to reports/ and the top model pipeline to models/.
"""

import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import joblib
import pandas as pd
import psutil
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
from pathlib import Path

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_model, generate_readable_summary
from features import (
    CSV_DTYPES,
    TARGET_COLUMN,
    FEATURE_DOCUMENTATION,
    get_preprocessor,
)


def print_step(title: str) -> None:
    """Prints a styled section header for readability."""
    print("\n" + "=" * 70)
    print(f"[*] {title}")
    print("=" * 70)


def check_system_resources() -> Dict[str, float]:
    """Inspects available RAM and CPU cores to ensure safe model training."""
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"System Check -> Cores: {cpu_cores} | Total RAM: {total_gb:.2f} GB | Available RAM: {available_gb:.2f} GB")
    return {"total_gb": total_gb, "available_gb": available_gb, "cores": cpu_cores}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Loads necessary columns with optimized data types to keep memory usage under 200 MB.
    Excludes identifiers ('nameOrig', 'nameDest') and existing rule flag ('isFlaggedFraud').
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")

    print(f"Loading dataset from: {csv_path}")
    cols_to_load = list(CSV_DTYPES.keys())
    print(f"Columns to load ({len(cols_to_load)}): {cols_to_load}")
    print("Excluded initially: ['nameOrig', 'nameDest'] (high cardinality), 'isFlaggedFraud' (heuristic flag)")

    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=cols_to_load, dtype=CSV_DTYPES)
    elapsed = time.time() - t0

    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Successfully loaded {len(df):,} rows in {elapsed:.2f} seconds.")
    print(f"DataFrame memory footprint: {mem_usage_mb:.2f} MB")
    return df


def prepare_data_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    legit_to_fraud_ratio: int = 20,
    random_state: int = 42,
) -> tuple:
    """
    Creates a stratified split where:
    - Test set remains completely untouched with natural class prevalence (~0.129% fraud).
    - Training set retains 100% of available fraud samples, while legitimate samples are
      reproducibly sampled to avoid out-of-memory bottlenecks.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"\nPerforming stratified train/test split (Test size = {test_size * 100:.0f}%)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    total_test = len(X_test)
    test_frauds = int(y_test.sum())
    test_fraud_pct = (test_frauds / total_test) * 100
    print(f"  - Test Set: {total_test:,} samples (Frauds: {test_frauds:,} -> {test_fraud_pct:.4f}%) [UNTOUCHED]")

    # Subsample training partition to stay safely within memory limits
    fraud_indices = y_train_full[y_train_full == 1].index
    n_frauds = len(fraud_indices)
    n_legit_sample = min(n_frauds * legit_to_fraud_ratio, int((y_train_full == 0).sum()))

    print(f"\nPreparing representative training subset:")
    print(f"  - Total training partition available: {len(X_train_full):,} rows ({n_frauds:,} frauds)")
    print(f"  - Retaining ALL {n_frauds:,} fraud cases (100% recall of training fraud)")
    print(f"  - Sampling {n_legit_sample:,} legitimate cases (Ratio: {legit_to_fraud_ratio}:1 legit-to-fraud)")

    non_fraud_indices = (
        y_train_full[y_train_full == 0]
        .sample(n=n_legit_sample, random_state=random_state)
        .index
    )

    train_subset_indices = fraud_indices.union(non_fraud_indices)
    X_train = X_train_full.loc[train_subset_indices].copy()
    y_train = y_train_full.loc[train_subset_indices].copy()

    # Free memory of full training set
    del X_train_full, y_train_full

    print(f"  - Final training set size: {len(X_train):,} samples (Frauds: {y_train.sum():,})")
    return X_train, y_train, X_test, y_test


def build_models() -> Dict[str, Any]:
    """Instantiates the 4 candidate models with fixed random states for reproducibility."""
    return {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=60,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=2,
        ),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "Synthetic_Financial_datasets_log.csv"
    reports_dir = project_root / "reports"
    models_dir = project_root / "models"

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print_step("Phase 1: Environment & Resource Inspection")
    resources = check_system_resources()

    print_step("Phase 2: Data Loading & Memory Optimization")
    df = load_dataset(data_path)

    print_step("Phase 3: Stratified Splitting & Partition Sampling")
    X_train, y_train, X_test, y_test = prepare_data_splits(
        df, test_size=0.20, legit_to_fraud_ratio=20, random_state=42
    )

    # Free the original loaded DataFrame now that splits are made
    del df

    print_step("Phase 4: Pipeline Construction & Model Training")
    candidate_models = build_models()
    evaluation_results: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    preprocessor = get_preprocessor()

    for model_name, classifier in candidate_models.items():
        print(f"\n[+] Fitting {model_name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        t_fit_start = time.time()
        pipeline.fit(X_train, y_train)
        fit_duration = time.time() - t_fit_start
        print(f"    Fit completed in {fit_duration:.2f}s.")

        print(f"    Evaluating {model_name} on untouched test set ({len(X_test):,} rows)...")
        eval_metrics = evaluate_model(
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            threshold=0.5,
        )
        eval_metrics["fit_time_sec"] = fit_duration

        print(f"    Results -> PR-AUC: {eval_metrics['pr_auc']:.4f} | "
              f"F1: {eval_metrics['f1_score']:.4f} | "
              f"Recall: {eval_metrics['recall']:.4f} | "
              f"Precision: {eval_metrics['precision']:.4f} | "
              f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
        print(f"    Confusion Matrix -> TN: {eval_metrics['true_negatives']:,}, "
              f"FP: {eval_metrics['false_positives']:,}, "
              f"FN: {eval_metrics['false_negatives']:,}, "
              f"TP: {eval_metrics['true_positives']:,}")

        evaluation_results.append(eval_metrics)
        fitted_pipelines[model_name] = pipeline

    print_step("Phase 5: Generating Reports & Saving Evaluation Results")
    results_df = pd.DataFrame(evaluation_results)
    csv_report_path = reports_dir / "model_comparison.csv"
    results_df.to_csv(csv_report_path, index=False)
    print(f"Evaluation metrics saved to: {csv_report_path}")

    # Build and save readable summary
    metadata = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": TARGET_COLUMN,
        "test_rows": len(X_test),
        "test_fraud_prevalence": f"{(y_test.sum() / len(y_test) * 100):.4f}% ({int(y_test.sum()):,} frauds)",
        "train_rows": len(X_train),
        "train_fraud_count": int(y_train.sum()),
        "sample_ratio": "20:1 legitimate-to-fraud",
    }
    summary_text = generate_readable_summary(evaluation_results, metadata)
    summary_report_path = reports_dir / "model_comparison.txt"
    summary_report_path.write_text(summary_text, encoding="utf-8")
    print(f"Readable summary report saved to: {summary_report_path}")

    print_step("Phase 6: Best Model Selection & Persistence")
    # Best model selected strictly based on non-baseline performance (PR-AUC as primary for extreme imbalance)
    learned_results = [r for r in evaluation_results if r["model_name"] != "DummyClassifier"]
    best_result = max(learned_results, key=lambda x: (x["pr_auc"], x["f1_score"]))
    best_model_name = best_result["model_name"]
    best_pipeline = fitted_pipelines[best_model_name]

    model_save_path = models_dir / "best_model_pipeline.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Selected Best Model: {best_model_name} (PR-AUC: {best_result['pr_auc']:.4f}, F1: {best_result['f1_score']:.4f})")
    print(f"Saved complete pipeline to: {model_save_path}")

    # Save model metadata
    metadata_save_path = models_dir / "model_metadata.json"
    metadata_content = {
        "best_model": best_model_name,
        "metrics": best_result,
        "feature_schema": {
            "numeric": preprocessor.transformers[0][2],
            "categorical": preprocessor.transformers[1][2],
        },
        "all_model_results": evaluation_results,
        "notes": "Trained on representative 20:1 train sample, evaluated on untouched natural test set.",
    }
    metadata_save_path.write_text(json.dumps(metadata_content, indent=2), encoding="utf-8")
    print(f"Saved model metadata to: {metadata_save_path}")

    print_step("Completed Successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL ERROR] Training pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
