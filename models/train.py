"""
train.py
--------
Trains an XGBoost anomaly detector on the engineered telecom KPI features.

Phase 1 additions (MLflow + DVC):
  - Hyperparameters now come from params.yaml (single source of truth, also
    a DVC pipeline dependency — changing a param triggers `dvc repro` to
    retrain).
  - Every run is logged to MLflow: params, metrics, the trained model
    (in both native XGBoost format and as an MLflow pyfunc), the fitted
    scaler, and metadata.json as artifacts.
  - On success, the model is registered in the MLflow Model Registry under
    the name in params.yaml (mlflow.registered_model_name), so `api/main.py`
    can eventually load "models:/telecom-anomaly-xgb/Production" instead of
    a local file path (wired up in a later phase alongside CI/CD gating).
  - models/model.json, models/scaler.pkl, models/metadata.json are still
    written locally too, so the existing FastAPI service keeps working
    unchanged in the meantime — MLflow is additive, not a breaking change.

Design decisions (unchanged from the original):
  - Time-based train/test split (no shuffle) — respects temporal ordering.
  - Class imbalance handled via scale_pos_weight.
  - Threshold tuned for a recall floor, not raw accuracy.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Allow imports from parent dir
sys.path.append(str(Path(__file__).parent.parent))
from data.features import get_feature_columns
from models.common import DATA_DIR, MODEL_DIR, load_params, load_data, time_split, tune_threshold


def main():
    print("=" * 60)
    print("  Telecom Anomaly Detection — Model Training")
    print("=" * 60)

    params = load_params()
    model_params     = params["model"]
    split_params     = params["split"]
    threshold_params = params["threshold"]
    mlflow_params    = params["mlflow"]

    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run() as run:
        print(f"\nMLflow run ID: {run.info.run_id}")

        # ── Load data ─────────────────────────────────────────────────────
        df = load_data()
        feature_cols = get_feature_columns(df)
        print(f"\nLoaded {len(df):,} rows | {len(feature_cols)} features")

        train_df, test_df = time_split(df, split_params["train_ratio"])
        print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

        X_train = train_df[feature_cols].values
        y_train = train_df["anomaly"].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df["anomaly"].values

        train_anomaly_rate = float(y_train.mean())
        test_anomaly_rate  = float(y_test.mean())
        print(f"\nTrain anomaly rate: {train_anomaly_rate:.2%}")
        print(f"Test  anomaly rate: {test_anomaly_rate:.2%}")

        # ── Scale features ───────────────────────────────────────────────
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ── Handle class imbalance ───────────────────────────────────────
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        print(f"\nClass ratio (neg/pos): {scale_pos_weight:.1f}x → used as scale_pos_weight")

        # ── Log params to MLflow ─────────────────────────────────────────
        mlflow.log_params({
            **model_params,
            "scale_pos_weight": scale_pos_weight,
            "train_ratio": split_params["train_ratio"],
            "min_recall_floor": threshold_params["min_recall"],
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "n_features": len(feature_cols),
            "train_anomaly_rate": train_anomaly_rate,
            "test_anomaly_rate": test_anomaly_rate,
        })

        # ── Train XGBoost ────────────────────────────────────────────────
        print("\nTraining XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            subsample=model_params["subsample"],
            colsample_bytree=model_params["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="aucpr",
            early_stopping_rounds=model_params["early_stopping_rounds"],
            random_state=model_params["random_state"],
            n_jobs=-1,
        )

        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=50,
        )

        mlflow.log_metric("best_iteration", model.best_iteration)

        # ── Threshold tuning ─────────────────────────────────────────────
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        best_threshold, best_recall, best_precision, best_f1 = tune_threshold(
            y_test, y_prob, threshold_params["min_recall"]
        )
        print(f"\nThreshold tuned for recall >= {threshold_params['min_recall']}:")
        print(f"  Threshold : {best_threshold:.3f}")
        print(f"  Recall    : {best_recall:.3f}")
        print(f"  Precision : {best_precision:.3f}")
        print(f"  F1        : {best_f1:.3f}")

        y_pred = (y_prob >= best_threshold).astype(int)

        # ── Evaluation ───────────────────────────────────────────────────
        print("\n── Classification Report ──────────────────────────────────")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall    = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"Confusion Matrix:\n{cm}")
        print(f"\nRecall    : {recall:.3f}  ← % of real anomalies caught")
        print(f"Precision : {precision:.3f}  ← % of flagged alerts that are real")

        mlflow.log_metrics({
            "threshold": float(best_threshold),
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(best_f1), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        })

        # ── Feature importance ───────────────────────────────────────────
        importance = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        print("\n── Top 10 Features ────────────────────────────────────────")
        print(importance.head(10).to_string())

        importance_path = MODEL_DIR / "feature_importance.csv"
        importance.to_csv(importance_path, header=["importance"])
        mlflow.log_artifact(str(importance_path))

        # ── Save artifacts locally (kept for the existing FastAPI service) ─
        MODEL_DIR.mkdir(exist_ok=True)

        model.save_model(MODEL_DIR / "model.json")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

        metadata = {
            "feature_cols": feature_cols,
            "threshold": float(best_threshold),
            "metrics": {
                "recall":    round(float(recall), 4),
                "precision": round(float(precision), 4),
                "f1":        round(float(best_f1), 4),
            },
            "train_rows": len(train_df),
            "test_rows":  len(test_df),
            "mlflow_run_id": run.info.run_id,
        }
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # ── Log model + scaler + metadata to MLflow ─────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=mlflow_params["registered_model_name"],
        )
        mlflow.log_artifact(str(MODEL_DIR / "scaler.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "metadata.json"))

        print(f"\nSaved → {MODEL_DIR}/model.json")
        print(f"Saved → {MODEL_DIR}/scaler.pkl")
        print(f"Saved → {MODEL_DIR}/metadata.json")
        print(f"Logged to MLflow run {run.info.run_id} "
              f"(registered as '{mlflow_params['registered_model_name']}')")
        print("\nTraining complete.")


if __name__ == "__main__":
    main()
