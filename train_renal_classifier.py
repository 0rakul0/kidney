import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


FEATURES_PATH = os.path.join("results", "renal_feature_analysis", "renal_features.csv")
LABELS_PATH = os.path.join("results", "renal_feature_analysis", "renal_labels.csv")
OUTPUT_DIR = os.path.join("results", "renal_classifier")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    features_df = pd.read_csv(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)

    labels_df["label"] = pd.to_numeric(labels_df["label"], errors="coerce")
    labels_df = labels_df.dropna(subset=["label"]).copy()
    labels_df["label"] = labels_df["label"].astype(int)

    df = features_df.merge(labels_df, on=["split", "image_name"], how="inner")
    return df


def select_feature_columns(df):
    excluded = {"split", "image_name", "label", "label_name", "reference_source"}
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_columns if col not in excluded]


def train_and_evaluate(df):
    if df.empty:
        raise ValueError("No labeled samples found. Fill in results/renal_feature_analysis/renal_labels.csv first.")

    train_df = df[df["split"].isin(["train", "val"])].copy()
    test_df = df[df["split"] == "test"].copy()
    all_labeled_df = df.copy()

    train_label_counts = train_df["label"].value_counts().to_dict()
    if train_df["label"].nunique() < 2:
        raise ValueError(
            "Training labels need at least two classes: 0 and 1. "
            f"Current labeled counts in train/val: {train_label_counts}"
        )

    feature_columns = select_feature_columns(df)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Preferred mode: holdout evaluation when test labels exist.
    if not train_df.empty and not test_df.empty:
        X_train = train_df[feature_columns]
        y_train = train_df["label"]
        X_test = test_df[feature_columns]
        y_test = test_df["label"]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None

        metrics = {
            "evaluation_mode": "holdout_test_split",
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        if y_prob is not None and len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred).tolist()

        classifier = model.named_steps["classifier"]
        importances = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": classifier.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        predictions = test_df[["split", "image_name", "label"]].copy()
        predictions["predicted_label"] = y_pred
        if y_prob is not None:
            predictions["prob_diseased"] = y_prob

        return metrics, report, matrix, importances, predictions

    # Fallback mode: cross-validation over all labeled samples.
    if all_labeled_df["label"].nunique() < 2:
        raise ValueError(
            "Need both classes 0 and 1 among labeled samples. "
            f"Current labeled counts: {all_labeled_df['label'].value_counts().to_dict()}"
        )

    min_class_count = all_labeled_df["label"].value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            "Need at least 2 labeled samples in each class for cross-validation. "
            f"Current labeled counts: {all_labeled_df['label'].value_counts().to_dict()}"
        )

    n_splits = min(5, int(min_class_count))
    X_all = all_labeled_df[feature_columns]
    y_all = all_labeled_df["label"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_all, y_all, cv=cv, method="predict")
    y_prob = cross_val_predict(model, X_all, y_all, cv=cv, method="predict_proba")[:, 1]

    model.fit(X_all, y_all)
    classifier = model.named_steps["classifier"]

    metrics = {
        "evaluation_mode": "cross_validation",
        "cv_folds": int(n_splits),
        "labeled_samples": int(len(all_labeled_df)),
        "accuracy": float(accuracy_score(y_all, y_pred)),
        "precision": float(precision_score(y_all, y_pred, zero_division=0)),
        "recall": float(recall_score(y_all, y_pred, zero_division=0)),
        "f1": float(f1_score(y_all, y_pred, zero_division=0)),
    }

    if len(np.unique(y_all)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_all, y_prob))

    report = classification_report(y_all, y_pred, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_all, y_pred).tolist()

    importances = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    predictions = all_labeled_df[["split", "image_name", "label"]].copy()
    predictions["predicted_label"] = y_pred
    predictions["prob_diseased"] = y_prob

    return metrics, report, matrix, importances, predictions


def main():
    df = load_data()
    metrics, report, matrix, importances, predictions = train_and_evaluate(df)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    report_path = os.path.join(OUTPUT_DIR, "classification_report.json")
    matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.json")
    importance_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    predictions_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)

    importances.to_csv(importance_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("Metrics saved:", metrics_path)
    print("Classification report saved:", report_path)
    print("Confusion matrix saved:", matrix_path)
    print("Feature importance saved:", importance_path)
    print("Predictions saved:", predictions_path)


if __name__ == "__main__":
    main()
