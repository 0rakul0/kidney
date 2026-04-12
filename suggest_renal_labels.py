import os
import numpy as np
import pandas as pd


FEATURES_PATH = os.path.join("results", "renal_feature_analysis", "renal_features.csv")
LABELS_PATH = os.path.join("results", "renal_feature_analysis", "renal_labels.csv")


def choose_ratio_columns(df):
    provided_col = "kidney_mean_to_provided_reference_mean_ratio"
    provided_inner_col = "inner_mean_to_provided_reference_mean_ratio"

    ratio = df["kidney_mean_to_reference_mean_ratio"].copy()
    inner_ratio = df["inner_mean_to_reference_mean_ratio"].copy()

    if provided_col in df.columns:
        ratio = df[provided_col].fillna(ratio)
    if provided_inner_col in df.columns:
        inner_ratio = df[provided_inner_col].fillna(inner_ratio)

    return ratio, inner_ratio


def percentile_rank(series):
    return series.rank(pct=True, method="average")


def main():
    features_df = pd.read_csv(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)

    if "label_source" not in labels_df.columns:
        labels_df["label_source"] = ""
    if "label_name" not in labels_df.columns:
        labels_df["label_name"] = ""

    labels_df["label_name"] = labels_df["label_name"].fillna("").astype(str)
    labels_df["label_source"] = labels_df["label_source"].fillna("").astype(str)

    labels_df["label"] = pd.to_numeric(labels_df["label"], errors="coerce")

    merged = features_df.merge(labels_df, on=["split", "image_name"], how="left")

    kidney_ratio, inner_ratio = choose_ratio_columns(merged)

    score = (
        0.45 * percentile_rank(kidney_ratio.fillna(kidney_ratio.median()))
        + 0.25 * percentile_rank(inner_ratio.fillna(inner_ratio.median()))
        + 0.20 * percentile_rank(merged["kidney_mean"].fillna(merged["kidney_mean"].median()))
        + 0.10 * percentile_rank(merged["kidney_bright_ratio"].fillna(merged["kidney_bright_ratio"].median()))
    )

    merged["heuristic_score"] = score

    # Conservative pseudo-labeling:
    # top-scoring cases -> provisional diseased
    # bottom-scoring cases -> provisional healthy
    diseased_mask = merged["heuristic_score"] >= 0.92
    healthy_mask = merged["heuristic_score"] <= 0.15
    unlabeled_mask = merged["label"].isna()

    apply_diseased = unlabeled_mask & diseased_mask
    apply_healthy = unlabeled_mask & healthy_mask

    labels_df = labels_df.merge(
        merged[["split", "image_name", "heuristic_score"]],
        on=["split", "image_name"],
        how="left",
    )

    for idx, row in labels_df.iterrows():
        if not pd.isna(row["label"]):
            continue

        current = merged[
            (merged["split"] == row["split"]) &
            (merged["image_name"] == row["image_name"])
        ].iloc[0]

        if current["heuristic_score"] >= 0.92:
            labels_df.at[idx, "label"] = 1
            labels_df.at[idx, "label_name"] = "heuristic_diseased"
            labels_df.at[idx, "label_source"] = "heuristic"
        elif current["heuristic_score"] <= 0.15:
            labels_df.at[idx, "label"] = 0
            labels_df.at[idx, "label_name"] = "heuristic_healthy"
            labels_df.at[idx, "label_source"] = "heuristic"

    labels_df.to_csv(LABELS_PATH, index=False)

    labeled = labels_df.dropna(subset=["label"]).copy()
    counts = labeled.groupby(["split", "label"]).size()

    print("Heuristic suggestions applied.")
    print("Labeled counts by split/class:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
