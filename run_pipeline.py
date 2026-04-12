import argparse
import os
import subprocess
import sys

import pandas as pd


FEATURES_PATH = os.path.join("results", "renal_feature_analysis", "renal_features.csv")
LABELS_PATH = os.path.join("results", "renal_feature_analysis", "renal_labels.csv")
REFERENCE_DIR = "reference_masks"


def run_python(script_name):
    command = [sys.executable, script_name]
    result = subprocess.run(command, check=False)
    return result.returncode


def count_reference_masks():
    total = 0
    by_split = {}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(REFERENCE_DIR, split)
        if not os.path.isdir(split_dir):
            by_split[split] = 0
            continue

        count = len(
            [
                name
                for name in os.listdir(split_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        by_split[split] = count
        total += count

    return total, by_split


def show_status():
    print("Pipeline status\n")

    if os.path.exists(FEATURES_PATH):
        features_df = pd.read_csv(FEATURES_PATH)
        print(f"Features file: OK ({len(features_df)} images)")
    else:
        print("Features file: missing")

    if os.path.exists(LABELS_PATH):
        labels_df = pd.read_csv(LABELS_PATH)
        labels_df["label"] = pd.to_numeric(labels_df["label"], errors="coerce")
        filled = labels_df.dropna(subset=["label"]).copy()

        print(f"Labels file: OK ({len(filled)} labeled images)")
        if not filled.empty:
            print("Labels by split and class:")
            counts = filled.groupby(["split", "label"]).size()
            for (split, label), count in counts.items():
                print(f"  {split} / class {int(label)}: {count}")
        else:
            print("  No labels filled yet.")
    else:
        print("Labels file: missing")

    total_refs, by_split = count_reference_masks()
    print(f"Reference masks: {total_refs}")
    for split, count in by_split.items():
        print(f"  {split}: {count}")

    print("\nRecommended next step:")
    if not os.path.exists(LABELS_PATH):
        print("  python run_pipeline.py init-labels")
    elif os.path.exists(LABELS_PATH):
        labels_df = pd.read_csv(LABELS_PATH)
        labels_df["label"] = pd.to_numeric(labels_df["label"], errors="coerce")
        filled = labels_df.dropna(subset=["label"])
        if filled.empty:
            print("  Fill results/renal_feature_analysis/renal_labels.csv")
        elif filled["label"].nunique() < 2:
            print("  Add at least one image with class 1 and one with class 0")
        elif not os.path.exists(FEATURES_PATH):
            print("  python run_pipeline.py extract")
        else:
            print("  python run_pipeline.py train")


def main():
    parser = argparse.ArgumentParser(description="Simple runner for the renal fibrosis pipeline.")
    parser.add_argument(
        "command",
        choices=["status", "init-labels", "extract", "train"],
        help="Pipeline command to execute",
    )

    args = parser.parse_args()

    if args.command == "status":
        show_status()
        return

    if args.command == "init-labels":
        sys.exit(run_python("prepare_renal_labels_template.py"))

    if args.command == "extract":
        sys.exit(run_python("extract_renal_features.py"))

    if args.command == "train":
        sys.exit(run_python("train_renal_classifier.py"))


if __name__ == "__main__":
    main()
