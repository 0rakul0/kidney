import os
import pandas as pd


FEATURES_PATH = os.path.join("results", "renal_feature_analysis", "renal_features.csv")
LABELS_PATH = os.path.join("results", "renal_feature_analysis", "renal_labels.csv")


def main():
    df = pd.read_csv(FEATURES_PATH)

    template = df[["split", "image_name"]].copy()
    template["label"] = ""
    template["label_name"] = ""

    template.to_csv(LABELS_PATH, index=False)
    print("Label template saved:", LABELS_PATH)
    print("Use label=0 for healthy and label=1 for diseased.")


if __name__ == "__main__":
    main()
