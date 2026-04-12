import os
import cv2
import pandas as pd

from utils.renal_features import (
    extract_renal_features,
    load_binary_mask,
    load_grayscale,
)


DATASET_DIR = "dataset"
OUTPUT_DIR = os.path.join("results", "renal_feature_analysis")
MASK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "candidate_masks")
REFERENCE_MASK_DIR = "reference_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)


def overlay_mask(image, mask, color):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image_rgb.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image_rgb, 0.75, overlay, 0.25, 0)


def save_debug_panel(image, kidney_mask, masks, output_path):
    kidney_overlay = overlay_mask(image, kidney_mask, (0, 255, 0))
    inner_overlay = overlay_mask(image, masks["inner_mask"], (255, 0, 0))
    pyramid_overlay = overlay_mask(image, masks["pyramid_candidate_mask"], (0, 0, 255))
    reference_overlay = overlay_mask(image, masks["reference_mask"], (255, 255, 0))
    provided_reference_overlay = overlay_mask(image, masks["provided_reference_mask"], (255, 0, 255))

    top = cv2.hconcat([cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), kidney_overlay])
    bottom = cv2.hconcat([inner_overlay, pyramid_overlay])
    panel = cv2.vconcat([top, bottom])

    # Save a separate reference view because it is often useful for echogenicity comparison.
    ref_panel = cv2.hconcat([reference_overlay, provided_reference_overlay])
    final = cv2.vconcat([panel, ref_panel])

    cv2.imwrite(output_path, final)


def iter_dataset():
    for split in ["train", "val", "test"]:
        image_dir = os.path.join(DATASET_DIR, split, "image")
        mask_dir = os.path.join(DATASET_DIR, split, "mask")

        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            continue

        for file_name in sorted(os.listdir(image_dir)):
            yield split, os.path.join(image_dir, file_name), os.path.join(mask_dir, file_name), file_name


def load_optional_reference_mask(split, file_name):
    ref_path = os.path.join(REFERENCE_MASK_DIR, split, file_name)
    if not os.path.exists(ref_path):
        return None
    return load_binary_mask(ref_path)


def main():
    rows = []

    for split, image_path, mask_path, file_name in iter_dataset():
        image = load_grayscale(image_path)
        kidney_mask = load_binary_mask(mask_path)
        reference_mask = load_optional_reference_mask(split, file_name)

        features, masks = extract_renal_features(image, kidney_mask, reference_mask=reference_mask)
        features["split"] = split
        features["image_name"] = file_name
        rows.append(features)

        panel_path = os.path.join(
            MASK_OUTPUT_DIR,
            f"{split}_{os.path.splitext(file_name)[0]}_analysis.png"
        )
        save_debug_panel(image, kidney_mask, masks, panel_path)

    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "image_name"]).reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, "renal_features.csv")
    df.to_csv(csv_path, index=False)

    summary = df.groupby("split").agg(
        images=("image_name", "count"),
        kidney_mean_mean=("kidney_mean", "mean"),
        kidney_bright_ratio_mean=("kidney_bright_ratio", "mean"),
        pyramid_candidate_ratio_mean=("pyramid_candidate_ratio", "mean"),
        kidney_mean_to_reference_mean_ratio_mean=("kidney_mean_to_reference_mean_ratio", "mean"),
    )

    summary_path = os.path.join(OUTPUT_DIR, "renal_features_summary.csv")
    summary.to_csv(summary_path)

    print("Features saved:", csv_path)
    print("Summary saved:", summary_path)
    print("Candidate mask panels saved in:", MASK_OUTPUT_DIR)


if __name__ == "__main__":
    main()
