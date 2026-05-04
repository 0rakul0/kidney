import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


TITLE_HEIGHT = 34
TILE_PADDING = 12
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def prepare_display_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Ultrasound frames use pure black around the acquisition cone. Converting
    # only those zero-value pixels preserves the anatomy while making the panel
    # print-friendly for the article.
    background = np.all(image_rgb == 0, axis=2)
    image_rgb[background] = 255

    return image_rgb


def overlay_mask(image, mask, color):
    image_rgb = prepare_display_image(image)
    overlay = image_rgb.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(image_rgb, 0.78, overlay, 0.22, 0)

    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        cv2.drawContours(blended, contours, -1, color, 2)

    return blended


def create_titled_tile(image, title):
    height, width = image.shape[:2]
    tile = np.full(
        (height + TITLE_HEIGHT + (2 * TILE_PADDING), width + (2 * TILE_PADDING), 3),
        255,
        dtype=np.uint8,
    )

    image_top = TILE_PADDING + TITLE_HEIGHT
    image_left = TILE_PADDING

    tile[image_top:image_top + height, image_left:image_left + width] = image

    cv2.putText(
        tile,
        title,
        (TILE_PADDING, TILE_PADDING + 22),
        LABEL_FONT,
        0.58,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        tile,
        (image_left - 1, image_top - 1),
        (image_left + width, image_top + height),
        (215, 215, 215),
        1,
    )

    return tile


def save_debug_panel(image, kidney_mask, masks, output_path):
    kidney_overlay = overlay_mask(image, kidney_mask, (0, 255, 0))
    inner_overlay = overlay_mask(image, masks["inner_mask"], (255, 0, 0))
    pyramid_overlay = overlay_mask(image, masks["pyramid_candidate_mask"], (0, 0, 255))
    reference_overlay = overlay_mask(image, masks["reference_mask"], (255, 255, 0))
    provided_reference_overlay = overlay_mask(image, masks["provided_reference_mask"], (255, 0, 255))

    tiles = [
        create_titled_tile(prepare_display_image(image), "Imagem original"),
        create_titled_tile(kidney_overlay, "Mascara renal"),
        create_titled_tile(inner_overlay, "Regiao interna"),
        create_titled_tile(pyramid_overlay, "Piramides candidatas"),
        create_titled_tile(reference_overlay, "Referencia usada"),
        create_titled_tile(provided_reference_overlay, "Referencia manual"),
    ]

    top_row = cv2.hconcat(tiles[:3])
    bottom_row = cv2.hconcat(tiles[3:])
    final = cv2.vconcat([top_row, bottom_row])

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
