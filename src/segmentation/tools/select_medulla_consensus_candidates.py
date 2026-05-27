import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEEPLAB_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "medulla_predictions_dataset_geral"
DEFAULT_ROI_UNET_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "medulla_roi_unet_predictions_dataset_geral"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "medulla_consensus_review"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Seleciona pseudo-mascaras de Medulla para revisao usando consenso "
            "entre DeepLab e MedullaROIUNet."
        )
    )
    parser.add_argument("--deeplab-root", type=Path, default=DEFAULT_DEEPLAB_ROOT)
    parser.add_argument("--roi-unet-root", type=Path, default=DEFAULT_ROI_UNET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-dice", type=float, default=0.75)
    parser.add_argument(
        "--prediction-status",
        default="candidate_existing_kidney_mask",
        choices=[
            "candidate_existing_kidney_mask",
            "candidate_requires_kidney_roi_review",
        ],
        help="Grupo de predicoes a comparar entre os modelos.",
    )
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--borderline-preview-count", type=int, default=12)
    return parser.parse_args()


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_image(path, mode=cv2.IMREAD_GRAYSCALE):
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def load_mask(path, shape=None):
    mask = load_image(path)
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def metrics(left, right):
    left = left > 0
    right = right > 0
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    left_area = int(left.sum())
    right_area = int(right.sum())
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "deeplab_pixels": left_area,
        "roi_unet_pixels": right_area,
        "model_dice": float(2 * intersection / (left_area + right_area + 1e-8)),
        "model_iou": float(intersection / (union + 1e-8)),
    }


def roi_bounds(mask, pad=10):
    ys, xs = np.where(mask > 0)
    if not xs.size:
        return (0, 0, mask.shape[1], mask.shape[0])
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(mask.shape[1], int(xs.max()) + pad + 1),
        min(mask.shape[0], int(ys.max()) + pad + 1),
    )


def crop(array, bounds):
    x1, y1, x2, y2 = bounds
    return array[y1:y2, x1:x2]


def overlay(base, mask, color):
    colored = base.copy()
    colored[mask > 0] = color
    return cv2.addWeighted(base, 0.72, colored, 0.28, 0)


def save_panel(path, row, label):
    image = load_image(Path(row["dataset_image_path"]))
    kidney = load_mask(Path(row["dataset_kidney_mask_path"]), image.shape)
    deeplab = load_mask(Path(row["deeplab_mask_path"]), image.shape)
    roi_unet = load_mask(Path(row["roi_unet_mask_path"]), image.shape)
    agreement = np.logical_and(deeplab > 0, roi_unet > 0).astype(np.uint8)
    bounds = roi_bounds(kidney)
    base = cv2.cvtColor(crop(image, bounds), cv2.COLOR_GRAY2BGR)
    tiles = [
        overlay(base, crop(kidney, bounds), (0, 180, 255)),
        overlay(base, crop(deeplab, bounds), (0, 0, 255)),
        overlay(base, crop(roi_unet, bounds), (255, 0, 255)),
        overlay(base, crop(agreement, bounds), (0, 255, 0)),
    ]
    resized = [cv2.resize(tile, (250, 205), interpolation=cv2.INTER_AREA) for tile in tiles]
    panel = cv2.hconcat(resized)
    cv2.putText(panel, label[:130], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), panel):
        raise RuntimeError(f"Nao foi possivel salvar imagem: {path}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows(root, prediction_status):
    return {
        row["image_id"]: row
        for row in read_rows(root / "manifest.csv")
        if row["prediction_status"] == prediction_status
    }


def main():
    args = parse_args()
    deeplab_rows = candidate_rows(args.deeplab_root, args.prediction_status)
    roi_unet_rows = candidate_rows(args.roi_unet_root, args.prediction_status)
    common_ids = sorted(set(deeplab_rows).intersection(roi_unet_rows))
    compared = []
    for image_id in common_ids:
        deeplab = deeplab_rows[image_id]
        roi_unet = roi_unet_rows[image_id]
        left_mask = load_mask(Path(deeplab["predicted_medulla_mask_path"]))
        right_mask = load_mask(Path(roi_unet["predicted_medulla_mask_path"]), left_mask.shape)
        agreement = metrics(left_mask, right_mask)
        compared.append(
            {
                "image_id": image_id,
                "source_name": deeplab["source_name"],
                "dataset_image_path": deeplab["dataset_image_path"],
                "dataset_kidney_mask_path": deeplab["dataset_kidney_mask_path"],
                "deeplab_mask_path": deeplab["predicted_medulla_mask_path"],
                "roi_unet_mask_path": roi_unet["predicted_medulla_mask_path"],
                **agreement,
                "deeplab_medulla_to_kidney_ratio": deeplab["medulla_to_kidney_ratio"],
                "roi_unet_medulla_to_kidney_ratio": roi_unet["medulla_to_kidney_ratio"],
                "ratio_absolute_difference": abs(
                    float(deeplab["medulla_to_kidney_ratio"])
                    - float(roi_unet["medulla_to_kidney_ratio"])
                ),
                "review_status": (
                    "selected_for_review"
                    if agreement["model_dice"] >= args.min_dice
                    else "below_consensus_threshold"
                ),
            }
        )
    compared.sort(key=lambda row: row["model_dice"], reverse=True)
    selected = [row for row in compared if row["review_status"] == "selected_for_review"]
    rejected = [row for row in compared if row["review_status"] != "selected_for_review"]
    write_csv(args.output_root / "consensus_candidates.csv", compared)
    write_csv(args.output_root / "selected_for_review.csv", selected)

    for index, row in enumerate(selected[: args.preview_count], 1):
        save_panel(
            args.output_root / "previews" / "selected" / f"{index:03d}_{row['image_id']}.png",
            row,
            f"{row['image_id']} | selected | Dice={row['model_dice']:.3f}",
        )
    borderline = sorted(rejected, key=lambda row: row["model_dice"], reverse=True)
    for index, row in enumerate(borderline[: args.borderline_preview_count], 1):
        save_panel(
            args.output_root / "previews" / "borderline" / f"{index:03d}_{row['image_id']}.png",
            row,
            f"{row['image_id']} | below threshold | Dice={row['model_dice']:.3f}",
        )

    dice_values = np.array([row["model_dice"] for row in compared], dtype=np.float32)
    threshold_counts = {
        f"at_least_{threshold:.2f}": sum(row["model_dice"] >= threshold for row in compared)
        for threshold in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
    }
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "deeplab_root": str(args.deeplab_root),
        "roi_unet_root": str(args.roi_unet_root),
        "selection_rule": {
            "prediction_status": args.prediction_status,
            "kidney_mask_status": (
                "existing"
                if args.prediction_status == "candidate_existing_kidney_mask"
                else "generated_requires_review"
            ),
            "both_models_candidate": True,
            "minimum_model_to_model_dice": args.min_dice,
        },
        "common_candidate_images": len(compared),
        "selected_for_review": len(selected),
        "model_to_model_dice": {
            "mean": round(float(dice_values.mean()), 6) if dice_values.size else None,
            "median": round(float(np.median(dice_values)), 6) if dice_values.size else None,
        },
        "threshold_counts": threshold_counts,
        "output_root": str(args.output_root),
        "note": (
            "A concordancia entre modelos serve para priorizar revisao visual; "
            "nao transforma as pseudo-mascaras em rotulos manuais."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
