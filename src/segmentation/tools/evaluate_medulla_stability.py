import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.build_dataset_geral import predict_probability, prepare_tensor
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.tools.predict.medulla_roi import (
    load_medulla_bundle,
    predict_roi_probability,
    roi_bounds,
)


INTRARENAL_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal"
DEFAULT_REGIONS_ROOT = INTRARENAL_ROOT / "intermediario" / "kidneyus_regions"
DEFAULT_TEST_MANIFEST = INTRARENAL_ROOT / "supervisionado" / "medulla_annotator_1" / "test" / "manifest.csv"
DEFAULT_KIDNEY_CHECKPOINT = PROJECT_ROOT / "models" / "dataset_geral_deeplab_resnet50_best.pth"
DEFAULT_MEDULLA_DEEPLAB_CHECKPOINT = PROJECT_ROOT / "models" / "medulla_deeplab_resnet50_annotator1_baseline.pth"
DEFAULT_MEDULLA_ROI_UNET_CHECKPOINT = PROJECT_ROOT / "models" / "medulla_roi_unet_annotator1.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "stability_evaluation"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia estabilidade da segmentacao de Medulla em holdout: "
            "anotadores, ROI manual versus ROI prevista e medidas de opacidade."
        )
    )
    parser.add_argument("--regions-root", type=Path, default=DEFAULT_REGIONS_ROOT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--kidney-checkpoint", type=Path, default=DEFAULT_KIDNEY_CHECKPOINT)
    parser.add_argument("--medulla-checkpoint", type=Path, default=None)
    parser.add_argument("--architecture", choices=["deeplab", "roi_unet"], default="deeplab")
    parser.add_argument("--model", choices=["deeplab"], default="deeplab", help="Modelo usado quando architecture=deeplab.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--pad-ratio", type=float, default=0.12)
    parser.add_argument("--preview-count", type=int, default=12)
    return parser.parse_args()


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def load_mask(path, shape=None):
    mask = load_image(path)
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def crop(array, bounds):
    x1, y1, x2, y2 = bounds
    return array[y1:y2, x1:x2]


def safe_mean(image, mask):
    values = image[mask > 0].astype(np.float32) / 255.0
    return None if not values.size else float(values.mean())


def ratio(value, denominator):
    if value is None or denominator is None or denominator == 0:
        return None
    return float(value / denominator)


def mask_metrics(prediction, target):
    prediction = prediction > 0
    target = target > 0
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    pred_area = int(prediction.sum())
    target_area = int(target.sum())
    return {
        "intersection": intersection,
        "union": union,
        "pred_area": pred_area,
        "target_area": target_area,
        "dice": float(2 * intersection / (pred_area + target_area + 1e-8)),
        "iou": float(intersection / (union + 1e-8)),
    }


def predict_kidney_mask(bundle, image, args, device):
    tensor = prepare_tensor(image, args.img_size, device)
    with torch.no_grad():
        probability = predict_probability(bundle, tensor, args.img_size)
    probability = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (probability >= float(bundle["threshold"])).astype(np.uint8)


def predict_medulla_mask(bundle, image, kidney_mask, args, device):
    bounds = roi_bounds(kidney_mask, args.pad_ratio)
    output = np.zeros_like(kidney_mask)
    if bounds is None:
        return output
    roi_image = crop(image, bounds)
    roi_kidney = crop(kidney_mask, bounds)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_image)
    probability = predict_roi_probability(bundle, enhanced, roi_kidney, args, device)
    probability = cv2.resize(probability, (roi_image.shape[1], roi_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    predicted = ((probability >= float(bundle["threshold"])) & (roi_kidney > 0)).astype(np.uint8)
    x1, y1, x2, y2 = bounds
    output[y1:y2, x1:x2] = predicted
    return output


def aggregate_metrics(rows, prefix):
    intersection = sum(row[f"{prefix}_intersection"] for row in rows)
    union = sum(row[f"{prefix}_union"] for row in rows)
    pred_area = sum(row[f"{prefix}_pred_area"] for row in rows)
    target_area = sum(row[f"{prefix}_target_area"] for row in rows)
    dice_values = [row[f"{prefix}_dice"] for row in rows]
    iou_values = [row[f"{prefix}_iou"] for row in rows]
    return {
        "images": len(rows),
        "global_dice": round(float(2 * intersection / (pred_area + target_area + 1e-8)), 6),
        "global_iou": round(float(intersection / (union + 1e-8)), 6),
        "mean_per_image_dice": round(float(np.mean(dice_values)), 6),
        "mean_per_image_iou": round(float(np.mean(iou_values)), 6),
    }


def valid_pairs(rows, left, right):
    return [
        (row[left], row[right])
        for row in rows
        if row[left] is not None and row[right] is not None
    ]


def value_stability(rows, left, right):
    pairs = valid_pairs(rows, left, right)
    if not pairs:
        return {"images": 0, "mae": None, "mean_difference": None, "correlation": None}
    left_values = np.array([pair[0] for pair in pairs], dtype=np.float64)
    right_values = np.array([pair[1] for pair in pairs], dtype=np.float64)
    correlation = None
    if len(pairs) >= 2 and left_values.std() > 0 and right_values.std() > 0:
        correlation = round(float(np.corrcoef(left_values, right_values)[0, 1]), 6)
    return {
        "images": len(pairs),
        "mae": round(float(np.mean(np.abs(left_values - right_values))), 6),
        "mean_difference": round(float(np.mean(right_values - left_values)), 6),
        "correlation": correlation,
    }


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def save_panel(path, image, manual_medulla, manual_prediction, cascade_prediction, title):
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tiles = []
    for mask, color in (
        (manual_medulla, (255, 0, 255)),
        (manual_prediction, (0, 0, 255)),
        (cascade_prediction, (0, 255, 255)),
    ):
        overlay = base.copy()
        overlay[mask > 0] = color
        overlay = cv2.addWeighted(base, 0.72, overlay, 0.28, 0)
        tiles.append(cv2.resize(overlay, (280, 220), interpolation=cv2.INTER_AREA))
    panel = cv2.hconcat(tiles)
    cv2.putText(panel, title[:100], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.medulla_checkpoint is None:
        args.medulla_checkpoint = (
            DEFAULT_MEDULLA_ROI_UNET_CHECKPOINT
            if args.architecture == "roi_unet"
            else DEFAULT_MEDULLA_DEEPLAB_CHECKPOINT
        )
    args.checkpoint = args.medulla_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kidney_bundle = load_model_bundle("deeplab", device=device, checkpoint_path=args.kidney_checkpoint)
    medulla_bundle = load_medulla_bundle(args, device)
    test_rows = read_rows(args.test_manifest)
    results = []
    previews = 0

    for row in test_rows:
        file_name = row["filename"]
        image = load_image(args.regions_root / "images" / file_name)
        kidney_manual = load_mask(args.regions_root / "full_masks" / "annotator_1" / "capsule" / file_name, image.shape)
        medulla_a1 = load_mask(args.regions_root / "full_masks" / "annotator_1" / "medulla" / file_name, image.shape)
        medulla_a2 = load_mask(args.regions_root / "full_masks" / "annotator_2" / "medulla" / file_name, image.shape)
        kidney_predicted = predict_kidney_mask(kidney_bundle, image, args, device)
        medulla_manual_roi = predict_medulla_mask(medulla_bundle, image, kidney_manual, args, device)
        medulla_cascade = predict_medulla_mask(medulla_bundle, image, kidney_predicted, args, device)
        kidney_eval = mask_metrics(kidney_predicted, kidney_manual)
        manual_eval = mask_metrics(medulla_manual_roi, medulla_a1)
        cascade_eval = mask_metrics(medulla_cascade, medulla_a1)
        a2_available = bool(medulla_a2.any())
        manual_a2_eval = mask_metrics(medulla_manual_roi, medulla_a2) if a2_available else None
        cascade_a2_eval = mask_metrics(medulla_cascade, medulla_a2) if a2_available else None
        kidney_mean = safe_mean(image, kidney_manual)
        row_result = {
            "filename": file_name,
            **prefixed_metrics("kidney_model2_vs_manual", kidney_eval),
            **prefixed_metrics("manual_roi_vs_annotator1", manual_eval),
            **prefixed_metrics("cascade_roi_vs_annotator1", cascade_eval),
            "annotator2_has_medulla": a2_available,
            "manual_medulla_mean": safe_mean(image, medulla_a1),
            "manual_medulla_to_kidney_mean": ratio(safe_mean(image, medulla_a1), kidney_mean),
            "manual_roi_prediction_mean": safe_mean(image, medulla_manual_roi),
            "manual_roi_prediction_to_kidney_mean": ratio(safe_mean(image, medulla_manual_roi), kidney_mean),
            "cascade_prediction_mean": safe_mean(image, medulla_cascade),
            "cascade_prediction_to_kidney_mean": ratio(safe_mean(image, medulla_cascade), kidney_mean),
        }
        if manual_a2_eval is not None:
            row_result.update(prefixed_metrics("manual_roi_vs_annotator2", manual_a2_eval))
            row_result.update(prefixed_metrics("cascade_roi_vs_annotator2", cascade_a2_eval))
            row_result["annotator2_medulla_mean"] = safe_mean(image, medulla_a2)
            row_result["annotator2_medulla_to_kidney_mean"] = ratio(safe_mean(image, medulla_a2), kidney_mean)
        else:
            for prefix in ("manual_roi_vs_annotator2", "cascade_roi_vs_annotator2"):
                for key in ("intersection", "union", "pred_area", "target_area", "dice", "iou"):
                    row_result[f"{prefix}_{key}"] = None
            row_result["annotator2_medulla_mean"] = None
            row_result["annotator2_medulla_to_kidney_mean"] = None
        results.append(row_result)
        if previews < args.preview_count:
            previews += 1
            save_panel(
                args.output_root / args.architecture / "previews" / f"{previews:02d}_{file_name}",
                image,
                medulla_a1,
                medulla_manual_roi,
                medulla_cascade,
                f"{file_name} | manual={manual_eval['dice']:.3f} cascade={cascade_eval['dice']:.3f}",
            )

    annotator2_rows = [row for row in results if row["annotator2_has_medulla"]]
    summary = {
        "architecture": args.architecture,
        "medulla_checkpoint": str(args.medulla_checkpoint),
        "kidney_checkpoint": str(args.kidney_checkpoint),
        "holdout_images": len(results),
        "medulla_against_annotator1": {
            "manual_kidney_roi": aggregate_metrics(results, "manual_roi_vs_annotator1"),
            "model2_kidney_roi": aggregate_metrics(results, "cascade_roi_vs_annotator1"),
        },
        "model2_kidney_against_manual_capsule": aggregate_metrics(results, "kidney_model2_vs_manual"),
        "medulla_against_annotator2_on_holdout": {
            "eligible_images": len(annotator2_rows),
            "manual_kidney_roi": aggregate_metrics(annotator2_rows, "manual_roi_vs_annotator2") if annotator2_rows else None,
            "model2_kidney_roi": aggregate_metrics(annotator2_rows, "cascade_roi_vs_annotator2") if annotator2_rows else None,
        },
        "opacity_stability": {
            "manual_annotation_vs_prediction_manual_roi_mean": value_stability(
                results, "manual_medulla_mean", "manual_roi_prediction_mean"
            ),
            "manual_annotation_vs_prediction_cascade_mean": value_stability(
                results, "manual_medulla_mean", "cascade_prediction_mean"
            ),
            "manual_annotation_vs_prediction_manual_roi_ratio": value_stability(
                results, "manual_medulla_to_kidney_mean", "manual_roi_prediction_to_kidney_mean"
            ),
            "manual_annotation_vs_prediction_cascade_ratio": value_stability(
                results, "manual_medulla_to_kidney_mean", "cascade_prediction_to_kidney_mean"
            ),
            "annotator1_vs_annotator2_mean": value_stability(
                annotator2_rows, "manual_medulla_mean", "annotator2_medulla_mean"
            ),
        },
    }
    output_dir = args.output_root / args.architecture
    write_csv(output_dir / "per_image_metrics.csv", results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
