"""Avalia a cascata completa com a ROI da cápsula predita.

O teste compara duas condições para a estudante intrarrenal:

* ``oracle_roi``: usa a máscara renal fornecida pelo conjunto;
* ``predicted_roi``: usa a máscara produzida pela estudante da cápsula.

A segunda condição reproduz o fluxo operacional da Curadoria Web.
"""

from __future__ import annotations

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

from src.segmentation.build_dataset_geral import keep_largest_component, predict_probability, prepare_tensor
from src.segmentation.core.checkpoint_metadata import load_checkpoint_metadata
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.experiments.train_inner_deeplab import CLASS_NAMES, roi_bounds
from src.segmentation.experiments.train_unet import UNet


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1_grouped"
)
DEFAULT_CAPSULE = PROJECT_ROOT / "models" / "capsule_from_regions_grouped_student_unet_deeplab.pth"
DEFAULT_INNER = PROJECT_ROOT / "models" / "intrarenal_grouped_roi_student_unet_deeplab.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "intrarenal_distillation_grouped_roi" / "end_to_end"


def read_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_mask(mask, shape):
    if mask.shape[:2] == shape[:2]:
        return mask > 0
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def capsule_prediction(image, bundle, device, prepare_tensor_fn, predict_probability_fn):
    tensor = prepare_tensor_fn(image, 256, device, clahe=True)
    with torch.no_grad():
        probability = predict_probability_fn(bundle, tensor, 256)
    probability = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return keep_largest_component(probability >= float(bundle["threshold"])), probability


def inner_prediction(image, kidney, model, device):
    bounds = roi_bounds(kidney)
    if bounds is None:
        return np.zeros_like(image, dtype=np.uint8)
    x1, y1, x2, y2 = bounds
    roi_image = image[y1:y2, x1:x2]
    roi_kidney = kidney[y1:y2, x1:x2]
    resized = cv2.resize(roi_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    resized_kidney = cv2.resize(roi_kidney.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    normalized = clahe.astype(np.float32) / 255.0
    channels = np.stack(
        [normalized, normalized * resized_kidney.astype(np.float32), resized_kidney.astype(np.float32)],
        axis=0,
    )
    tensor = torch.tensor(channels, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        labels = torch.argmax(model(tensor), dim=1).cpu().numpy()[0].astype(np.uint8)
    labels[~resized_kidney] = 0
    labels = cv2.resize(labels, (roi_image.shape[1], roi_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    output = np.zeros_like(image, dtype=np.uint8)
    output[y1:y2, x1:x2] = labels
    return output


def metrics(prediction, reference):
    result = {}
    foreground_dice = []
    foreground_iou = []
    for class_id, name in enumerate(CLASS_NAMES):
        predicted = prediction == class_id
        truth = reference == class_id
        intersection = int(np.logical_and(predicted, truth).sum())
        predicted_sum = int(predicted.sum())
        truth_sum = int(truth.sum())
        union = int(np.logical_or(predicted, truth).sum())
        dice = 1.0 if predicted_sum + truth_sum == 0 else 2.0 * intersection / (predicted_sum + truth_sum)
        iou = 1.0 if union == 0 else intersection / union
        result[f"{name}_dice"] = float(dice)
        result[f"{name}_iou"] = float(iou)
        if class_id > 0:
            foreground_dice.append(dice)
            foreground_iou.append(iou)
    result["mean_foreground_dice"] = float(np.mean(foreground_dice))
    result["mean_foreground_iou"] = float(np.mean(foreground_iou))
    result["score"] = (result["mean_foreground_dice"] + result["mean_foreground_iou"]) / 2.0
    return result


def binary_dice(prediction, reference):
    prediction = prediction > 0
    reference = reference > 0
    denominator = int(prediction.sum() + reference.sum())
    return 1.0 if denominator == 0 else float(2.0 * np.logical_and(prediction, reference).sum() / denominator)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--capsule-checkpoint", type=Path, default=DEFAULT_CAPSULE)
    parser.add_argument("--inner-checkpoint", type=Path, default=DEFAULT_INNER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    capsule = load_model_bundle("unet", device=device, checkpoint_path=args.capsule_checkpoint.resolve())
    inner_meta = load_checkpoint_metadata(args.inner_checkpoint.resolve())
    model_kwargs = inner_meta.get("model_kwargs", {})
    inner = UNet(
        in_channels=int(model_kwargs.get("in_channels", 3)),
        out_channels=int(model_kwargs.get("out_channels", len(CLASS_NAMES))),
        base_channels=int(model_kwargs.get("base_channels", 32)),
    ).to(device)
    inner.load_state_dict(torch.load(args.inner_checkpoint.resolve(), map_location=device))
    inner.eval()

    from src.segmentation.build_dataset_geral import predict_probability as predict_probability_fn
    from src.segmentation.build_dataset_geral import prepare_tensor as prepare_tensor_fn

    rows = []
    oracle_metric_rows = []
    predicted_metric_rows = []
    for image_path in sorted((args.dataset_root / "test" / "image").glob("*.png")):
        image = read_gray(image_path)
        reference = cv2.imread(str(args.dataset_root / "test" / "mask" / image_path.name), cv2.IMREAD_GRAYSCALE)
        kidney_reference = read_gray(args.dataset_root / "test" / "kidney_mask" / image_path.name)
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        kidney_reference = resize_mask(kidney_reference, image.shape)
        kidney_prediction, _ = capsule_prediction(image, capsule, device, prepare_tensor_fn, predict_probability_fn)
        oracle_labels = inner_prediction(image, kidney_reference, inner, device)
        predicted_labels = inner_prediction(image, kidney_prediction, inner, device)
        oracle_metrics = metrics(oracle_labels, reference)
        predicted_metrics = metrics(predicted_labels, reference)
        oracle_metric_rows.append(oracle_metrics)
        predicted_metric_rows.append(predicted_metrics)
        rows.append(
            {
                "filename": image_path.name,
                "capsule_dice": binary_dice(kidney_prediction, kidney_reference),
                "oracle_mean_foreground_dice": oracle_metrics["mean_foreground_dice"],
                "predicted_roi_mean_foreground_dice": predicted_metrics["mean_foreground_dice"],
                "oracle_score": oracle_metrics["score"],
                "predicted_roi_score": predicted_metrics["score"],
            }
        )

    oracle_summary = {
        key: float(np.mean([row[key] for row in oracle_metric_rows]))
        for key in ("mean_foreground_dice", "mean_foreground_iou", "score")
    }
    predicted_summary = {
        key: float(np.mean([row[key] for row in predicted_metric_rows]))
        for key in ("mean_foreground_dice", "mean_foreground_iou", "score")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "end_to_end_predicted_capsule_roi",
        "device": device,
        "dataset_root": str(args.dataset_root.resolve()),
        "samples": len(rows),
        "capsule_checkpoint": str(args.capsule_checkpoint.resolve()),
        "inner_checkpoint": str(args.inner_checkpoint.resolve()),
        "capsule_dice_mean": float(np.mean([row["capsule_dice"] for row in rows])),
        "oracle_roi": oracle_summary,
        "predicted_roi": predicted_summary,
        "mean_predicted_roi_dice_from_per_image": float(np.mean([row["predicted_roi_mean_foreground_dice"] for row in rows])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
