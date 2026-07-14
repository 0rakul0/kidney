import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.build_dataset_geral import (
    binary_dice,
    keep_largest_component,
    predict_probability,
    prepare_tensor,
)
from src.segmentation.core.model_loader import load_model_bundle


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1_deduplicated"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_unet.pth"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "segmentation_experiments"
    / "kidneyus_capsule_deduplicated_benchmark"
    / "pseudomask_review_calibration.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibra indicadores para ordenar a revisao de pseudomascaras. "
            "Os limiares nao validam automaticamente uma mascara."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model", choices=["unet", "unetplusplus", "deeplab", "segformer"], default="unet")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--reference-percentile", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model_bundle(
        args.model,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    threshold = float(bundle["threshold"])
    image_dir = args.dataset_root / args.split / "image"
    mask_dir = args.dataset_root / args.split / "mask"
    rows = []

    for image_path in sorted(image_dir.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(mask_dir / image_path.name), cv2.IMREAD_GRAYSCALE) > 0
        original_tensor = prepare_tensor(image, args.img_size, device, clahe=True)
        flipped_tensor = prepare_tensor(cv2.flip(image, 1), args.img_size, device, clahe=True)

        with torch.no_grad():
            original_probability = predict_probability(
                bundle,
                original_tensor,
                args.img_size,
            )
            flipped_probability = np.fliplr(
                predict_probability(bundle, flipped_tensor, args.img_size)
            )

        tta_consistency = binary_dice(
            original_probability >= threshold,
            flipped_probability >= threshold,
        )
        probability = (original_probability + flipped_probability) / 2.0
        probability = cv2.resize(
            probability,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        prediction = keep_largest_component(probability >= threshold) > 0
        confidence = float(probability[prediction].mean()) if prediction.any() else 0.0
        dice = binary_dice(prediction, reference)
        rows.append(
            {
                "filename": image_path.name,
                "manual_dice": dice,
                "confidence": confidence,
                "tta_consistency_dice": tta_consistency,
                "area_ratio": float(prediction.sum() / max(prediction.size, 1)),
            }
        )

    confidence_values = np.asarray([row["confidence"] for row in rows], dtype=float)
    consistency_values = np.asarray(
        [row["tta_consistency_dice"] for row in rows],
        dtype=float,
    )
    dice_values = np.asarray([row["manual_dice"] for row in rows], dtype=float)
    percentile = float(args.reference_percentile)
    summary = {
        "purpose": (
            "reference distribution for review prioritization only; "
            "it does not validate external pseudomasks"
        ),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "samples": len(rows),
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "model_threshold": threshold,
        "clahe": True,
        "tta_horizontal_flip": True,
        "reference_percentile": percentile,
        "review_reference": {
            "confidence_p05": float(np.percentile(confidence_values, percentile)),
            "tta_consistency_dice_p05": float(
                np.percentile(consistency_values, percentile)
            ),
        },
        "manual_validation": {
            "mean_dice": float(dice_values.mean()),
            "median_dice": float(np.median(dice_values)),
            "minimum_dice": float(dice_values.min()),
            "confidence_dice_correlation": float(
                np.corrcoef(confidence_values, dice_values)[0, 1]
            ),
            "tta_dice_correlation": float(
                np.corrcoef(consistency_values, dice_values)[0, 1]
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
