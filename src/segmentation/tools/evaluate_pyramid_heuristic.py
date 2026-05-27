import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.renal_features import heuristic_pyramid_mask, normalize_image


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal" / "kidneyus_regions"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "heuristic_medulla_baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia a mascara heuristica de piramides contra anotacoes kidneyUS de Medulla."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--annotator",
        choices=["annotator_1", "annotator_2", "all"],
        default="all",
    )
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Nao foi possivel ler mascara: {path}")
    return (mask > 0).astype(np.uint8)


def metrics(prediction, target):
    prediction = prediction > 0
    target = target > 0
    intersection = int(np.logical_and(prediction, target).sum())
    prediction_area = int(prediction.sum())
    target_area = int(target.sum())
    union = int(np.logical_or(prediction, target).sum())
    return {
        "dice": float(2 * intersection / (prediction_area + target_area + 1e-8)),
        "iou": float(intersection / (union + 1e-8)),
        "precision": float(intersection / (prediction_area + 1e-8)),
        "recall": float(intersection / (target_area + 1e-8)),
        "predicted_pixels": prediction_area,
        "target_pixels": target_area,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_row(dataset_root, row):
    annotator = row["annotator"]
    file_name = row["filename"]
    image = cv2.imread(str(dataset_root / "images" / file_name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {file_name}")
    capsule = load_mask(dataset_root / "full_masks" / annotator / "capsule" / file_name)
    medulla = load_mask(dataset_root / "full_masks" / annotator / "medulla" / file_name)
    prediction = heuristic_pyramid_mask(normalize_image(image), capsule)
    values = metrics(prediction, medulla)
    return {
        "filename": file_name,
        "annotator": annotator,
        **{key: f"{value:.6f}" if isinstance(value, float) else value for key, value in values.items()},
    }


def summarize(rows):
    summary = {"target": "Medulla", "baseline": "heuristic_pyramid_mask", "by_annotator": {}}
    for annotator in sorted({row["annotator"] for row in rows}):
        subset = [row for row in rows if row["annotator"] == annotator]
        summary["by_annotator"][annotator] = {
            "images": len(subset),
            "mean_dice": round(float(np.mean([float(row["dice"]) for row in subset])), 6),
            "mean_iou": round(float(np.mean([float(row["iou"]) for row in subset])), 6),
            "mean_precision": round(float(np.mean([float(row["precision"]) for row in subset])), 6),
            "mean_recall": round(float(np.mean([float(row["recall"]) for row in subset])), 6),
        }
    return summary


def main():
    args = parse_args()
    manifest = read_manifest(args.dataset_root / "manifest.csv")
    eligible = [
        row
        for row in manifest
        if row["eligible_medulla_training"] == "true"
        and (args.annotator == "all" or row["annotator"] == args.annotator)
    ]
    if not eligible:
        raise ValueError("Nenhuma imagem elegivel de medula foi encontrada.")

    rows = [evaluate_row(args.dataset_root, row) for row in eligible]
    summary = summarize(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "per_image_metrics.csv", rows)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
