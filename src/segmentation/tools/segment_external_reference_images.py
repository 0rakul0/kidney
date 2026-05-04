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

from src.segmentation.build_dataset_geral import analyze_mask, predict_probability, prepare_tensor
from src.segmentation.core.model_loader import load_model_bundle


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT_DIR = PROJECT_ROOT / "external_data" / "reference_ultrasound" / "images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "external_reference_segmentation"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "dataset_geral_deeplab_resnet50_best.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aplica o modelo 2, DeepLab campeao, em imagens externas de referencia "
            "e gera paineis com contorno amarelo produzido pela segmentacao."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", choices=["deeplab", "unet", "unetplusplus", "segformer"], default="deeplab")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None, help="Sobrescreve o limiar salvo no checkpoint.")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.80)
    parser.add_argument("--min-foreground-pixels", type=int, default=300)
    parser.add_argument("--max-components", type=int, default=6)
    return parser.parse_args()


def iter_images(directory):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def read_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Nao foi possivel ler imagem: {path}")
    return image


def classify_reference_label(path):
    text = " ".join(part.lower() for part in path.parts)
    if any(token in text for token in ["healthy", "normal", "saudavel", "bom"]):
        return "reference_healthy_or_normal"
    if any(token in text for token in ["ckd", "fibrose", "fibrosis", "suspicious", "alteracao", "ruim"]):
        return "reference_suspicious_or_altered"
    return "reference_unlabeled"


def prediction_for_image(bundle, image, args, device):
    tensor = prepare_tensor(image, args.img_size, device)
    with torch.no_grad():
        probability = predict_probability(bundle, tensor, args.img_size)
    threshold = float(args.threshold if args.threshold is not None else bundle["threshold"])
    return analyze_mask(
        probability,
        threshold,
        image.shape,
        {
            "confidence_threshold": args.confidence_threshold,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_foreground_pixels": args.min_foreground_pixels,
            "max_components": args.max_components,
        },
    )


def resize_to_height(image, height):
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_panel(image, mask, title):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = rgb.copy()
    overlay[mask > 0] = (0, 160, 255)
    overlay = cv2.addWeighted(rgb, 0.76, overlay, 0.24, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 190, 255), 2)

    mask_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tiles = [resize_to_height(tile, 260) for tile in [rgb, overlay, mask_rgb]]
    panel = cv2.hconcat(tiles)
    cv2.putText(panel, title[:95], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 255, 255), 2, cv2.LINE_AA)
    return panel


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Pasta de entrada nao encontrada: {args.input_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {args.checkpoint}")

    output_image_dir = args.output_root / "images"
    output_mask_dir = args.output_root / "masks_model2"
    output_panel_dir = args.output_root / "panels_model2"
    for directory in [output_image_dir, output_mask_dir, output_panel_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model_bundle(args.model, device=device, checkpoint_path=args.checkpoint)

    rows = []
    for index, image_path in enumerate(iter_images(args.input_dir), 1):
        image = read_grayscale(image_path)
        analysis = prediction_for_image(bundle, image, args, device)
        mask = analysis["mask"].astype(np.uint8)

        image_id = f"{image_path.stem}_{index:03d}"
        output_image_path = output_image_dir / f"{image_id}.png"
        output_mask_path = output_mask_dir / f"{image_id}.png"
        output_panel_path = output_panel_dir / f"{image_id}.png"

        cv2.imwrite(str(output_image_path), image)
        cv2.imwrite(str(output_mask_path), mask * 255)
        panel = draw_panel(
            image,
            mask,
            f"{classify_reference_label(image_path)} conf={analysis['confidence']:.3f} {image_path.name}",
        )
        cv2.imwrite(str(output_panel_path), panel)

        rows.append(
            {
                "image_id": image_id,
                "reference_label": classify_reference_label(image_path),
                "source_image_path": str(image_path),
                "output_image_path": str(output_image_path),
                "model2_mask_path": str(output_mask_path),
                "model2_panel_path": str(output_panel_path),
                "model2_mask_status": "accepted_by_thresholds" if analysis["accepted"] else "rejected_by_thresholds",
                "rejection_reason": analysis["rejection_reason"],
                "confidence": f"{analysis['confidence']:.6f}",
                "foreground_pixels": analysis["foreground_pixels"],
                "area_ratio": f"{analysis['area_ratio']:.6f}",
                "components": analysis["components"],
                "largest_component_pixels": analysis["largest_component_pixels"],
            }
        )

    write_csv(args.output_root / "external_reference_model2_manifest.csv", rows)
    summary = {
        "input_dir": str(args.input_dir),
        "output_root": str(args.output_root),
        "checkpoint": str(args.checkpoint),
        "device": device,
        "images_processed": len(rows),
        "accepted_by_thresholds": sum(1 for row in rows if row["model2_mask_status"] == "accepted_by_thresholds"),
        "rejected_by_thresholds": sum(1 for row in rows if row["model2_mask_status"] == "rejected_by_thresholds"),
        "note": "A linha amarela dos paineis e o contorno da mascara prevista pelo modelo 2.",
    }
    (args.output_root / "external_reference_model2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
