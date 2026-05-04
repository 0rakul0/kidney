import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.build_dataset_geral import (
    analyze_mask,
    predict_probability,
    prepare_tensor,
    summarize,
    write_csv,
)
from src.segmentation.core.model_loader import load_model_bundle


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_geral"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "dataset_geral_deeplab_resnet50_best.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera mascaras faltantes do dataset_geral com o modelo campeao, "
            "sem sobrescrever mascaras existentes."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model", choices=["unet", "unetplusplus", "deeplab", "segformer"], default="deeplab")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--min-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
    parser.add_argument("--min-foreground-pixels", type=int, default=800)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Limita a quantidade de imagens processadas.")
    parser.add_argument("--dry-run", action="store_true", help="Calcula a inferencia sem salvar mascaras ou manifestos.")
    parser.add_argument(
        "--refresh-summary-only",
        action="store_true",
        help="Atualiza apenas summary.json e relatorios derivados do manifesto existente.",
    )
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def mask_is_missing(row):
    if row.get("has_mask", "").lower() != "true":
        return True
    mask_path = row.get("dataset_mask_path", "")
    return not mask_path or not Path(mask_path).exists()


def read_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def generate_one(bundle, row, args, device):
    image_path = Path(row["dataset_image_path"])
    image = read_grayscale(image_path)
    tensor = prepare_tensor(image, args.img_size, device)
    threshold = float(bundle["threshold"])

    with torch.no_grad():
        probability = predict_probability(bundle, tensor, args.img_size)

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


def update_summary(dataset_root, rows, args, device, processed, accepted, rejected):
    summary = summarize(rows)
    previous_summary_path = dataset_root / "summary.json"
    previous_summary = {}
    if previous_summary_path.exists():
        with previous_summary_path.open("r", encoding="utf-8") as file:
            previous_summary = json.load(file)

    summary.update(
        {
            "output_root": str(dataset_root),
            "image_dir": str(dataset_root / "imagens"),
            "mask_dir": str(dataset_root / "mascaras"),
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "device": device,
            "confidence_threshold": args.confidence_threshold,
            "duplicates_skipped": previous_summary.get("duplicates_skipped", 0),
            "last_missing_mask_generation": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "script": str(Path(__file__).resolve()),
                "checkpoint": str(args.checkpoint),
                "model": args.model,
                "processed": processed,
                "accepted": accepted,
                "rejected": rejected,
                "threshold_from_checkpoint": None,
            },
        }
    )
    return summary


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    manifest_path = dataset_root / "manifest.csv"
    mask_dir = dataset_root / "mascaras"
    report_dir = dataset_root / "relatorios"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto nao encontrado: {manifest_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {args.checkpoint}")

    rows = read_manifest(manifest_path)
    targets = [row for row in rows if mask_is_missing(row)]
    if args.limit is not None:
        targets = targets[: args.limit]

    mask_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.refresh_summary_only:
        summary = update_summary(dataset_root, rows, args, device, 0, 0, 0)
        previous_generation = {}
        previous_summary_path = dataset_root / "summary.json"
        if previous_summary_path.exists():
            with previous_summary_path.open("r", encoding="utf-8") as file:
                previous_generation = json.load(file).get("last_missing_mask_generation", {})
        if previous_generation:
            summary["last_missing_mask_generation"] = previous_generation
        write_csv(report_dir / "faltando_mascara.csv", [row for row in rows if row.get("has_mask") == "false"])
        generated_rows = [row for row in rows if row.get("mask_status", "").startswith("generated")]
        write_csv(report_dir / "mascaras_geradas.csv", generated_rows)
        with (dataset_root / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    bundle = load_model_bundle(args.model, device=device, checkpoint_path=args.checkpoint)

    accepted_rows = []
    rejected_rows = []
    accepted = 0
    rejected = 0

    for index, row in enumerate(targets, 1):
        analysis = generate_one(bundle, row, args, device)
        quality = {
            "confidence": f"{analysis['confidence']:.6f}",
            "foreground_pixels": analysis["foreground_pixels"],
            "area_ratio": f"{analysis['area_ratio']:.6f}",
            "components": analysis["components"],
            "largest_component_pixels": analysis["largest_component_pixels"],
            "rejection_reason": analysis["rejection_reason"],
        }
        row.update(quality)
        row["mask_confidence_threshold"] = args.confidence_threshold

        if analysis["accepted"]:
            output_path = mask_dir / f"{row['image_id']}.png"
            if not args.dry_run:
                cv2.imwrite(str(output_path), analysis["mask"].astype(np.uint8) * 255)
            row["dataset_mask_path"] = str(output_path)
            row["has_mask"] = "true"
            row["mask_status"] = "generated_accepted_champion_deeplab"
            accepted += 1
            accepted_rows.append(dict(row))
        else:
            row["dataset_mask_path"] = ""
            row["has_mask"] = "false"
            row["mask_status"] = "generated_rejected_champion_deeplab"
            rejected += 1
            rejected_rows.append(dict(row))

        if index % 100 == 0:
            print(f"Processadas {index}/{len(targets)} imagens sem mascara")

    if not args.dry_run:
        write_csv(manifest_path, rows)
        write_csv(report_dir / "faltando_mascara.csv", [row for row in rows if row.get("has_mask") == "false"])
        write_csv(report_dir / "mascaras_geradas_modelo_campeao.csv", accepted_rows + rejected_rows)
        generated_rows = [row for row in rows if row.get("mask_status", "").startswith("generated")]
        write_csv(report_dir / "mascaras_geradas.csv", generated_rows)

        summary = update_summary(dataset_root, rows, args, device, len(targets), accepted, rejected)
        summary["last_missing_mask_generation"]["threshold_from_checkpoint"] = float(bundle["threshold"])
        with (dataset_root / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
    else:
        summary = {
            "dry_run": True,
            "targets": len(targets),
            "accepted": accepted,
            "rejected": rejected,
            "checkpoint": str(args.checkpoint),
            "threshold_from_checkpoint": float(bundle["threshold"]),
            "device": device,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
