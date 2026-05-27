import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.model_loader import load_model_bundle


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATASET_INICIAL_ROOT = PROJECT_ROOT / "dataset_inicial"
DATASET_AUMENTADO_ROOT = PROJECT_ROOT / "dataset_aumentado"
FONTES_ROOT = DATASET_AUMENTADO_ROOT / "fontes"
DEFAULT_OUTPUT_ROOT = DATASET_AUMENTADO_ROOT / "dataset_geral"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "augmented_deeplab_resnet50_baseline.pth"


@dataclass
class ImageCandidate:
    source_name: str
    image_path: Path
    mask_path: Path | None
    label_source: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Monta dataset_geral com todas as imagens disponiveis, copia "
            "mascaras existentes e gera pseudo-mascaras faltantes com controle "
            "de qualidade."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", choices=["unet", "unetplusplus", "deeplab", "segformer"], default="deeplab")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--min-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
    parser.add_argument("--min-foreground-pixels", type=int, default=800)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Cria dataset_geral apenas com mascaras existentes; nao gera faltantes.",
    )
    parser.add_argument(
        "--include-monai",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inclui PNGs curados de dataset_aumentado/fontes/external_data/processed/*/images, incluindo MONAI e outros datasets externos.",
    )
    return parser.parse_args()


def iter_images(directory):
    directory = Path(directory)
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def collect_split_dataset(root, source_name, label_source):
    root = Path(root)
    candidates = []
    for split in ("train", "val", "test"):
        image_dir = root / split / "image"
        mask_dir = root / split / "mask"
        for image_path in iter_images(image_dir):
            mask_path = mask_dir / image_path.name
            candidates.append(
                ImageCandidate(
                    source_name=f"{source_name}_{split}",
                    image_path=image_path,
                    mask_path=mask_path if mask_path.exists() else None,
                    label_source=label_source,
                )
            )
    return candidates


def collect_flat_pair(image_dir, mask_dir, source_name, label_source):
    candidates = []
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    for image_path in iter_images(image_dir):
        mask_path = mask_dir / image_path.name
        candidates.append(
            ImageCandidate(
                source_name=source_name,
                image_path=image_path,
                mask_path=mask_path if mask_path.exists() else None,
                label_source=label_source,
            )
        )
    return candidates


def collect_all_candidates(include_monai):
    candidates = []
    candidates.extend(collect_split_dataset(DATASET_INICIAL_ROOT, "dataset", "manual_or_primary"))
    candidates.extend(
        collect_split_dataset(
            DATASET_AUMENTADO_ROOT / "expansao_pseudorrotulada",
            "dataset_augmented",
            "mixed_existing_or_pseudo",
        )
    )
    candidates.extend(
        collect_flat_pair(
            FONTES_ROOT / "identificada" / "image",
            FONTES_ROOT / "identificada" / "mask",
            "identificada",
            "legacy_identificada",
        )
    )
    candidates.extend(
        collect_flat_pair(
            DATASET_AUMENTADO_ROOT / "pseudo_labels" / "accepted" / "image",
            DATASET_AUMENTADO_ROOT / "pseudo_labels" / "accepted" / "mask",
            "pseudo_labels_accepted",
            "pseudo_existing",
        )
    )
    for image_path in iter_images(FONTES_ROOT / "dataset_loader"):
        candidates.append(ImageCandidate("dataset_loader", image_path, None, "unlabeled"))
    for image_path in iter_images(FONTES_ROOT / "kidneyUS_images_25_june_2025"):
        candidates.append(ImageCandidate("kidneyus_external_png", image_path, None, "unlabeled"))
    if include_monai:
        processed_root = FONTES_ROOT / "external_data" / "processed"
        for image_dir in sorted(processed_root.glob("*/images")):
            source_name = image_dir.parent.name
            for image_path in iter_images(image_dir):
                candidates.append(ImageCandidate(source_name, image_path, None, "unlabeled"))
    return candidates


def file_sha1(path):
    sha = hashlib.sha1()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def safe_name(value):
    keep = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def build_image_id(candidate, digest, used_ids):
    base = safe_name(candidate.image_path.stem)
    source = safe_name(candidate.source_name)
    image_id = f"{source}__{base}"
    if image_id in used_ids:
        image_id = f"{image_id}__{digest[:10]}"
    used_ids.add(image_id)
    return image_id


def read_grayscale_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def copy_image(candidate, image_id, output_image_dir):
    image = read_grayscale_image(candidate.image_path)
    output_path = output_image_dir / f"{image_id}.png"
    cv2.imwrite(str(output_path), image)
    return output_path, image.shape


def copy_existing_mask(mask_path, image_shape, image_id, output_mask_dir):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, "existing_mask_unreadable"
    if mask.shape[:2] != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 0).astype(np.uint8) * 255)
    output_path = output_mask_dir / f"{image_id}.png"
    cv2.imwrite(str(output_path), mask)
    return output_path, "existing_mask_copied"


def prepare_tensor(image, img_size, device):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    stacked = np.stack([normalized, normalized, normalized], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)


def predict_probability(bundle, tensor, img_size):
    model = bundle["model"]
    display_name = bundle["display_name"]
    if display_name == "SegFormer":
        logits = model(pixel_values=tensor).logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    elif display_name == "DeepLab":
        logits = model(tensor)["out"]
    else:
        logits = model(tensor)
    return torch.sigmoid(logits).squeeze().detach().cpu().numpy()


def analyze_mask(probability, threshold, original_shape, config):
    probability = cv2.resize(
        probability,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = (probability >= threshold).astype(np.uint8)
    foreground_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    area_ratio = float(foreground_pixels / max(total_pixels, 1))

    if foreground_pixels == 0:
        confidence = 0.0
    else:
        confidence = float(probability[mask > 0].mean())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area > 0:
            components.append(area)

    accepted = True
    rejection_reasons = []
    if confidence < config["confidence_threshold"]:
        accepted = False
        rejection_reasons.append("low_confidence")
    if foreground_pixels < config["min_foreground_pixels"]:
        accepted = False
        rejection_reasons.append("too_few_foreground_pixels")
    if area_ratio < config["min_area_ratio"]:
        accepted = False
        rejection_reasons.append("area_ratio_too_low")
    if area_ratio > config["max_area_ratio"]:
        accepted = False
        rejection_reasons.append("area_ratio_too_high")
    if len(components) > config["max_components"]:
        accepted = False
        rejection_reasons.append("too_many_components")

    return {
        "accepted": accepted,
        "rejection_reason": ";".join(rejection_reasons),
        "mask": mask,
        "confidence": confidence,
        "foreground_pixels": foreground_pixels,
        "area_ratio": area_ratio,
        "components": len(components),
        "largest_component_pixels": max(components) if components else 0,
    }


def generate_mask(bundle, image_path, image_shape, image_id, output_mask_dir, args, device):
    image = read_grayscale_image(image_path)
    tensor = prepare_tensor(image, args.img_size, device)
    threshold = float(bundle["threshold"])
    with torch.no_grad():
        probability = predict_probability(bundle, tensor, args.img_size)

    analysis = analyze_mask(
        probability,
        threshold,
        image_shape,
        {
            "confidence_threshold": args.confidence_threshold,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_foreground_pixels": args.min_foreground_pixels,
            "max_components": args.max_components,
        },
    )

    if analysis["accepted"]:
        output_path = output_mask_dir / f"{image_id}.png"
        cv2.imwrite(str(output_path), analysis["mask"].astype(np.uint8) * 255)
        analysis["mask_path"] = output_path
    else:
        analysis["mask_path"] = ""
    return analysis


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {
        "total_unique_images": len(rows),
        "with_mask": sum(1 for row in rows if row["has_mask"] == "true"),
        "without_mask": sum(1 for row in rows if row["has_mask"] == "false"),
        "existing_masks": sum(1 for row in rows if row["mask_status"] == "existing"),
        "generated_masks_accepted": sum(1 for row in rows if row["mask_status"].startswith("generated_accepted")),
        "generated_masks_rejected": sum(1 for row in rows if row["mask_status"].startswith("generated_rejected")),
        "missing_not_generated": sum(1 for row in rows if row["mask_status"] == "missing_not_generated"),
        "by_source": {},
    }
    for row in rows:
        source = row["source_name"]
        summary["by_source"].setdefault(source, {"images": 0, "with_mask": 0})
        summary["by_source"][source]["images"] += 1
        if row["has_mask"] == "true":
            summary["by_source"][source]["with_mask"] += 1
    return summary


def main():
    args = parse_args()
    if args.clear_output and args.output_root.exists():
        resolved = args.output_root.resolve()
        if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
            raise RuntimeError(f"Saida insegura para remover: {resolved}")
        shutil.rmtree(resolved)

    image_dir = args.output_root / "imagens"
    mask_dir = args.output_root / "mascaras"
    report_dir = args.output_root / "relatorios"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_all_candidates(include_monai=args.include_monai)
    seen_hashes = {}
    used_ids = set()
    rows = []
    duplicate_rows = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = None
    if not args.inventory_only:
        bundle = load_model_bundle(
            args.model,
            device=device,
            checkpoint_path=args.checkpoint,
        )

    for index, candidate in enumerate(candidates, 1):
        digest = file_sha1(candidate.image_path)
        if digest in seen_hashes:
            duplicate_rows.append(
                {
                    "duplicate_image_path": str(candidate.image_path),
                    "kept_image_id": seen_hashes[digest],
                    "source_name": candidate.source_name,
                    "sha1": digest,
                }
            )
            continue

        image_id = build_image_id(candidate, digest, used_ids)
        seen_hashes[digest] = image_id
        output_image_path, image_shape = copy_image(candidate, image_id, image_dir)

        mask_status = "missing_not_generated"
        output_mask_path = ""
        mask_quality = {
            "confidence": "",
            "foreground_pixels": "",
            "area_ratio": "",
            "components": "",
            "largest_component_pixels": "",
            "rejection_reason": "",
        }

        if candidate.mask_path is not None:
            copied_mask_path, copy_reason = copy_existing_mask(
                candidate.mask_path,
                image_shape,
                image_id,
                mask_dir,
            )
            if copied_mask_path:
                output_mask_path = str(copied_mask_path)
                mask_status = "existing"
            else:
                mask_status = "existing_rejected"
                mask_quality["rejection_reason"] = copy_reason
        elif not args.inventory_only and bundle is not None:
            analysis = generate_mask(
                bundle,
                output_image_path,
                image_shape,
                image_id,
                mask_dir,
                args,
                device,
            )
            mask_quality = {
                "confidence": f"{analysis['confidence']:.6f}",
                "foreground_pixels": analysis["foreground_pixels"],
                "area_ratio": f"{analysis['area_ratio']:.6f}",
                "components": analysis["components"],
                "largest_component_pixels": analysis["largest_component_pixels"],
                "rejection_reason": analysis["rejection_reason"],
            }
            if analysis["accepted"]:
                output_mask_path = str(analysis["mask_path"])
                mask_status = "generated_accepted"
            else:
                mask_status = "generated_rejected"

        has_mask = output_mask_path != ""
        rows.append(
            {
                "image_id": image_id,
                "source_name": candidate.source_name,
                "label_source": candidate.label_source,
                "sha1": digest,
                "original_image_path": str(candidate.image_path),
                "original_mask_path": str(candidate.mask_path or ""),
                "dataset_image_path": str(output_image_path),
                "dataset_mask_path": output_mask_path,
                "has_mask": str(has_mask).lower(),
                "mask_status": mask_status,
                "mask_confidence_threshold": args.confidence_threshold,
                **mask_quality,
            }
        )

        if index % 100 == 0:
            print(f"Processadas {index}/{len(candidates)} imagens candidatas")

    write_csv(args.output_root / "manifest.csv", rows)
    write_csv(report_dir / "duplicadas_por_hash.csv", duplicate_rows)
    write_csv(report_dir / "faltando_mascara.csv", [row for row in rows if row["has_mask"] == "false"])
    write_csv(report_dir / "mascaras_geradas.csv", [row for row in rows if row["mask_status"].startswith("generated")])

    summary = summarize(rows)
    summary.update(
        {
            "output_root": str(args.output_root),
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "device": device,
            "confidence_threshold": args.confidence_threshold,
            "duplicates_skipped": len(duplicate_rows),
        }
    )
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

