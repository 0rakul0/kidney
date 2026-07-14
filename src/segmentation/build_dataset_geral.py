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
DATASET_AUMENTADO_ROOT = PROJECT_ROOT / "dataset_aumentado"
FONTES_ROOT = DATASET_AUMENTADO_ROOT / "fontes"
KIDNEYUS_CAPSULE_ROOT = (
    DATASET_AUMENTADO_ROOT
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1_deduplicated"
)
DEFAULT_OUTPUT_ROOT = DATASET_AUMENTADO_ROOT / "dataset_geral_v2"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_unet.pth"
DEFAULT_REVIEW_CALIBRATION = (
    PROJECT_ROOT
    / "results"
    / "segmentation_experiments"
    / "kidneyus_capsule_deduplicated_benchmark"
    / "pseudomask_review_calibration.json"
)


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
    parser.add_argument("--model", choices=["unet", "unetplusplus", "deeplab", "segformer"], default="unet")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replica o CLAHE utilizado no treinamento da U-Net.",
    )
    parser.add_argument(
        "--tta-horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Combina a predicao original com a predicao da imagem espelhada e "
            "registra a concordancia entre elas para priorizar revisao."
        ),
    )
    parser.add_argument(
        "--review-calibration",
        type=Path,
        default=DEFAULT_REVIEW_CALIBRATION,
        help=(
            "Distribuicao de referencia calculada em imagens manuais. Os "
            "limiares servem apenas para ordenar revisao, nao para aceitar."
        ),
    )
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
    parser.add_argument(
        "--include-256x256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mantem imagens 256x256. O padrao e inclui-las porque a base "
            "canonica kidneyUS Capsule usa essa resolucao."
        ),
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
    """Collect only the canonical manual base and approved external sources.

    Legacy DeepLab outputs (identificada, pseudo_labels and the old expanded
    dataset) are intentionally excluded. Candidate order is significant:
    kidneyUS comes first so its manual mask wins any hash deduplication.
    """
    candidates = []
    candidates.extend(collect_split_dataset(KIDNEYUS_CAPSULE_ROOT, "kidneyus_capsule", "kidneyus_capsule"))
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


def is_legacy_256_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image.shape[:2] == (256, 256)


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
    mask = keep_largest_component((mask > 0).astype(np.uint8)) * 255
    output_path = output_mask_dir / f"{image_id}.png"
    cv2.imwrite(str(output_path), mask)
    return output_path, "existing_mask_copied"


def prepare_tensor(image, img_size, device, clahe=False):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if clahe:
        resized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
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


def keep_largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 2:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def binary_dice(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = int(first.sum() + second.sum())
    if denominator == 0:
        return 1.0
    return float((2.0 * np.logical_and(first, second).sum()) / denominator)


def load_review_reference(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    reference = payload.get("review_reference", {})
    return {
        "confidence": reference.get("confidence_p05"),
        "tta_consistency_dice": reference.get("tta_consistency_dice_p05"),
        "source": str(path),
    }


def analyze_mask(probability, threshold, original_shape, config=None):
    """Describe a prediction without treating image resolution as quality.

    A non-empty prediction is always saved as an unvalidated pseudomask.
    Pixel count, relative area and confidence are descriptive fields used to
    order human review; they are not acceptance criteria.
    """
    config = config or {}
    probability = cv2.resize(
        probability,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = (probability >= threshold).astype(np.uint8)
    if config.get("keep_largest_component", True):
        mask = keep_largest_component(mask)
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

    prediction_available = foreground_pixels > 0
    review_flags = []
    if not prediction_available:
        review_flags.append("empty_prediction")

    return {
        "accepted": prediction_available,
        "prediction_available": prediction_available,
        "rejection_reason": "" if prediction_available else "empty_prediction",
        "review_flags": review_flags,
        "mask": mask,
        "confidence": confidence,
        "foreground_pixels": foreground_pixels,
        "area_ratio": area_ratio,
        "components": len(components),
        "largest_component_pixels": max(components) if components else 0,
    }


def generate_mask(
    bundle,
    image_path,
    image_shape,
    image_id,
    output_mask_dir,
    args,
    device,
    review_reference,
):
    image = read_grayscale_image(image_path)
    tensor = prepare_tensor(image, args.img_size, device, clahe=args.clahe)
    threshold = float(bundle["threshold"])
    with torch.no_grad():
        probability = predict_probability(bundle, tensor, args.img_size)
        if args.tta_horizontal_flip:
            flipped = cv2.flip(image, 1)
            flipped_tensor = prepare_tensor(
                flipped,
                args.img_size,
                device,
                clahe=args.clahe,
            )
            flipped_probability = predict_probability(
                bundle,
                flipped_tensor,
                args.img_size,
            )
            flipped_probability = np.fliplr(flipped_probability)
        else:
            flipped_probability = probability.copy()

    original_binary = probability >= threshold
    flipped_binary = flipped_probability >= threshold
    tta_consistency_dice = binary_dice(original_binary, flipped_binary)
    probability = (probability + flipped_probability) / 2.0

    analysis = analyze_mask(
        probability,
        threshold,
        image_shape,
    )
    analysis["tta_consistency_dice"] = tta_consistency_dice
    confidence_reference = review_reference.get("confidence")
    consistency_reference = review_reference.get("tta_consistency_dice")
    if (
        confidence_reference is not None
        and analysis["confidence"] < float(confidence_reference)
    ):
        analysis["review_flags"].append("confidence_below_manual_reference_p05")
    if (
        consistency_reference is not None
        and tta_consistency_dice < float(consistency_reference)
    ):
        analysis["review_flags"].append("tta_below_manual_reference_p05")
    if not analysis["prediction_available"]:
        analysis["review_priority"] = "high"
    elif len(analysis["review_flags"]) >= 2:
        analysis["review_priority"] = "high"
    elif analysis["review_flags"]:
        analysis["review_priority"] = "medium"
    else:
        analysis["review_priority"] = "routine"

    if analysis["prediction_available"]:
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
        "generated_pseudomasks": sum(1 for row in rows if row["mask_status"] == "generated_unvalidated"),
        "empty_predictions": sum(1 for row in rows if row["mask_status"] == "empty_prediction"),
        "pending_human_review": sum(
            1 for row in rows if row["validation_status"] == "pending_human_review"
        ),
        "pseudomask_review_priority": {
            priority: sum(
                1
                for row in rows
                if row["validation_status"] == "pending_human_review"
                and row["review_priority"] == priority
            )
            for priority in ("high", "medium", "routine")
        },
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
    excluded_256_rows = []
    if not args.include_256x256:
        filtered_candidates = []
        for candidate in candidates:
            if is_legacy_256_image(candidate.image_path):
                excluded_256_rows.append(
                    {
                        "source_name": candidate.source_name,
                        "image_path": str(candidate.image_path),
                        "mask_path": str(candidate.mask_path or ""),
                        "label_source": candidate.label_source,
                    }
                )
            else:
                filtered_candidates.append(candidate)
        candidates = filtered_candidates
    seen_hashes = {}
    used_ids = set()
    rows = []
    duplicate_rows = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = None
    review_reference = load_review_reference(args.review_calibration)
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
            "tta_consistency_dice": "",
            "review_flags": "",
            "review_priority": "",
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
                review_reference,
            )
            mask_quality = {
                "confidence": f"{analysis['confidence']:.6f}",
                "foreground_pixels": analysis["foreground_pixels"],
                "area_ratio": f"{analysis['area_ratio']:.6f}",
                "components": analysis["components"],
                "largest_component_pixels": analysis["largest_component_pixels"],
                "tta_consistency_dice": f"{analysis['tta_consistency_dice']:.6f}",
                "review_flags": ";".join(analysis["review_flags"]),
                "review_priority": analysis["review_priority"],
                "rejection_reason": analysis["rejection_reason"],
            }
            if analysis["prediction_available"]:
                output_mask_path = str(analysis["mask_path"])
                mask_status = "generated_unvalidated"
            else:
                mask_status = "empty_prediction"

        has_mask = output_mask_path != ""
        if mask_status == "existing":
            validation_status = "manual_reference"
        elif mask_status == "generated_unvalidated":
            validation_status = "pending_human_review"
        elif mask_status == "empty_prediction":
            validation_status = "no_prediction"
        else:
            validation_status = "not_available"
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
                "validation_status": validation_status,
                "mask_confidence_threshold": "",
                **mask_quality,
            }
        )

        if index % 100 == 0:
            print(f"Processadas {index}/{len(candidates)} imagens candidatas")

    write_csv(args.output_root / "manifest.csv", rows)
    write_csv(report_dir / "duplicadas_por_hash.csv", duplicate_rows)
    write_csv(report_dir / "excluidas_256x256.csv", excluded_256_rows)
    write_csv(report_dir / "faltando_mascara.csv", [row for row in rows if row["has_mask"] == "false"])
    write_csv(
        report_dir / "pseudomascaras_nao_validadas.csv",
        [row for row in rows if row["mask_status"] == "generated_unvalidated"],
    )
    write_csv(
        report_dir / "predicoes_vazias.csv",
        [row for row in rows if row["mask_status"] == "empty_prediction"],
    )
    review_rows = sorted(
        [
            row
            for row in rows
            if row["validation_status"] in {"pending_human_review", "no_prediction"}
        ],
        key=lambda row: (
            {"high": 0, "medium": 1, "routine": 2}.get(
                row["review_priority"],
                3,
            ),
            float(row["tta_consistency_dice"] or 0.0),
            float(row["confidence"] or 0.0),
        ),
    )
    write_csv(report_dir / "fila_revisao.csv", review_rows)

    summary = summarize(rows)
    summary.update(
        {
            "output_root": str(args.output_root),
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "device": device,
            "model_threshold": float(bundle["threshold"]) if bundle is not None else None,
            "clahe": args.clahe,
            "tta_horizontal_flip": args.tta_horizontal_flip,
            "review_reference": review_reference,
            "quality_policy": (
                "all non-empty predictions are stored as unvalidated pseudomasks; "
                "no absolute pixel-count or area threshold is used for acceptance"
            ),
            "duplicates_skipped": len(duplicate_rows),
            "excluded_256x256": len(excluded_256_rows),
        }
    )
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

