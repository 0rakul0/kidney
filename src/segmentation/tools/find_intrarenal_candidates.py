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


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_geral"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "reference_matching"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Busca candidatos para revisao intrarrenal no dataset_geral. "
            "O script usa a mascara renal para recortar a ROI e ranqueia "
            "imagens com padrao visual mais saudavel ou mais sugestivo de "
            "alteracao textural."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-area-ratio", type=float, default=0.025)
    parser.add_argument("--max-area-ratio", type=float, default=0.70)
    parser.add_argument("--pad", type=int, default=24)
    parser.add_argument(
        "--include-generated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inclui pseudo-mascaras aceitas, alem de mascaras existentes.",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Usa apenas mascaras existentes/manuais ou primarias.",
    )
    parser.add_argument(
        "--panel-prefix",
        default="",
        help="Prefixo opcional para os paineis gerados.",
    )
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "sim"}


def eligible(row, args):
    if not normalize_bool(row.get("has_mask")):
        return False
    image_path = Path(row.get("dataset_image_path", ""))
    mask_path = Path(row.get("dataset_mask_path", ""))
    if not image_path.exists() or not mask_path.exists():
        return False

    mask_status = row.get("mask_status", "")
    if args.manual_only:
        return mask_status == "existing"
    if not args.include_generated and mask_status != "existing":
        return False
    return True


def load_pair(row):
    image = cv2.imread(row["dataset_image_path"], cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(row["dataset_mask_path"], cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None, None
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, mask


def bbox_from_mask(mask, pad):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, mask.shape[1] - 1)
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad, mask.shape[0] - 1)
    return x1, y1, x2, y2


def crop_roi(image, mask, pad):
    bbox = bbox_from_mask(mask, pad)
    if bbox is None:
        return None, None, None
    x1, y1, x2, y2 = bbox
    return image[y1 : y2 + 1, x1 : x2 + 1], mask[y1 : y2 + 1, x1 : x2 + 1], bbox


def safe_percentile(values, q):
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def count_dark_pyramid_candidates(roi_image, roi_mask, renal_mean):
    blurred = cv2.GaussianBlur(roi_image, (5, 5), 0)
    dark = ((blurred < renal_mean * 0.72) & (roi_mask > 0)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    components = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 20 <= area <= 2500:
            components += 1
    return components, dark


def compute_features(image, mask, args):
    total_pixels = image.shape[0] * image.shape[1]
    foreground_pixels = int(mask.sum())
    area_ratio = foreground_pixels / max(total_pixels, 1)
    if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
        return None

    roi_image, roi_mask, bbox = crop_roi(image, mask, args.pad)
    if roi_image is None:
        return None

    renal_pixels = roi_image[roi_mask > 0].astype(np.float32)
    if renal_pixels.size < 800:
        return None

    mean = float(renal_pixels.mean())
    std = float(renal_pixels.std())
    p10 = safe_percentile(renal_pixels, 10)
    p90 = safe_percentile(renal_pixels, 90)
    contrast = float((p90 - p10) / 255.0)
    normalized_mean = float(mean / 255.0)
    normalized_std = float(std / 255.0)

    laplacian = cv2.Laplacian(roi_image, cv2.CV_32F)
    texture = float(np.abs(laplacian[roi_mask > 0]).mean() / 255.0)

    pyramid_components, dark_map = count_dark_pyramid_candidates(roi_image, roi_mask, mean)
    dark_ratio = float(dark_map.sum() / max(int(roi_mask.sum()), 1))

    healthy_score = (
        0.34 * min(contrast / 0.45, 1.0)
        + 0.26 * min(pyramid_components / 8.0, 1.0)
        + 0.20 * min(dark_ratio / 0.28, 1.0)
        + 0.20 * min(normalized_std / 0.24, 1.0)
    )
    suspicious_score = (
        0.34 * min(normalized_mean / 0.62, 1.0)
        + 0.28 * (1.0 - min(contrast / 0.38, 1.0))
        + 0.20 * (1.0 - min(pyramid_components / 5.0, 1.0))
        + 0.18 * min(texture / 0.24, 1.0)
    )

    return {
        "foreground_pixels": foreground_pixels,
        "area_ratio": area_ratio,
        "bbox": bbox,
        "renal_mean": mean,
        "renal_std": std,
        "renal_p10": p10,
        "renal_p90": p90,
        "contrast_p90_p10": contrast,
        "normalized_mean": normalized_mean,
        "normalized_std": normalized_std,
        "texture_laplacian": texture,
        "dark_pyramid_candidate_components": pyramid_components,
        "dark_pyramid_candidate_ratio": dark_ratio,
        "healthy_candidate_score": float(healthy_score),
        "suspicious_candidate_score": float(suspicious_score),
    }


def make_panel(row, output_path, label):
    image, mask = load_pair(row)
    if image is None:
        return
    roi_image, roi_mask, _ = crop_roi(image, mask, 24)
    if roi_image is None:
        return

    roi_image = cv2.resize(roi_image, (320, 240), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(roi_mask, (320, 240), interpolation=cv2.INTER_NEAREST)

    rgb = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    overlay = rgb.copy()
    overlay[roi_mask > 0] = (0, 160, 255)
    overlay = cv2.addWeighted(rgb, 0.76, overlay, 0.24, 0)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 180, 255), 2)

    mask_rgb = cv2.cvtColor((roi_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    panel = cv2.hconcat([rgb, overlay, mask_rgb])
    cv2.putText(panel, label[:80], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "contorno amarelo = mascara renal usada como ROI",
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), panel)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def enrich_row(row, features):
    output = {
        "image_id": row.get("image_id", ""),
        "source_name": row.get("source_name", ""),
        "label_source": row.get("label_source", ""),
        "mask_status": row.get("mask_status", ""),
        "dataset_image_path": row.get("dataset_image_path", ""),
        "dataset_mask_path": row.get("dataset_mask_path", ""),
        "model_confidence": row.get("confidence", ""),
    }
    for key, value in features.items():
        if key == "bbox":
            output[key] = ",".join(str(item) for item in value)
        elif isinstance(value, float):
            output[key] = f"{value:.6f}"
        else:
            output[key] = value
    return output


def main():
    args = parse_args()
    manifest_path = args.dataset_root / "manifest.csv"
    rows = read_manifest(manifest_path)

    scored_rows = []
    for row in rows:
        if not eligible(row, args):
            continue
        image, mask = load_pair(row)
        if image is None:
            continue
        features = compute_features(image, mask, args)
        if features is None:
            continue
        scored_rows.append(enrich_row(row, features))

    healthy = sorted(scored_rows, key=lambda item: float(item["healthy_candidate_score"]), reverse=True)
    suspicious = sorted(scored_rows, key=lambda item: float(item["suspicious_candidate_score"]), reverse=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "intrarenal_candidate_scores.csv", scored_rows)
    write_csv(args.output_root / "intrarenal_healthy_candidates.csv", healthy[: args.top_k])
    write_csv(args.output_root / "intrarenal_suspicious_candidates.csv", suspicious[: args.top_k])

    panel_dir = args.output_root / "intrarenal_candidate_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    for rank, row in enumerate(healthy[: min(args.top_k, 12)], 1):
        make_panel(
            row,
            panel_dir / f"{args.panel_prefix}healthy_candidate_{rank:02d}.png",
            f"healthy {rank:02d} score={row['healthy_candidate_score']} {row['image_id']}",
        )
    for rank, row in enumerate(suspicious[: min(args.top_k, 12)], 1):
        make_panel(
            row,
            panel_dir / f"{args.panel_prefix}suspicious_candidate_{rank:02d}.png",
            f"suspicious {rank:02d} score={row['suspicious_candidate_score']} {row['image_id']}",
        )

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "eligible_scored_images": len(scored_rows),
        "top_k": args.top_k,
        "include_generated": args.include_generated,
        "manual_only": args.manual_only,
        "outputs": {
            "all_scores": str(args.output_root / "intrarenal_candidate_scores.csv"),
            "healthy_candidates": str(args.output_root / "intrarenal_healthy_candidates.csv"),
            "suspicious_candidates": str(args.output_root / "intrarenal_suspicious_candidates.csv"),
            "panels": str(panel_dir),
        },
        "note": (
            "Os rankings sao heuristicas de triagem visual, nao diagnostico. "
            "A lista suspeita indica alteracao textural potencial e precisa de revisao."
        ),
    }
    (args.output_root / "intrarenal_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
