import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_geral"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mantem apenas a maior componente conectada nas mascaras renais do dataset_geral."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {
        "total_unique_images": len(rows),
        "with_mask": sum(row.get("has_mask", "").lower() == "true" for row in rows),
        "without_mask": sum(row.get("has_mask", "").lower() != "true" for row in rows),
        "existing_masks": sum(row.get("mask_status", "") == "existing" for row in rows),
        "generated_masks_accepted": sum(row.get("mask_status", "").startswith("generated_accepted") for row in rows),
        "generated_masks_rejected": sum(row.get("mask_status", "").startswith("generated_rejected") for row in rows),
        "missing_not_generated": sum(row.get("mask_status", "") == "missing_not_generated" for row in rows),
        "by_source": {},
    }
    for row in rows:
        source = row.get("source_name", "")
        summary["by_source"].setdefault(source, {"images": 0, "with_mask": 0})
        summary["by_source"][source]["images"] += 1
        if row.get("has_mask", "").lower() == "true":
            summary["by_source"][source]["with_mask"] += 1
    return summary


def mask_components(mask):
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = [
        (idx, int(stats[idx, cv2.CC_STAT_AREA]))
        for idx in range(1, num_labels)
        if int(stats[idx, cv2.CC_STAT_AREA]) > 0
    ]
    return binary, labels, components


def keep_largest(binary, labels, components):
    if not components:
        return binary * 0
    largest_label, _ = max(components, key=lambda item: item[1])
    return (labels == largest_label).astype(np.uint8)


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    manifest_path = dataset_root / "manifest.csv"
    report_dir = dataset_root / "relatorios"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = dataset_root / "backups" / f"kidney_masks_before_largest_component_{timestamp}"

    rows = read_rows(manifest_path)
    changed_rows = []
    unreadable_rows = []

    for row in rows:
        mask_value = row.get("dataset_mask_path", "")
        if row.get("has_mask", "").lower() != "true" or not mask_value:
            continue
        mask_path = Path(mask_value)
        if not mask_path.is_absolute():
            mask_path = PROJECT_ROOT / mask_path
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            unreadable_rows.append({"image_id": row.get("image_id", ""), "dataset_mask_path": str(mask_path)})
            continue
        binary, labels, components = mask_components(mask)
        if len(components) <= 1:
            row["components"] = len(components)
            row["largest_component_pixels"] = components[0][1] if components else 0
            continue

        cleaned = keep_largest(binary, labels, components)
        removed_pixels = int(binary.sum() - cleaned.sum())
        changed_rows.append(
            {
                "image_id": row.get("image_id", ""),
                "dataset_mask_path": str(mask_path),
                "components_before": len(components),
                "largest_component_pixels": int(cleaned.sum()),
                "removed_pixels": removed_pixels,
                "component_pixels_before": ";".join(str(area) for _, area in sorted(components, key=lambda item: item[1], reverse=True)),
            }
        )
        row["foreground_pixels"] = int(cleaned.sum())
        row["area_ratio"] = f"{float(cleaned.sum() / max(cleaned.size, 1)):.6f}"
        row["components"] = 1
        row["largest_component_pixels"] = int(cleaned.sum())
        if not args.dry_run:
            relative = mask_path.relative_to(dataset_root) if mask_path.is_relative_to(dataset_root) else Path(mask_path.name)
            destination = backup_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_path, destination)
            cv2.imwrite(str(mask_path), cleaned * 255)

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"mascaras_rim_limpas_maior_componente_{timestamp}.csv"
    unreadable_path = report_dir / f"mascaras_rim_inlegiveis_maior_componente_{timestamp}.csv"

    if not args.dry_run:
        write_rows(manifest_path, rows)
        write_rows(report_path, changed_rows)
        write_rows(unreadable_path, unreadable_rows)
        summary = summarize(rows)
        previous_summary_path = dataset_root / "summary.json"
        if previous_summary_path.exists():
            with previous_summary_path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            summary.update(
                {
                    key: value
                    for key, value in previous.items()
                    if key not in summary and key not in {"last_largest_component_cleanup"}
                }
            )
        summary["last_largest_component_cleanup"] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "changed_masks": len(changed_rows),
            "unreadable_masks": len(unreadable_rows),
            "backup_dir": str(backup_dir),
            "report_path": str(report_path),
        }
        with previous_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "dataset_root": str(dataset_root),
                "changed_masks": len(changed_rows),
                "unreadable_masks": len(unreadable_rows),
                "backup_dir": str(backup_dir),
                "report_path": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
