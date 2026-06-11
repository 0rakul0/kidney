import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_MANIFEST = PROJECT_ROOT / "dataset_aumentado" / "dataset_geral" / "manifest.csv"
DEFAULT_MANIFESTS = [
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "intrarenal_multiclass_predictions_dataset_geral"
    / "manifest.csv",
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "intrarenal_unet_multiclass_predictions_dataset_geral"
    / "manifest.csv",
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "intrarenal_multiclass_predictions_dataset_geral_unet085"
    / "manifest.csv",
]
MASK_FIELDS = [
    "predicted_cortex_mask_path",
    "predicted_medulla_mask_path",
    "predicted_central_echo_complex_mask_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recorta mascaras intrarrenais pela mascara renal limpa do dataset_geral."
    )
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--prediction-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def load_binary(path, shape=None):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def main():
    args = parse_args()
    prediction_manifests = args.prediction_manifest or DEFAULT_MANIFESTS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_rows = read_rows(args.dataset_manifest)
    kidney_masks = {
        row["image_id"]: resolve_path(row.get("dataset_mask_path", ""))
        for row in dataset_rows
        if row.get("has_mask", "").lower() == "true"
    }

    all_reports = []
    total_changed = 0
    total_missing = 0

    for manifest_path in prediction_manifests:
        if not manifest_path.exists():
            continue
        rows = read_rows(manifest_path)
        backup_dir = manifest_path.parent / "backups" / f"intrarenal_masks_before_kidney_constraint_{timestamp}"
        report_rows = []
        missing_rows = []

        for row in rows:
            image_id = row.get("image_id", "")
            kidney_path = kidney_masks.get(image_id)
            if kidney_path is None:
                continue
            kidney = load_binary(kidney_path)
            if kidney is None:
                missing_rows.append({"image_id": image_id, "missing": str(kidney_path), "kind": "kidney"})
                continue
            for field in MASK_FIELDS:
                mask_path = resolve_path(row.get(field, ""))
                if mask_path is None:
                    continue
                mask = load_binary(mask_path, kidney.shape)
                if mask is None:
                    missing_rows.append({"image_id": image_id, "missing": row.get(field, ""), "kind": field})
                    continue
                constrained = (mask & kidney).astype(np.uint8)
                removed_pixels = int(mask.sum() - constrained.sum())
                if removed_pixels <= 0:
                    continue
                report_rows.append(
                    {
                        "image_id": image_id,
                        "field": field,
                        "mask_path": str(mask_path),
                        "removed_pixels": removed_pixels,
                        "pixels_after": int(constrained.sum()),
                    }
                )
                if not args.dry_run:
                    destination = backup_dir / field / mask_path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mask_path, destination)
                    cv2.imwrite(str(mask_path), constrained * 255)

        total_changed += len(report_rows)
        total_missing += len(missing_rows)
        report_path = manifest_path.parent / f"intrarenal_masks_constrained_to_kidney_{timestamp}.csv"
        missing_path = manifest_path.parent / f"intrarenal_masks_constraint_missing_{timestamp}.csv"
        if not args.dry_run:
            write_rows(report_path, report_rows)
            write_rows(missing_path, missing_rows)
            summary_path = manifest_path.parent / "summary.json"
            summary = {}
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            summary["last_kidney_constraint_cleanup"] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "script": str(Path(__file__).resolve()),
                "changed_masks": len(report_rows),
                "missing_masks": len(missing_rows),
                "backup_dir": str(backup_dir),
                "report_path": str(report_path),
            }
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
        all_reports.append(
            {
                "manifest": str(manifest_path),
                "changed_masks": len(report_rows),
                "missing_masks": len(missing_rows),
                "backup_dir": str(backup_dir),
            }
        )

    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "prediction_manifests": all_reports,
                "total_changed_masks": total_changed,
                "total_missing_masks": total_missing,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
