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

from src.segmentation.build_dataset_geral import binary_dice
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.run_capsule_model_consensus import predict_mask


FOLDS_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_oof_5fold"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "segmentation_experiments"
    / "capsule_unet_deeplab_consensus_oof"
)


def rounded_hundredth(value):
    return float(np.round(value + 1e-12, 2))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    fold_rows = []
    for fold_index in range(1, 6):
        fold = f"{fold_index:02d}"
        fold_root = FOLDS_ROOT / f"fold_{fold}"
        bundles = {
            "unet": load_model_bundle(
                "unet",
                device=device,
                checkpoint_path=PROJECT_ROOT
                / "models"
                / f"capsule_oof_unet_fold_{fold}.pth",
            ),
            "deeplab": load_model_bundle(
                "deeplab",
                device=device,
                checkpoint_path=PROJECT_ROOT
                / "models"
                / f"capsule_oof_deeplab_fold_{fold}.pth",
            ),
        }
        fold_start = len(rows)
        for image_path in sorted((fold_root / "test" / "image").glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            reference = (
                cv2.imread(
                    str(fold_root / "test" / "mask" / image_path.name),
                    cv2.IMREAD_GRAYSCALE,
                )
                > 0
            )
            unet = predict_mask(bundles["unet"], image, 256, device) > 0
            deeplab = predict_mask(bundles["deeplab"], image, 256, device) > 0
            rows.append(
                {
                    "fold": fold_index,
                    "filename": image_path.name,
                    "model_consensus_dice": binary_dice(unet, deeplab),
                    "unet_manual_dice": binary_dice(unet, reference),
                    "deeplab_manual_dice": binary_dice(deeplab, reference),
                }
            )
        current = rows[fold_start:]
        fold_rows.append(
            {
                "fold": fold_index,
                "samples": len(current),
                "consensus_mean": float(
                    np.mean([row["model_consensus_dice"] for row in current])
                ),
                "unet_manual_dice": float(
                    np.mean([row["unet_manual_dice"] for row in current])
                ),
                "deeplab_manual_dice": float(
                    np.mean([row["deeplab_manual_dice"] for row in current])
                ),
            }
        )
        del bundles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    consensus = np.asarray(
        [row["model_consensus_dice"] for row in rows], dtype=float
    )
    minimum_manual = np.asarray(
        [
            min(row["unet_manual_dice"], row["deeplab_manual_dice"])
            for row in rows
        ],
        dtype=float,
    )
    p10 = float(np.percentile(consensus, 10))
    p25 = float(np.percentile(consensus, 25))
    summary = {
        "method": "5-fold out-of-fold calibration grouped by examination",
        "samples": len(rows),
        "unique_filenames": len({row["filename"] for row in rows}),
        "each_image_evaluated_once": len(rows)
        == len({row["filename"] for row in rows})
        == 468,
        "device": device,
        "folds": fold_rows,
        "consensus": {
            "mean": float(consensus.mean()),
            "minimum": float(consensus.min()),
            "p10": p10,
            "p25": p25,
            "median": float(np.median(consensus)),
            "correlation_with_minimum_manual_dice": float(
                np.corrcoef(consensus, minimum_manual)[0, 1]
            ),
        },
        "manual_performance": {
            "unet_mean_dice": float(
                np.mean([row["unet_manual_dice"] for row in rows])
            ),
            "deeplab_mean_dice": float(
                np.mean([row["deeplab_manual_dice"] for row in rows])
            ),
        },
        "operational_thresholds": {
            "priority_below": rounded_hundredth(p10),
            "routine_at_or_above": rounded_hundredth(p25),
            "origin": (
                "p10 and p25 of 468 out-of-fold model-consensus scores, "
                "rounded to two decimals"
            ),
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "calibration_oof.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
