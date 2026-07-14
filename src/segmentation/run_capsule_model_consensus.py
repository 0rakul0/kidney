import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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
DEFAULT_EXTERNAL_MANIFEST = (
    PROJECT_ROOT / "dataset_aumentado" / "dataset_geral_v2" / "manifest.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "segmentation_experiments"
    / "capsule_unet_deeplab_consensus"
)
DEFAULT_FIGURE = (
    PROJECT_ROOT
    / "artigo"
    / "SBBD_2026___Jefferson"
    / "figures"
    / "capsule_model_consensus.png"
)
DEFAULT_OOF_CALIBRATION = (
    PROJECT_ROOT
    / "results"
    / "segmentation_experiments"
    / "capsule_unet_deeplab_consensus_oof"
    / "summary.json"
)
CHECKPOINTS = {
    "unet": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_unet.pth",
    "deeplab": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_deeplab.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibra e executa o consenso entre U-Net e DeepLabV3 para "
            "priorizar revisao de pseudomascaras externas."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--calibration-split", choices=["val", "test"], default="val")
    parser.add_argument("--external-manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--medium-threshold", type=float, default=0.89)
    parser.add_argument("--routine-threshold", type=float, default=0.94)
    parser.add_argument(
        "--oof-calibration",
        type=Path,
        default=DEFAULT_OOF_CALIBRATION,
    )
    return parser.parse_args()


def predict_mask(bundle, image, img_size, device):
    original = prepare_tensor(image, img_size, device, clahe=True)
    flipped = prepare_tensor(cv2.flip(image, 1), img_size, device, clahe=True)
    with torch.no_grad():
        probability = predict_probability(bundle, original, img_size)
        flipped_probability = np.fliplr(
            predict_probability(bundle, flipped, img_size)
        )
    probability = (probability + flipped_probability) / 2.0
    probability = cv2.resize(
        probability,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return keep_largest_component(
        probability >= float(bundle["threshold"])
    ).astype(np.uint8)


def load_bundles(device):
    return {
        name: load_model_bundle(
            name,
            device=device,
            checkpoint_path=checkpoint,
        )
        for name, checkpoint in CHECKPOINTS.items()
    }


def calibrate(args, bundles, device):
    image_dir = args.dataset_root / args.calibration_split / "image"
    mask_dir = args.dataset_root / args.calibration_split / "mask"
    rows = []
    for image_path in sorted(image_dir.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        reference = (
            cv2.imread(str(mask_dir / image_path.name), cv2.IMREAD_GRAYSCALE) > 0
        )
        unet = predict_mask(bundles["unet"], image, args.img_size, device) > 0
        deeplab = predict_mask(bundles["deeplab"], image, args.img_size, device) > 0
        rows.append(
            {
                "filename": image_path.name,
                "model_consensus_dice": binary_dice(unet, deeplab),
                "unet_manual_dice": binary_dice(unet, reference),
                "deeplab_manual_dice": binary_dice(deeplab, reference),
            }
        )

    consensus = np.asarray(
        [row["model_consensus_dice"] for row in rows],
        dtype=float,
    )
    minimum_manual = np.asarray(
        [
            min(row["unet_manual_dice"], row["deeplab_manual_dice"])
            for row in rows
        ],
        dtype=float,
    )
    return rows, {
        "split": args.calibration_split,
        "samples": len(rows),
        "consensus_mean": float(consensus.mean()),
        "consensus_p10": float(np.percentile(consensus, 10)),
        "consensus_p25": float(np.percentile(consensus, 25)),
        "consensus_median": float(np.median(consensus)),
        "consensus_minimum": float(consensus.min()),
        "correlation_with_minimum_manual_dice": float(
            np.corrcoef(consensus, minimum_manual)[0, 1]
        ),
        "operational_thresholds": {
            "priority": f"consensus < {args.medium_threshold:.2f}",
            "intermediate": (
                f"{args.medium_threshold:.2f} <= consensus "
                f"< {args.routine_threshold:.2f}"
            ),
            "routine": f"consensus >= {args.routine_threshold:.2f}",
        },
        "threshold_origin": (
            "rounded values near percentiles 10 and 25 of the manually "
            "annotated validation distribution"
        ),
    }


def read_external_rows(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row["source_name"] == "monai_renal_png"
        ]


def classify_consensus(unet, deeplab, score, args):
    unet_present = bool(unet.any())
    deeplab_present = bool(deeplab.any())
    if not unet_present and not deeplab_present:
        return "no_prediction", "both_empty"
    if unet_present != deeplab_present:
        return "priority", "single_model_prediction"
    if score < args.medium_threshold:
        return "priority", "low_consensus"
    if score < args.routine_threshold:
        return "intermediate", "intermediate_consensus"
    return "routine", "high_consensus"


def process_external(args, bundles, device):
    deep_mask_dir = args.output_root / "deeplab_masks"
    deep_mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    external = read_external_rows(args.external_manifest)
    for index, source in enumerate(external, 1):
        image = cv2.imread(source["dataset_image_path"], cv2.IMREAD_GRAYSCALE)
        unet = predict_mask(bundles["unet"], image, args.img_size, device)
        deeplab = predict_mask(bundles["deeplab"], image, args.img_size, device)
        if unet.any() and deeplab.any():
            score = binary_dice(unet, deeplab)
        else:
            score = 0.0
        category, reason = classify_consensus(unet, deeplab, score, args)
        deep_path = ""
        if deeplab.any():
            deep_path = str(deep_mask_dir / f"{source['image_id']}.png")
            cv2.imwrite(deep_path, deeplab * 255)
        rows.append(
            {
                "image_id": source["image_id"],
                "image_path": source["dataset_image_path"],
                "unet_mask_path": source["dataset_mask_path"],
                "deeplab_mask_path": deep_path,
                "unet_prediction": str(bool(unet.any())).lower(),
                "deeplab_prediction": str(bool(deeplab.any())).lower(),
                "model_consensus_dice": f"{score:.6f}",
                "review_category": category,
                "review_reason": reason,
            }
        )
        if index % 100 == 0:
            print(f"Consenso processado: {index}/{len(external)}")
    return rows


def overlay(image, unet, deeplab):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for mask, color in ((unet, (0, 220, 0)), (deeplab, (255, 70, 40))):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(rgb, contours, -1, color, 3)
    return rgb


def choose_example(rows, category, target):
    eligible = [
        row
        for row in rows
        if row["review_category"] == category
        and row["unet_prediction"] == "true"
        and row["deeplab_prediction"] == "true"
    ]
    if not eligible:
        raise RuntimeError(f"Nenhum exemplo disponivel para {category}")
    return min(
        eligible,
        key=lambda row: abs(float(row["model_consensus_dice"]) - target),
    )


def build_figure(args, rows):
    selected = [
        ("routine", 0.97, "Consenso alto"),
        ("intermediate", 0.915, "Consenso intermediário"),
        ("priority", 0.80, "Consenso baixo"),
    ]
    examples = [
        (choose_example(rows, category, target), label)
        for category, target, label in selected
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 7.7))
    titles = ["Ultrassonografia", "U-Net", "DeepLabV3", "Sobreposição"]
    for row_index, (row, label) in enumerate(examples):
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        unet = cv2.imread(row["unet_mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        deeplab = cv2.imread(row["deeplab_mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        panels = [image, unet, deeplab, overlay(image, unet, deeplab)]
        for column, panel in enumerate(panels):
            axis = axes[row_index, column]
            axis.imshow(panel, cmap="gray" if column < 3 else None)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(titles[column], fontsize=11, pad=7)
        axes[row_index, 0].text(
            0.02,
            0.04,
            f"{label}\nDice={float(row['model_consensus_dice']):.3f}",
            transform=axes[row_index, 0].transAxes,
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
        )
    plt.tight_layout(pad=0.5, h_pad=0.7, w_pad=0.4)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [row["image_id"] for row, _ in examples]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundles = load_bundles(device)
    calibration_rows, legacy_calibration = calibrate(args, bundles, device)
    if args.oof_calibration.exists():
        with args.oof_calibration.open("r", encoding="utf-8") as file:
            calibration = json.load(file)
    else:
        calibration = legacy_calibration
    external_rows = process_external(args, bundles, device)
    examples = build_figure(args, external_rows)

    write_csv(args.output_root / "calibration.csv", calibration_rows)
    write_csv(args.output_root / "external_consensus.csv", external_rows)
    counts = {
        category: sum(
            row["review_category"] == category for row in external_rows
        )
        for category in ("routine", "intermediate", "priority", "no_prediction")
    }
    summary = {
        "device": device,
        "models": {
            name: {
                "checkpoint": str(bundle["checkpoint_path"]),
                "threshold": float(bundle["threshold"]),
            }
            for name, bundle in bundles.items()
        },
        "calibration": calibration,
        "external_images": len(external_rows),
        "external_counts": counts,
        "figure": str(args.figure),
        "figure_examples": examples,
        "interpretation": (
            "model agreement is a review-prioritization indicator and is not "
            "equivalent to validation against a manual annotation"
        ),
    }
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
