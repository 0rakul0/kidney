import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.dataset import KidneyDataset
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.tools.benchmark_models import evaluate_model


DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1_deduplicated"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "segmentation_experiments"
    / "kidneyus_capsule_deduplicated_benchmark"
)
ARTICLE_FIGURE = (
    PROJECT_ROOT
    / "artigo"
    / "SBBD_2026___Jefferson"
    / "figures"
    / "capsule_good_failure_comparison.png"
)
CHECKPOINTS = {
    "unet": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_unet.pth",
    "unetplusplus": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_unetplusplus.pth",
    "deeplab": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_deeplab.pth",
    "segformer": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_segformer.pth",
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def predict_probability(bundle, tensor):
    model = bundle["model"]
    name = bundle["display_name"]
    with torch.no_grad():
        if name == "SegFormer":
            logits = model(pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(256, 256), mode="bilinear", align_corners=False
            )
        elif name == "DeepLab":
            logits = model(tensor)["out"]
        else:
            logits = model(tensor)
    return torch.sigmoid(logits).squeeze().cpu().numpy()


def overlap_metrics(reference, prediction):
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(reference, prediction).sum()
    union = np.logical_or(reference, prediction).sum()
    predicted = prediction.sum()
    target = reference.sum()
    dice = 2.0 * intersection / max(predicted + target, 1)
    iou = intersection / max(union, 1)
    precision = intersection / max(predicted, 1)
    recall = intersection / max(target, 1)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def unet_per_image(bundle, device):
    rows = []
    image_dir = DATASET_ROOT / "test" / "image"
    mask_dir = DATASET_ROOT / "test" / "mask"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for image_path in sorted(image_dir.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(mask_dir / image_path.name), cv2.IMREAD_GRAYSCALE)
        image_256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        reference_256 = cv2.resize(
            reference, (256, 256), interpolation=cv2.INTER_NEAREST
        ) > 0
        enhanced = clahe.apply(image_256).astype(np.float32) / 255.0
        tensor = (
            torch.from_numpy(np.stack([enhanced] * 3))
            .unsqueeze(0)
            .to(device)
        )
        probability = predict_probability(bundle, tensor)
        prediction = probability >= float(bundle["threshold"])
        metrics = overlap_metrics(reference_256, prediction)
        rows.append(
            {
                "filename": image_path.name,
                **{key: f"{value:.6f}" for key, value in metrics.items()},
            }
        )
    return rows


def overlay_contours(image, reference, prediction):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    reference_contours, _ = cv2.findContours(
        reference.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    prediction_contours, _ = cv2.findContours(
        prediction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(rgb, reference_contours, -1, (0, 220, 0), 2)
    cv2.drawContours(rgb, prediction_contours, -1, (255, 70, 40), 2)
    return rgb


def build_qualitative_figure(bundle, rows, device):
    ranked = sorted(rows, key=lambda row: float(row["dice"]))
    selected = [
        (ranked[-1], "Caso favorável"),
        (ranked[0], "Caso desafiador"),
    ]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.4))
    titles = [
        "Ultrassonografia",
        "Cápsula manual",
        "Cápsula prevista",
        "Sobreposição",
    ]
    for row_index, (row, label) in enumerate(selected):
        filename = row["filename"]
        image = cv2.imread(
            str(DATASET_ROOT / "test" / "image" / filename), cv2.IMREAD_GRAYSCALE
        )
        reference = cv2.imread(
            str(DATASET_ROOT / "test" / "mask" / filename), cv2.IMREAD_GRAYSCALE
        )
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        reference = cv2.resize(
            reference, (256, 256), interpolation=cv2.INTER_NEAREST
        ) > 0
        enhanced = clahe.apply(image).astype(np.float32) / 255.0
        tensor = (
            torch.from_numpy(np.stack([enhanced] * 3))
            .unsqueeze(0)
            .to(device)
        )
        prediction = (
            predict_probability(bundle, tensor) >= float(bundle["threshold"])
        )
        panels = [
            image,
            reference,
            prediction,
            overlay_contours(image, reference, prediction),
        ]
        for column, panel in enumerate(panels):
            axis = axes[row_index, column]
            axis.imshow(panel, cmap="gray" if column < 3 else None)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(titles[column], fontsize=11, pad=7)
        axes[row_index, 0].set_ylabel(
            f"{label}\nDice={float(row['dice']):.3f}; IoU={float(row['iou']):.3f}",
            fontsize=10,
            labelpad=10,
        )
    plt.tight_layout(rect=(0.02, 0.03, 1, 1), h_pad=1.1, w_pad=0.5)
    ARTICLE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ARTICLE_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "best_case": selected[0][0],
        "challenging_case": selected[1][0],
        "figure": str(ARTICLE_FIGURE),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = KidneyDataset(
        DATASET_ROOT / "test", img_size=256, augment=False, clahe=True
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    results = []
    bundles = {}
    for model_key, checkpoint in CHECKPOINTS.items():
        bundle = load_model_bundle(
            model_key,
            device=device,
            checkpoint_path=checkpoint,
            model_dir=PROJECT_ROOT / "models",
        )
        bundles[model_key] = bundle
        metrics = evaluate_model(
            bundle["model"],
            bundle["display_name"],
            bundle["threshold"],
            loader,
            dataset,
            device,
            256,
        )
        results.append(
            {
                "model_key": model_key,
                "model": bundle["display_name"],
                "threshold": float(bundle["threshold"]),
                "dice": float(metrics["Dice"]),
                "iou": float(metrics["IoU"]),
                "precision": float(metrics["Precision"]),
                "recall": float(metrics["Recall"]),
                "f1": float(metrics["F1"]),
                "hausdorff": float(metrics["Hausdorff"]),
                "fps": float(metrics["FPS"]),
                "checkpoint": str(checkpoint),
            }
        )

    results.sort(key=lambda row: (row["dice"], row["iou"]), reverse=True)
    write_csv(OUTPUT_ROOT / "benchmark_results.csv", results)
    (OUTPUT_ROOT / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    per_image = unet_per_image(bundles["unet"], device)
    write_csv(OUTPUT_ROOT / "unet_per_image_metrics.csv", per_image)
    qualitative = build_qualitative_figure(
        bundles["unet"], per_image, device
    )
    summary = {
        "dataset_root": str(DATASET_ROOT),
        "test_samples": len(dataset),
        "device": device,
        "clahe": True,
        "best_model": results[0],
        "qualitative": qualitative,
        "results_csv": str(OUTPUT_ROOT / "benchmark_results.csv"),
        "per_image_csv": str(OUTPUT_ROOT / "unet_per_image_metrics.csv"),
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
