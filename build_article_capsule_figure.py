from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.model_loader import load_model_bundle


DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
    / "test"
)
CHECKPOINT = PROJECT_ROOT / "models" / "kidneyus_capsule_unet.pth"
OUTPUT = (
    PROJECT_ROOT
    / "artigo"
    / "SBBD_2026___Jefferson"
    / "figures"
    / "capsule_good_failure_comparison.png"
)

CASES = [
    ("291_IM-0708-0047_anon.png", "Caso favorável"),
    ("249_IM-0589-0014_anon.png", "Caso desafiador"),
]


def dice_iou(reference, prediction):
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(reference, prediction).sum()
    union = np.logical_or(reference, prediction).sum()
    dice = (2.0 * intersection) / max(reference.sum() + prediction.sum(), 1)
    iou = intersection / max(union, 1)
    return float(dice), float(iou)


def predict(bundle, image, device):
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    normalized = enhanced.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.stack([normalized] * 3)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = bundle["model"](tensor)
    probability = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probability >= float(bundle["threshold"])


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model_bundle(
        "unet",
        device=device,
        checkpoint_path=CHECKPOINT,
        model_dir=PROJECT_ROOT / "models",
    )

    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.4))
    column_titles = [
        "Ultrassonografia",
        "Cápsula manual",
        "Cápsula prevista",
        "Sobreposição",
    ]

    for row, (filename, case_title) in enumerate(CASES):
        image = cv2.imread(
            str(DATASET_ROOT / "image" / filename), cv2.IMREAD_GRAYSCALE
        )
        reference = cv2.imread(
            str(DATASET_ROOT / "mask" / filename), cv2.IMREAD_GRAYSCALE
        )
        if image is None or reference is None:
            raise FileNotFoundError(filename)

        image_256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        reference_256 = (
            cv2.resize(reference, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
        )
        prediction = predict(bundle, image, device)
        dice, iou = dice_iou(reference_256, prediction)
        overlay = overlay_contours(image_256, reference_256, prediction)

        panels = [image_256, reference_256, prediction, overlay]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray" if col < 3 else None)
            ax.axis("off")
            if row == 0:
                ax.set_title(column_titles[col], fontsize=11, pad=7)

        axes[row, 0].set_ylabel(
            f"{case_title}\nDice={dice:.3f}; IoU={iou:.3f}",
            fontsize=10,
            labelpad=10,
        )

    fig.text(
        0.79,
        0.012,
        "contorno manual",
        color=(0.0, 0.55, 0.0),
        fontsize=9,
        ha="right",
    )
    fig.text(
        0.805,
        0.012,
        "—",
        color=(0.0, 0.55, 0.0),
        fontsize=11,
        ha="center",
    )
    fig.text(
        0.91,
        0.012,
        "contorno previsto",
        color=(0.85, 0.1, 0.03),
        fontsize=9,
        ha="center",
    )
    plt.tight_layout(rect=(0.02, 0.04, 1, 1), h_pad=1.1, w_pad=0.5)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
