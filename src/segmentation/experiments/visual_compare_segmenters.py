import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.model_loader import load_model_bundle

DEFAULT_LOCAL_RESULTS = (
    PROJECT_ROOT / "results" / "segmentation_experiments" / "dataset_variant_comparison.csv"
)
DEFAULT_EXTERNAL_WEIGHTS_ROOT = Path(r"E:\weights\weights")
DEFAULT_DATASET_SPLIT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
    / "test"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "qualitative_comparison"
DEFAULT_IMAGE_NAME = "1_IM-0001-0059_anon.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera uma figura comparativa em uma imagem com GT, kidneyUS "
            "nnUNet e os melhores segmentadores locais."
        )
    )
    parser.add_argument("--image-name", type=str, default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--dataset-split", type=Path, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--local-results", type=Path, default=DEFAULT_LOCAL_RESULTS)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_EXTERNAL_WEIGHTS_ROOT)
    parser.add_argument(
        "--external-group",
        choices=["mixed", "annotator_1", "annotator_2"],
        default="mixed",
        help="Grupo de pesos kidneyUS usado como reproducao de referencia.",
    )
    parser.add_argument(
        "--external-task",
        choices=["Task001_KidneyCapsule", "Task002_KidneyRegions"],
        default="Task001_KidneyCapsule",
        help="Tarefa externa a usar na reproducao.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Arquivo nao encontrado ou invalido: {path}")
    return image


def load_binary_mask(path):
    mask = load_grayscale(path)
    return (mask > 0).astype(np.uint8)


def load_external_prediction(weights_root, external_group, external_task, image_stem, target_shape):
    pattern = (
        f"{external_group}/{external_task}/**/validation_raw_postprocessed/{image_stem}.nii.gz"
    )
    matches = sorted(weights_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "Predicao externa nao encontrada para "
            f"group={external_group}, task={external_task}, image={image_stem}"
        )

    mask = nib.load(str(matches[0])).get_fdata()
    mask = np.squeeze(mask).astype(np.uint8)
    resized = cv2.resize(
        mask,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized, matches[0]


def infer_local_prediction(bundle, image_gray):
    metadata = bundle.get("metadata", {})
    img_size = int(metadata.get("img_size") or metadata.get("hyperparameters", {}).get("img_size", 256))

    resized = cv2.resize(image_gray, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    image_3ch = np.stack([normalized, normalized, normalized], axis=0)
    tensor = torch.tensor(image_3ch, dtype=torch.float32).unsqueeze(0).to(next(bundle["model"].parameters()).device)

    with torch.no_grad():
        if bundle["display_name"] == "SegFormer":
            logits = bundle["model"](pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
        elif bundle["display_name"] == "DeepLab":
            logits = bundle["model"](tensor)["out"]
        else:
            logits = bundle["model"](tensor)

        probs = torch.sigmoid(logits)
        pred = (probs > bundle["threshold"]).float().cpu().numpy()[0, 0]

    pred = pred.astype(np.uint8)
    pred = cv2.resize(
        pred,
        (image_gray.shape[1], image_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return pred


def dice_score(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def draw_mask_overlay(image_gray, mask, color_rgb, alpha=0.35):
    base = np.stack([image_gray, image_gray, image_gray], axis=-1).astype(np.float32) / 255.0
    color = np.array(color_rgb, dtype=np.float32) / 255.0

    if mask.max() > 0:
        base[mask > 0] = (1 - alpha) * base[mask > 0] + alpha * color

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base_uint8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    cv2.drawContours(base_uint8, contours, -1, color_rgb, 2)
    return base_uint8


def build_local_model_rows(local_results_path):
    df = pd.read_csv(local_results_path)
    best_rows = (
        df.sort_values("dice_binary_mean", ascending=False)
        .groupby("model_name", as_index=False)
        .head(1)
        .sort_values("model_name")
        .reset_index(drop=True)
    )
    return best_rows


def create_comparison_figure(image_gray, gt_mask, external_panel, local_panels, output_path, image_name):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    panels = [
        {
            "title": "Imagem Original",
            "image": np.stack([image_gray, image_gray, image_gray], axis=-1),
        },
        {
            "title": "Ground Truth",
            "image": draw_mask_overlay(image_gray, gt_mask, (0, 255, 0)),
        },
        external_panel,
        *local_panels,
    ]

    overlay = np.stack([image_gray, image_gray, image_gray], axis=-1).astype(np.uint8)
    overlay = draw_mask_overlay(image_gray, gt_mask, (0, 255, 0), alpha=0.18)
    overlay = draw_mask_overlay(cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY), external_panel["mask"], (255, 165, 0), alpha=0.0)

    palette = {
        "DeepLab": (255, 0, 0),
        "SegFormer": (0, 255, 255),
        "UNet": (255, 0, 255),
        "UNet++": (255, 255, 0),
    }

    base_for_overlay = overlay
    for panel in local_panels:
        gray_again = cv2.cvtColor(base_for_overlay, cv2.COLOR_RGB2GRAY)
        color = palette.get(panel["short_name"], (255, 255, 255))
        base_for_overlay = draw_mask_overlay(gray_again, panel["mask"], color, alpha=0.0)

    panels.append(
        {
            "title": "Contornos Sobrepostos",
            "image": base_for_overlay,
        }
    )

    for ax, panel in zip(axes, panels):
        ax.imshow(panel["image"])
        ax.set_title(panel["title"], fontsize=11)
        ax.axis("off")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"Comparativo de Segmentadores - {image_name}", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    palette = {
        "DeepLab": (255, 0, 0),
        "SegFormer": (0, 255, 255),
        "UNet": (255, 0, 255),
        "UNet++": (255, 255, 0),
    }

    image_path = args.dataset_split / "image" / args.image_name
    gt_mask_path = args.dataset_split / "mask" / args.image_name
    image_stem = Path(args.image_name).stem

    image_gray = load_grayscale(image_path)
    gt_mask = load_binary_mask(gt_mask_path)

    external_mask, external_path = load_external_prediction(
        args.weights_root,
        args.external_group,
        args.external_task,
        image_stem,
        image_gray.shape,
    )
    external_dice = dice_score(external_mask, gt_mask)
    external_panel = {
        "title": (
            f"kidneyUS {args.external_group}\n"
            f"{args.external_task}\nDice={external_dice:.4f}"
        ),
        "image": draw_mask_overlay(image_gray, external_mask, (255, 165, 0)),
        "mask": external_mask,
        "source_path": str(external_path),
    }

    family_rows = build_local_model_rows(args.local_results)
    local_panels = []
    local_report_rows = []

    for _, row in family_rows.iterrows():
        checkpoint_path = row["checkpoint_path"]
        model_name = row["model_key"]
        bundle = load_model_bundle(model_name, device=device, checkpoint_path=checkpoint_path)
        pred_mask = infer_local_prediction(bundle, image_gray)
        pred_dice = dice_score(pred_mask, gt_mask)

        local_panels.append(
            {
                "title": f"{row['model_name']}\n{row['experiment_name']}\nDice={pred_dice:.4f}",
                "image": draw_mask_overlay(image_gray, pred_mask, palette.get(row["model_name"], (255, 255, 255))),
                "mask": pred_mask,
                "short_name": row["model_name"],
            }
        )
        local_report_rows.append(
            {
                "model_name": row["model_name"],
                "experiment_name": row["experiment_name"],
                "dataset_variant": row["dataset_variant"],
                "checkpoint_path": checkpoint_path,
                "dice_on_selected_image": pred_dice,
            }
        )

    local_panels = sorted(local_panels, key=lambda item: item["short_name"])
    output_path = args.output_dir / f"segmenter_comparison_{Path(args.image_name).stem}.png"
    create_comparison_figure(
        image_gray,
        gt_mask,
        external_panel,
        local_panels,
        output_path,
        args.image_name,
    )

    report_path = args.output_dir / f"segmenter_comparison_{Path(args.image_name).stem}.csv"
    pd.DataFrame(
        [
            {
                "model_name": "kidneyUS",
                "experiment_name": f"{args.external_group}_{args.external_task}",
                "dataset_variant": "external_reference",
                "checkpoint_path": str(external_path),
                "dice_on_selected_image": external_dice,
            },
            *local_report_rows,
        ]
    ).to_csv(report_path, index=False)

    print(f"Figura salva em: {output_path}")
    print(f"CSV salvo em: {report_path}")
    print(f"Imagem usada: {image_path}")
    print(f"Referencia externa: {external_path}")


if __name__ == "__main__":
    main()

