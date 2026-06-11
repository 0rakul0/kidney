import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.checkpoint_metadata import load_checkpoint_metadata
from src.segmentation.experiments.train_inner_deeplab import CLASS_NAMES, build_model as build_deeplab
from src.segmentation.experiments.train_unet import UNet


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1"
)
DEFAULT_UNET = PROJECT_ROOT / "models" / "intrarenal_unet_multiclass_annotator1.pth"
DEFAULT_DEEPLAB = PROJECT_ROOT / "models" / "intrarenal_deeplab_resnet50_multiclass_annotator1.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "intrarenal_model3" / "unet_deeplab_divergences_kidneyus"


def parse_args():
    parser = argparse.ArgumentParser(description="Compara divergencias intrarrenais entre U-Net e DeepLabV3.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--unet-checkpoint", type=Path, default=DEFAULT_UNET)
    parser.add_argument("--deeplab-checkpoint", type=Path, default=DEFAULT_DEEPLAB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Nao foi possivel salvar: {path}")


def load_models(args, device):
    unet_meta = load_checkpoint_metadata(args.unet_checkpoint)
    unet_config = unet_meta.get("config", {})
    img_size = int(unet_config.get("img_size", 256))
    base_channels = int(unet_config.get("base_channels", 64))
    unet = UNet(in_channels=3, out_channels=len(CLASS_NAMES), base_channels=base_channels).to(device)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))

    deeplab_meta = load_checkpoint_metadata(args.deeplab_checkpoint)
    deeplab_config = deeplab_meta.get("config", {})
    backbone = deeplab_config.get("backbone", "resnet50")
    deeplab = build_deeplab(backbone=backbone, pretrained=False, num_classes=len(CLASS_NAMES)).to(device)
    deeplab.load_state_dict(torch.load(args.deeplab_checkpoint, map_location=device))

    return unet.eval(), deeplab.eval(), img_size, unet_meta, deeplab_meta


def prepare_tensor(image, kidney, img_size, device):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    kidney_resized = cv2.resize(kidney, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    image_float = clahe.astype(np.float32) / 255.0
    kidney_float = (kidney_resized > 0).astype(np.float32)
    channels = np.stack([image_float, image_float * kidney_float, kidney_float], axis=0)
    return torch.tensor(channels, dtype=torch.float32).unsqueeze(0).to(device), kidney_resized


def predict(model, model_kind, image, kidney, img_size, device):
    tensor, kidney_resized = prepare_tensor(image, kidney, img_size, device)
    with torch.no_grad():
        logits = model(tensor)["out"] if model_kind == "deeplab" else model(tensor)
        pred = logits.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)
    pred[kidney_resized == 0] = 0
    pred = cv2.resize(pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred[kidney == 0] = 0
    return pred


def dice_class(pred, target, class_id):
    pred_mask = pred == class_id
    target_mask = target == class_id
    denom = int(pred_mask.sum() + target_mask.sum())
    if denom == 0:
        return None
    return float(2 * np.logical_and(pred_mask, target_mask).sum() / denom)


def label_overlay(image, label, kidney=None):
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    if kidney is not None:
        overlay[kidney > 0] = (80, 80, 80)
    overlay[label == 1] = (255, 220, 0)
    overlay[label == 2] = (0, 220, 255)
    overlay[label == 3] = (0, 128, 255)
    return cv2.addWeighted(base, 0.62, overlay, 0.38, 0)


def make_panel(image, target, unet_pred, deeplab_pred, kidney, row):
    disagreement = ((unet_pred != deeplab_pred) & (kidney > 0)).astype(np.uint8) * 255
    disagreement_rgb = cv2.cvtColor(disagreement, cv2.COLOR_GRAY2BGR)
    tiles = [
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        label_overlay(image, target),
        label_overlay(image, unet_pred, kidney),
        label_overlay(image, deeplab_pred, kidney),
        disagreement_rgb,
    ]
    tiles = [cv2.resize(tile, (240, 190), interpolation=cv2.INTER_AREA) for tile in tiles]
    panel = cv2.hconcat(tiles)
    label = (
        f"{row['filename']} | diff={row['model_disagreement_ratio']} | "
        f"dU={row['unet_mean_fg_dice']} dD={row['deeplab_mean_fg_dice']}"
    )
    cv2.putText(panel, label[:150], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "original | anotacao | U-Net | DeepLab | divergencia",
        (8, 182),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def fmt(value):
    return "" if value is None else f"{value:.6f}"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet, deeplab, img_size, unet_meta, deeplab_meta = load_models(args, device)

    split_root = args.dataset_root / args.split
    names = sorted(path.name for path in (split_root / "image").glob("*.png"))
    rows = []
    predictions = {}
    for name in names:
        image = read_gray(split_root / "image" / name)
        target = read_gray(split_root / "mask" / name)
        kidney = (read_gray(split_root / "kidney_mask" / name) > 0).astype(np.uint8)
        unet_pred = predict(unet, "unet", image, kidney, img_size, device)
        deeplab_pred = predict(deeplab, "deeplab", image, kidney, img_size, device)

        row = {"filename": name}
        unet_scores = []
        deeplab_scores = []
        for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
            u = dice_class(unet_pred, target, class_id)
            d = dice_class(deeplab_pred, target, class_id)
            row[f"unet_{class_name}_dice"] = fmt(u)
            row[f"deeplab_{class_name}_dice"] = fmt(d)
            row[f"delta_{class_name}_dice_unet_minus_deeplab"] = fmt(None if u is None or d is None else u - d)
            if u is not None:
                unet_scores.append(u)
            if d is not None:
                deeplab_scores.append(d)
        row["unet_mean_fg_dice"] = fmt(float(np.mean(unet_scores)) if unet_scores else None)
        row["deeplab_mean_fg_dice"] = fmt(float(np.mean(deeplab_scores)) if deeplab_scores else None)
        row["delta_mean_fg_dice_unet_minus_deeplab"] = fmt(
            float(np.mean(unet_scores) - np.mean(deeplab_scores)) if unet_scores and deeplab_scores else None
        )
        disagreement = float(((unet_pred != deeplab_pred) & (kidney > 0)).sum() / max(int(kidney.sum()), 1))
        row["model_disagreement_ratio"] = f"{disagreement:.6f}"
        rows.append(row)
        predictions[name] = (image, target, unet_pred, deeplab_pred, kidney)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / f"{args.split}_divergence_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda item: float(item["model_disagreement_ratio"]), reverse=True)
    for index, row in enumerate(ranked[: args.top_k], start=1):
        panel = make_panel(*predictions[row["filename"]], row)
        save_image(args.output_root / "top_divergence_previews" / f"{index:03d}_{row['filename']}", panel)

    summary = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "processed": len(rows),
        "device": device,
        "manifest": str(manifest_path),
        "top_preview_dir": str(args.output_root / "top_divergence_previews"),
        "unet_checkpoint": str(args.unet_checkpoint),
        "deeplab_checkpoint": str(args.deeplab_checkpoint),
        "unet_summary": unet_meta,
        "deeplab_summary": deeplab_meta,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in ranked[: min(10, args.top_k)]:
        print(
            row["filename"],
            "diff=", row["model_disagreement_ratio"],
            "delta_mean=", row["delta_mean_fg_dice_unet_minus_deeplab"],
        )


if __name__ == "__main__":
    main()
