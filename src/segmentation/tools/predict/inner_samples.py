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
from src.segmentation.experiments.train_inner_deeplab import CLASS_NAMES
from src.segmentation.experiments.train_unet import UNet


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "intrarenal_unet_multiclass_annotator1.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "intrarenal_unet_multiclass_predictions_kidneyus"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera predicoes e previews da U-Net intrarrenal multiclasse no kidneyUS supervisionado."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview-count", type=int, default=50)
    return parser.parse_args()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Nao foi possivel salvar: {path}")


def load_model(checkpoint, device):
    metadata = load_checkpoint_metadata(checkpoint)
    config = metadata.get("config", {})
    base_channels = int(config.get("base_channels", 64))
    img_size = int(config.get("img_size", 256))
    model = UNet(in_channels=3, out_channels=len(CLASS_NAMES), base_channels=base_channels).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval(), img_size, metadata


def predict(model, image, kidney, img_size, device):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    kidney_resized = cv2.resize(kidney, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    clahe_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    image_float = clahe_image.astype(np.float32) / 255.0
    kidney_float = (kidney_resized > 0).astype(np.float32)
    channels = np.stack([image_float, image_float * kidney_float, kidney_float], axis=0)
    tensor = torch.tensor(channels, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).cpu().numpy()[0].astype(np.uint8)
    prediction[kidney_resized == 0] = 0
    prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    prediction[kidney == 0] = 0
    return prediction


def overlay_prediction(image, kidney, prediction, target, label):
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pred_overlay = base.copy()
    pred_overlay[kidney > 0] = (80, 80, 80)
    pred_overlay[prediction == 1] = (255, 220, 0)
    pred_overlay[prediction == 2] = (0, 220, 255)
    pred_overlay[prediction == 3] = (0, 128, 255)
    pred_overlay = cv2.addWeighted(base, 0.62, pred_overlay, 0.38, 0)

    target_overlay = base.copy()
    target_overlay[target == 1] = (255, 220, 0)
    target_overlay[target == 2] = (0, 220, 255)
    target_overlay[target == 3] = (0, 128, 255)
    target_overlay = cv2.addWeighted(base, 0.62, target_overlay, 0.38, 0)

    tiles = [
        cv2.resize(tile, (280, 220), interpolation=cv2.INTER_AREA)
        for tile in (base, target_overlay, pred_overlay)
    ]
    panel = cv2.hconcat(tiles)
    cv2.putText(panel, label[:108], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "original | anotacao | U-Net", (8, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def dice_per_class(prediction, target):
    result = {}
    for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
        pred = prediction == class_id
        truth = target == class_id
        denom = int(pred.sum() + truth.sum())
        result[f"{class_name}_dice"] = "" if denom == 0 else f"{(2 * np.logical_and(pred, truth).sum() / denom):.6f}"
    return result


def iter_splits(split):
    return ("train", "val", "test") if split == "all" else (split,)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, img_size, metadata = load_model(args.checkpoint, device)
    rows = []
    preview_count = 0
    for split in iter_splits(args.split):
        image_dir = args.dataset_root / split / "image"
        mask_dir = args.dataset_root / split / "mask"
        kidney_dir = args.dataset_root / split / "kidney_mask"
        names = sorted(path.name for path in image_dir.glob("*.png"))
        if args.limit is not None:
            names = names[: args.limit]
        for name in names:
            image = read_image(image_dir / name)
            target = read_image(mask_dir / name)
            kidney = (read_image(kidney_dir / name) > 0).astype(np.uint8)
            prediction = predict(model, image, kidney, img_size, device)

            label_path = args.output_root / split / "labels" / name
            save_image(label_path, prediction)
            class_paths = {}
            for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
                class_path = args.output_root / split / "masks" / class_name / name
                save_image(class_path, ((prediction == class_id).astype(np.uint8) * 255))
                class_paths[class_name] = class_path

            metrics = dice_per_class(prediction, target)
            if preview_count < args.preview_count:
                preview_count += 1
                panel = overlay_prediction(
                    image,
                    kidney,
                    prediction,
                    target,
                    f"{split} | {name} | c={metrics['cortex_dice']} m={metrics['medulla_dice']} cec={metrics['central_echo_complex_dice']}",
                )
                save_image(args.output_root / "previews" / f"{preview_count:03d}_{split}_{name}", panel)

            rows.append(
                {
                    "split": split,
                    "filename": name,
                    "source_image_path": str(image_dir / name),
                    "target_mask_path": str(mask_dir / name),
                    "predicted_label_path": str(label_path),
                    "predicted_cortex_mask_path": str(class_paths["cortex"]),
                    "predicted_medulla_mask_path": str(class_paths["medulla"]),
                    "predicted_central_echo_complex_mask_path": str(class_paths["central_echo_complex"]),
                    **metrics,
                }
            )

    manifest_path = args.output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "output_root": str(args.output_root),
        "split": args.split,
        "processed": len(rows),
        "device": device,
        "img_size": img_size,
        "classes": {index: name for index, name in enumerate(CLASS_NAMES)},
        "model_summary": metadata,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
