import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.model_loader import load_model_bundle


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
    / "test"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "kidneyus_capsule_unet.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "segmentation_experiments" / "unet_preprocess_comparison"


VARIANTS = {
    "unet_raw": "U-Net",
    "unet_clahe": "CLAHE + U-Net",
    "unet_superres_clahe": "Super-resolucao + CLAHE + U-Net",
    "unet_swinir_clahe": "SwinIR + CLAHE + U-Net",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara U-Net sem CLAHE, com CLAHE e com super-resolucao + CLAHE."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--superres-scale", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-swinir", action="store_true")
    parser.add_argument(
        "--swinir-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "super_resolution" / "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
    )
    return parser.parse_args()


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def apply_clahe(image):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def apply_superres(image, scale):
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_LANCZOS4)


def load_swinir_model(model_path, device):
    swinir_root = PROJECT_ROOT / "external_tools" / "SwinIR"
    if str(swinir_root) not in sys.path:
        sys.path.insert(0, str(swinir_root))
    from models.network_swinir import SwinIR

    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("params", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def swinir_upscale(image, model, device, scale=2, window_size=8, tile=256, tile_overlap=32):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float().unsqueeze(0).to(device)
    _, _, h_old, w_old = tensor.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    tensor = torch.cat([tensor, torch.flip(tensor, [2])], 2)[:, :, : h_old + h_pad, :]
    tensor = torch.cat([tensor, torch.flip(tensor, [3])], 3)[:, :, :, : w_old + w_pad]

    _, _, h, w = tensor.size()
    output = torch.zeros((1, 3, h * scale, w * scale), device=device)
    weight = torch.zeros_like(output)
    stride = tile - tile_overlap

    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y0 = min(y, max(h - tile, 0))
                x0 = min(x, max(w - tile, 0))
                patch = tensor[:, :, y0 : y0 + tile, x0 : x0 + tile]
                patch_out = model(patch)
                oy0, ox0 = y0 * scale, x0 * scale
                output[:, :, oy0 : oy0 + patch_out.shape[2], ox0 : ox0 + patch_out.shape[3]] += patch_out
                weight[:, :, oy0 : oy0 + patch_out.shape[2], ox0 : ox0 + patch_out.shape[3]] += 1
    output = output / torch.clamp(weight, min=1)
    output = output[:, :, : h_old * scale, : w_old * scale]
    array = output.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    rgb_out = np.transpose(array, (1, 2, 0))
    gray = cv2.cvtColor((rgb_out * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray


def prepare_variant(image, variant, scale, swinir_model, device):
    if variant == "unet_raw":
        return image
    if variant == "unet_clahe":
        return apply_clahe(image)
    if variant == "unet_superres_clahe":
        return apply_clahe(apply_superres(image, scale))
    if variant == "unet_swinir_clahe":
        return apply_clahe(swinir_upscale(image, swinir_model, device, scale=scale))
    raise ValueError(f"Variante desconhecida: {variant}")


def prepare_tensor(image, img_size, device):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    stacked = np.stack([normalized, normalized, normalized], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)


def predict_mask(bundle, image, original_shape, args, device):
    tensor = prepare_tensor(image, args.img_size, device)
    with torch.no_grad():
        logits = bundle["model"](tensor)
        probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    probability = cv2.resize(
        probability,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return (probability >= float(bundle["threshold"])).astype(np.uint8)


def dice_iou(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = np.logical_or(pred, target).sum()
    dice = (2.0 * intersection) / max(pred_sum + target_sum, 1)
    iou = intersection / max(union, 1)
    return float(dice), float(iou)


def overlay_contour(image, mask, color):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, color, 2)
    return rgb


def label_tile(tile, title, subtitle=""):
    canvas = tile.copy()
    pad = 34
    header = np.zeros((pad, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(header, subtitle, (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 220, 235), 1, cv2.LINE_AA)
    return np.vstack([header, canvas])


def make_panel(image, target, predictions, metrics, output_path, active_variants):
    width = 300
    scale = width / image.shape[1]
    height = int(image.shape[0] * scale)

    tiles = []
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    original = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
    tiles.append(label_tile(original, "Imagem original"))

    gt = overlay_contour(image, target, (0, 220, 0))
    gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_AREA)
    tiles.append(label_tile(gt, "Mascara manual"))

    colors = {
        "unet_raw": (255, 255, 255),
        "unet_clahe": (0, 220, 255),
        "unet_superres_clahe": (255, 150, 0),
        "unet_swinir_clahe": (210, 80, 255),
    }
    for key, label in active_variants.items():
        tile = overlay_contour(image, predictions[key], colors[key])
        tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)
        dice, iou = metrics[key]
        tiles.append(label_tile(tile, label, f"Dice {dice:.3f} | IoU {iou:.3f}"))

    panel = np.hstack(tiles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)


def summarize(rows, active_variants):
    summary = {}
    for key, label in active_variants.items():
        dice_values = [float(row[f"{key}_dice"]) for row in rows]
        iou_values = [float(row[f"{key}_iou"]) for row in rows]
        summary[key] = {
            "label": label,
            "dice_mean": float(np.mean(dice_values)),
            "dice_std": float(np.std(dice_values)),
            "iou_mean": float(np.mean(iou_values)),
            "iou_std": float(np.std(iou_values)),
        }
    return summary


def main():
    args = parse_args()
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_dir = run_dir / "previews"
    image_dir = args.dataset_root / "image"
    mask_dir = args.dataset_root / "mask"
    image_paths = sorted(image_dir.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model_bundle("unet", device=device, checkpoint_path=args.checkpoint)
    active_variants = {
        key: label
        for key, label in VARIANTS.items()
        if not (args.skip_swinir and key == "unet_swinir_clahe")
    }
    swinir_model = None
    if "unet_swinir_clahe" in active_variants:
        if not args.swinir_model.exists():
            raise FileNotFoundError(f"Peso SwinIR nao encontrado: {args.swinir_model}")
        swinir_model = load_swinir_model(args.swinir_model, device)

    rows = []
    preview_candidates = []
    for image_path in image_paths:
        image = read_gray(image_path)
        target = (read_gray(mask_dir / image_path.name) > 0).astype(np.uint8)

        predictions = {}
        metrics = {}
        row = {"image_name": image_path.name}
        for key in active_variants:
            model_image = prepare_variant(image, key, args.superres_scale, swinir_model, device)
            pred = predict_mask(bundle, model_image, image.shape, args, device)
            predictions[key] = pred
            metrics[key] = dice_iou(pred, target)
            row[f"{key}_dice"] = f"{metrics[key][0]:.6f}"
            row[f"{key}_iou"] = f"{metrics[key][1]:.6f}"

        row["clahe_minus_raw_dice"] = f"{metrics['unet_clahe'][0] - metrics['unet_raw'][0]:.6f}"
        row["superres_minus_clahe_dice"] = (
            f"{metrics['unet_superres_clahe'][0] - metrics['unet_clahe'][0]:.6f}"
        )
        rows.append(row)

        spread = max(value[0] for value in metrics.values()) - min(value[0] for value in metrics.values())
        preview_candidates.append((spread, image_path, image, target, predictions, metrics))

    preview_candidates.sort(key=lambda item: item[0], reverse=True)
    for index, (_, image_path, image, target, predictions, metrics) in enumerate(
        preview_candidates[: args.preview_count],
        start=1,
    ):
        make_panel(
            image,
            target,
            predictions,
            metrics,
            preview_dir / f"{index:03d}_{image_path.stem}.png",
            active_variants,
        )

    results_csv = run_dir / "metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "device": device,
        "threshold": float(bundle["threshold"]),
        "img_size": args.img_size,
        "superres_method": "Lanczos",
        "superres_scale": args.superres_scale,
        "samples": len(rows),
        "variants": summarize(rows, active_variants),
        "metrics_csv": str(results_csv),
        "preview_dir": str(preview_dir),
        "note": (
            "Super-resolucao classica Lanczos 2x avaliada apenas como pre-processamento "
            "auxiliar para segmentacao; imagens originais permanecem preservadas."
        ),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
