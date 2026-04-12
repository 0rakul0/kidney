import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff

from utils.dataset import KidneyDataset
from utils.model_loader import load_model_bundle, list_supported_models


# ==================================
# CONFIG
# ==================================

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================
# ARGS
# ==================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Benchmark de modelos de segmentacao renal."
    )
    parser.add_argument(
        "--model",
        choices=["all", *list_supported_models()],
        default="all",
        help="Modelo a ser avaliado. Use 'all' para comparar os checkpoints padrao.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint externo .pth para um unico modelo.",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101"],
        help="Override do backbone para checkpoint DeepLab sem metadata.",
    )
    parser.add_argument(
        "--segformer-backbone",
        type=str,
        help="Override do backbone Hugging Face para checkpoint SegFormer sem metadata.",
    )
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-path", type=str, default="dataset")
    parser.add_argument("--model-dir", type=str, default="models")
    return parser.parse_args()


# ==================================
# METRICS
# ==================================

def dice(pred, target):

    intersection = (pred * target).sum()

    return (2 * intersection) / (pred.sum() + target.sum() + 1e-8)


def iou(pred, target):

    intersection = (pred * target).sum()

    union = pred.sum() + target.sum() - intersection

    return intersection / (union + 1e-8)


def precision(pred, target):

    tp = (pred * target).sum()

    fp = (pred * (1 - target)).sum()

    return tp / (tp + fp + 1e-8)


def recall(pred, target):

    tp = (pred * target).sum()

    fn = ((1 - pred) * target).sum()

    return tp / (tp + fn + 1e-8)


def f1_score(pred, target):

    p = precision(pred, target)
    r = recall(pred, target)

    return 2 * (p * r) / (p + r + 1e-8)


def hausdorff(pred, target):

    pred_points = np.argwhere(pred == 1)
    target_points = np.argwhere(target == 1)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan

    d1 = directed_hausdorff(pred_points, target_points)[0]
    d2 = directed_hausdorff(target_points, pred_points)[0]

    return max(d1, d2)


# ==================================
# INFERENCE
# ==================================

def predict(model, imgs, name, threshold, img_size):

    if name == "SegFormer":

        preds = model(pixel_values=imgs).logits

        preds = torch.nn.functional.interpolate(
            preds,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        )

    elif name == "DeepLab":

        preds = model(imgs)["out"]

    else:

        preds = model(imgs)

    preds = torch.sigmoid(preds)

    preds = (preds > threshold).float()

    return preds


# ==================================
# EVALUATION
# ==================================

def evaluate_model(model, name, threshold, test_loader, test_dataset, device, img_size):

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    hausdorff_list = []

    start = time.time()

    with torch.no_grad():

        for imgs, masks in tqdm(test_loader, desc=name):

            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = predict(model, imgs, name, threshold, img_size)

            preds = preds.squeeze(1).cpu().numpy()
            masks = masks.cpu().numpy()

            for p, m in zip(preds, masks):

                dice_list.append(dice(p, m))
                iou_list.append(iou(p, m))
                precision_list.append(precision(p, m))
                recall_list.append(recall(p, m))
                f1_list.append(f1_score(p, m))
                hausdorff_list.append(hausdorff(p, m))

    total_time = time.time() - start

    fps = len(test_dataset) / total_time

    return {
        "Dice": np.nanmean(dice_list),
        "IoU": np.nanmean(iou_list),
        "Precision": np.nanmean(precision_list),
        "Recall": np.nanmean(recall_list),
        "F1": np.nanmean(f1_list),
        "Hausdorff": np.nanmean(hausdorff_list),
        "FPS": fps
    }


# ==================================
def main():

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint and args.model == "all":
        raise SystemExit("Use --checkpoint junto com um unico --model.")

    test_dataset = KidneyDataset(
        os.path.join(args.dataset_path, "test"),
        args.img_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )

    print("Test samples:", len(test_dataset))

    selected_models = (
        list_supported_models()
        if args.model == "all"
        else [args.model]
    )

    results = {}

    for model_name in selected_models:
        bundle = load_model_bundle(
            model_name,
            device=device,
            checkpoint_path=args.checkpoint,
            model_dir=args.model_dir,
            deeplab_backbone=args.backbone,
            segformer_backbone=args.segformer_backbone,
        )
        display_name = bundle["display_name"]
        metrics = evaluate_model(
            bundle["model"],
            display_name,
            bundle["threshold"],
            test_loader,
            test_dataset,
            device,
            args.img_size,
        )
        results[display_name] = {
            **metrics,
            "Threshold": bundle["threshold"],
            "Checkpoint": str(bundle["checkpoint_path"]),
        }
        print(display_name, results[display_name])

    df = pd.DataFrame(results).T
    df = df.sort_values(["Dice", "IoU"], ascending=False)

    print("\nFinal ranking\n")
    print(df)

    log_path = os.path.join(OUTPUT_DIR, "benchmark_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Benchmark Results\n\n")
        f.write(df.to_string())
    print("Log saved:", log_path)

    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    df.to_csv(csv_path)
    print("CSV saved:", csv_path)

    if len(df) > 1:
        plt.figure(figsize=(8, 5))
        plt.bar(df.index, df["Dice"])
        plt.ylabel("Dice Score")
        plt.title("Model Comparison")
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, "dice_comparison.png")
        plt.savefig(plot_path)
        print("Plot saved:", plot_path)


if __name__ == "__main__":
    main()
