import time

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.core.dataset import KidneyDataset


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


def predict_binary_mask(model, display_name, imgs, threshold, img_size):
    if display_name == "SegFormer":
        preds = model(pixel_values=imgs).logits
        preds = torch.nn.functional.interpolate(
            preds,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    elif display_name == "DeepLab":
        preds = model(imgs)["out"]
    else:
        preds = model(imgs)

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    return preds


def evaluate_segmentation_model(
    model,
    display_name,
    threshold,
    dataset_path,
    img_size=256,
    batch_size=8,
    split="test",
    device=None,
    num_workers=0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KidneyDataset(
        f"{dataset_path}/{split}",
        img_size=img_size,
        augment=False,
        clahe=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    hausdorff_list = []

    start = time.time()

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"{display_name} {split}", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = predict_binary_mask(model, display_name, imgs, threshold, img_size)

            preds = preds.squeeze(1).cpu().numpy()
            masks = masks.cpu().numpy()

            for pred_mask, target_mask in zip(preds, masks):
                dice_list.append(dice(pred_mask, target_mask))
                iou_list.append(iou(pred_mask, target_mask))
                precision_list.append(precision(pred_mask, target_mask))
                recall_list.append(recall(pred_mask, target_mask))
                f1_list.append(f1_score(pred_mask, target_mask))
                hausdorff_list.append(hausdorff(pred_mask, target_mask))

    elapsed_seconds = time.time() - start
    fps = len(dataset) / elapsed_seconds if elapsed_seconds > 0 else np.nan

    return {
        "eval_split": split,
        "dice": float(np.nanmean(dice_list)),
        "iou": float(np.nanmean(iou_list)),
        "precision": float(np.nanmean(precision_list)),
        "recall": float(np.nanmean(recall_list)),
        "f1": float(np.nanmean(f1_list)),
        "hausdorff": float(np.nanmean(hausdorff_list)),
        "fps": float(fps),
        "samples": int(len(dataset)),
        "img_size": int(img_size),
        "batch_size": int(batch_size),
    }

