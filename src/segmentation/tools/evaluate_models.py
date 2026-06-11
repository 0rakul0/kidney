import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CAPSULE_DATASET = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
)

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.segmentation.core.dataset import KidneyDataset
from src.segmentation.core.metrics import dice_score
from src.segmentation.core.model_loader import load_model_bundle, list_supported_models


# ==================================
# CONFIG
# ==================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Avaliacao de Dice para modelos de segmentacao renal."
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
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_CAPSULE_DATASET))
    parser.add_argument("--model-dir", type=str, default="models")
    return parser.parse_args()


# ==================================
# EVALUATION
# ==================================

def evaluate_model(model, name, threshold, test_loader, device):

    dice = 0

    with torch.no_grad():

        for imgs, masks in tqdm(test_loader, desc=name):

            imgs = imgs.to(device)
            masks = masks.to(device)

            if name == "SegFormer":

                preds = model(pixel_values=imgs).logits

                preds = torch.nn.functional.interpolate(
                    preds,
                    size=masks.shape[1:],
                    mode="bilinear",
                    align_corners=False
                )

            elif name == "DeepLab":

                preds = model(imgs)["out"]

            else:

                preds = model(imgs)

            dice += dice_score(preds.squeeze(1), masks, threshold=threshold)

    dice /= len(test_loader)

    return dice


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
        score = evaluate_model(
            bundle["model"],
            display_name,
            bundle["threshold"],
            test_loader,
            device,
        )
        results[display_name] = {
            "Dice": score,
            "Threshold": bundle["threshold"],
            "Checkpoint": str(bundle["checkpoint_path"]),
        }
        print(
            f"{display_name} Dice: {score:.4f} | "
            f"threshold={bundle['threshold']:.2f} | "
            f"checkpoint={bundle['checkpoint_path']}"
        )

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.sort_values("Dice", ascending=False)

    print("\nRanking final:")
    print(df)


if __name__ == "__main__":
    main()


