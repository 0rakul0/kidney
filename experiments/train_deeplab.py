import argparse
import sys
from pathlib import Path

import torch
import torchvision
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.segmentation_training import (
    add_training_args,
    build_training_config,
    train_segmentation_model,
)


# =========================
# MODEL BUILDER
# =========================

def build_model(backbone="resnet50", pretrained=True):

    builders = {
        "resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
        "resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
    }

    if backbone not in builders:
        raise ValueError(f"Backbone nao suportado: {backbone}")

    weights = "DEFAULT" if pretrained else None

    model = builders[backbone](weights=weights)

    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    return model


def parse_args():

    parser = argparse.ArgumentParser(description="Treino configuravel do DeepLabV3 para segmentacao renal.")
    add_training_args(
        parser,
        default_model_name="DeepLab",
        default_checkpoint_name="deeplab_best.pth",
        default_experiment_name="deeplab_baseline"
    )
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--no-pretrained", action="store_true")

    return parser.parse_args()


def main():

    args = parse_args()

    config = build_training_config(
        args,
        model_kind="deeplab",
        model_kwargs={
            "backbone": args.backbone,
            "pretrained": not args.no_pretrained,
        }
    )

    train_segmentation_model(build_model, config)


if __name__ == "__main__":
    main()
