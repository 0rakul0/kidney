import argparse
import sys
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.segmentation_training import (
    add_training_args,
    build_training_config,
    train_segmentation_model,
)


# =========================
# MODEL BUILDER
# =========================

def build_model(backbone_name="nvidia/segformer-b0-finetuned-ade-512-512"):

    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone_name,
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    return model


def parse_args():

    parser = argparse.ArgumentParser(description="Treino configuravel do SegFormer para segmentacao renal.")
    add_training_args(
        parser,
        default_model_name="SegFormer",
        default_checkpoint_name="segformer_best.pth",
        default_experiment_name="segformer_baseline"
    )
    parser.add_argument(
        "--backbone-name",
        choices=[
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-ade-512-512",
        ],
        default="nvidia/segformer-b0-finetuned-ade-512-512"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = build_training_config(
        args,
        model_kind="segformer",
        model_kwargs={"backbone_name": args.backbone_name}
    )

    train_segmentation_model(build_model, config)


if __name__ == "__main__":
    main()

