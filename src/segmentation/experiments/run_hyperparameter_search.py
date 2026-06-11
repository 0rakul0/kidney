import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPSULE_DATASET = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.experiments.train_deeplab import build_model as build_deeplab
from src.segmentation.experiments.train_segformer import build_model as build_segformer
from src.segmentation.experiments.train_unet import build_model as build_unet
from src.segmentation.experiments.train_unetplusplus import build_model as build_unetplusplus
from src.segmentation.core.segmentation_training import SegmentationTrainingConfig, train_segmentation_model


MODEL_REGISTRY = {
    "unet": {
        "display_name": "UNet",
        "model_kind": "plain",
        "checkpoint_prefix": "unet",
        "builder": build_unet,
        "experiments": [
            {
                "name": "baseline",
                "model_kwargs": {"base_channels": 64},
                "overrides": {}
            },
            {
                "name": "capacity_augmented",
                "model_kwargs": {"base_channels": 96},
                "overrides": {
                    "augment": True,
                    "weight_decay": 1e-4,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "threshold_search": True,
                    "auto_pos_weight": True,
                }
            },
            {
                "name": "high_resolution",
                "model_kwargs": {"base_channels": 96},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "augment": True,
                    "clahe": True,
                    "weight_decay": 1e-4,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "threshold_search": True,
                }
            },
        ]
    },
    "unetplusplus": {
        "display_name": "UNet++",
        "model_kind": "plain",
        "checkpoint_prefix": "unetplusplus",
        "builder": build_unetplusplus,
        "experiments": [
            {
                "name": "baseline",
                "model_kwargs": {"base_channels": 64},
                "overrides": {}
            },
            {
                "name": "capacity_augmented",
                "model_kwargs": {"base_channels": 96},
                "overrides": {
                    "augment": True,
                    "weight_decay": 1e-4,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "auto_pos_weight": True,
                }
            },
        ]
    },
    "deeplab": {
        "display_name": "DeepLab",
        "model_kind": "deeplab",
        "checkpoint_prefix": "deeplab",
        "builder": build_deeplab,
        "experiments": [
            {
                "name": "resnet50_baseline",
                "model_kwargs": {"backbone": "resnet50", "pretrained": True},
                "overrides": {}
            },
            {
                "name": "resnet101_capacity",
                "model_kwargs": {"backbone": "resnet101", "pretrained": True},
                "overrides": {
                    "learning_rate": 8e-5,
                    "weight_decay": 1e-4,
                    "optimizer_name": "adamw",
                    "scheduler_name": "plateau",
                    "auto_pos_weight": True,
                }
            },
        ]
    },
    "segformer": {
        "display_name": "SegFormer",
        "model_kind": "segformer",
        "checkpoint_prefix": "segformer",
        "builder": build_segformer,
        "experiments": [
            {
                "name": "b0_baseline",
                "model_kwargs": {"backbone_name": "nvidia/segformer-b0-finetuned-ade-512-512"},
                "overrides": {}
            },
            {
                "name": "b2_capacity",
                "model_kwargs": {"backbone_name": "nvidia/segformer-b2-finetuned-ade-512-512"},
                "overrides": {
                    "learning_rate": 8e-5,
                    "batch_size": 4,
                    "weight_decay": 1e-4,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "augment": True,
                    "auto_pos_weight": True,
                }
            },
        ]
    },
}


def parse_args():

    parser = argparse.ArgumentParser(description="Busca simples de hiperparametros para segmentacao renal.")
    parser.add_argument("--model", choices=["all", *MODEL_REGISTRY.keys()], default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_CAPSULE_DATASET))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-name", type=str, default="hyperparameter_search_summary.csv")

    return parser.parse_args()


def build_base_config(model_key, model_info, args):

    return SegmentationTrainingConfig(
        model_name=model_info["display_name"],
        experiment_name=f"{model_key}_baseline",
        checkpoint_name=f"{model_info['checkpoint_prefix']}_best.pth",
        dataset_path=args.dataset_path,
        model_kind=model_info["model_kind"],
        epochs=args.epochs,
        seed=args.seed,
        threshold_search=True,
        early_stopping_patience=8,
        model_kwargs={}
    )


def main():

    args = parse_args()

    selected_models = MODEL_REGISTRY.keys() if args.model == "all" else [args.model]
    summaries = []

    for model_key in selected_models:
        model_info = MODEL_REGISTRY[model_key]
        base_config = build_base_config(model_key, model_info, args)

        for experiment in model_info["experiments"]:
            experiment_name = f"{model_key}_{experiment['name']}"
            checkpoint_name = f"{experiment_name}.pth"

            config_kwargs = {
                **base_config.__dict__,
                **experiment["overrides"],
                "experiment_name": experiment_name,
                "checkpoint_name": checkpoint_name,
                "model_kwargs": experiment["model_kwargs"],
            }

            config = SegmentationTrainingConfig(**config_kwargs)

            print(f"\n=== {experiment_name} ===")
            summary = train_segmentation_model(model_info["builder"], config)
            summaries.append(
                {
                    "experiment_name": experiment_name,
                    "model_name": summary["model_name"],
                    "best_val_dice": summary["best_val_dice"],
                    "test_dice": summary["test_dice"],
                    "best_threshold": summary["best_threshold"],
                    "checkpoint_path": summary["checkpoint_path"],
                }
            )

    if not summaries:
        return

    summary_path = base_config.summary_path.parent / args.summary_name

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nResumo salvo:", summary_path)


if __name__ == "__main__":
    main()

