import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.experiments.train_deeplab import build_model as build_deeplab
from src.segmentation.experiments.train_segformer import build_model as build_segformer
from src.segmentation.experiments.train_unet import build_model as build_unet
from src.segmentation.experiments.train_unetplusplus import build_model as build_unetplusplus
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.core.segmentation_evaluation import evaluate_segmentation_model
from src.segmentation.core.segmentation_training import SegmentationTrainingConfig, train_segmentation_model


QUALITY_THRESHOLD_CANDIDATES = tuple(round(x / 100, 2) for x in range(25, 81, 5))


QUALITY_MODEL_REGISTRY = {
    "unet": {
        "display_name": "UNet",
        "model_kind": "plain",
        "builder": build_unet,
        "experiments": [
            {
                "name": "quality_baseline",
                "model_kwargs": {"base_channels": 64},
                "overrides": {
                    "loss_name": "bce_dice",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                },
            },
            {
                "name": "quality_tversky_320",
                "model_kwargs": {"base_channels": 96},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "tversky",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                },
            },
            {
                "name": "quality_dice_focal_320",
                "model_kwargs": {"base_channels": 96},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "dice_focal",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                    "focal_alpha": 0.5,
                },
            },
        ],
    },
    "unetplusplus": {
        "display_name": "UNet++",
        "model_kind": "plain",
        "builder": build_unetplusplus,
        "experiments": [
            {
                "name": "quality_baseline",
                "model_kwargs": {"base_channels": 64},
                "overrides": {
                    "loss_name": "bce_dice",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                },
            },
            {
                "name": "quality_tversky_320",
                "model_kwargs": {"base_channels": 80},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "tversky",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                },
            },
            {
                "name": "quality_focal_tversky",
                "model_kwargs": {"base_channels": 80},
                "overrides": {
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "focal_tversky",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                    "tversky_alpha": 0.3,
                    "tversky_beta": 0.7,
                    "focal_tversky_gamma": 1.5,
                },
            },
        ],
    },
    "deeplab": {
        "display_name": "DeepLab",
        "model_kind": "deeplab",
        "builder": build_deeplab,
        "experiments": [
            {
                "name": "quality_resnet50",
                "model_kwargs": {"backbone": "resnet50", "pretrained": True},
                "overrides": {
                    "loss_name": "bce_dice",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "plateau",
                    "weight_decay": 1e-4,
                },
            },
            {
                "name": "quality_resnet101_tversky_320",
                "model_kwargs": {"backbone": "resnet101", "pretrained": True},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "tversky",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "plateau",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                },
            },
            {
                "name": "quality_resnet101_dice_focal_320",
                "model_kwargs": {"backbone": "resnet101", "pretrained": True},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "dice_focal",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "plateau",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                    "focal_alpha": 0.5,
                },
            },
        ],
    },
    "segformer": {
        "display_name": "SegFormer",
        "model_kind": "segformer",
        "builder": build_segformer,
        "experiments": [
            {
                "name": "quality_b0",
                "model_kwargs": {"backbone_name": "nvidia/segformer-b0-finetuned-ade-512-512"},
                "overrides": {
                    "loss_name": "bce_dice",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "weight_decay": 1e-4,
                },
            },
            {
                "name": "quality_b2_tversky_320",
                "model_kwargs": {"backbone_name": "nvidia/segformer-b2-finetuned-ade-512-512"},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "tversky",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                },
            },
            {
                "name": "quality_b2_focal_tversky_320",
                "model_kwargs": {"backbone_name": "nvidia/segformer-b2-finetuned-ade-512-512"},
                "overrides": {
                    "img_size": 320,
                    "batch_size": 4,
                    "learning_rate": 8e-5,
                    "loss_name": "focal_tversky",
                    "augment": True,
                    "clahe": True,
                    "optimizer_name": "adamw",
                    "scheduler_name": "cosine",
                    "weight_decay": 1e-4,
                    "auto_pos_weight": True,
                    "tversky_alpha": 0.3,
                    "tversky_beta": 0.7,
                    "focal_tversky_gamma": 1.5,
                },
            },
        ],
    },
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="Busca quality-first para maximizacao de Dice e IoU."
    )
    parser.add_argument("--model", choices=["all", *QUALITY_MODEL_REGISTRY.keys()], default="all")
    parser.add_argument("--dataset-path", type=str, default="dataset_aumentado/expansao_pseudorrotulada")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-prefix", type=str, default="quality_search")
    parser.add_argument("--early-stopping", type=int, default=16)
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista os experimentos planejados sem treinar.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Limita o numero de execucoes para smoke tests.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)

    return parser.parse_args()


def selected_models(model_arg):

    if model_arg == "all":
        return list(QUALITY_MODEL_REGISTRY.keys())

    return [model_arg]


def build_runs(args):

    runs = []

    for model_key in selected_models(args.model):
        model_info = QUALITY_MODEL_REGISTRY[model_key]

        for experiment in model_info["experiments"]:
            experiment_name = f"{model_key}_{experiment['name']}"
            checkpoint_name = f"{experiment_name}.pth"

            config = SegmentationTrainingConfig(
                model_name=model_info["display_name"],
                experiment_name=experiment_name,
                checkpoint_name=checkpoint_name,
                dataset_path=args.dataset_path,
                model_kind=model_info["model_kind"],
                epochs=args.epochs,
                seed=args.seed,
                early_stopping_patience=args.early_stopping,
                threshold_search=True,
                threshold_candidates=QUALITY_THRESHOLD_CANDIDATES,
                threshold_metric="dice_iou",
                threshold_metric_weight=0.5,
                selection_metric="dice_iou",
                selection_metric_weight=0.5,
                model_kwargs=experiment["model_kwargs"],
                **experiment["overrides"],
            )

            runs.append(
                {
                    "model_key": model_key,
                    "model_name": model_info["display_name"],
                    "builder": model_info["builder"],
                    "config": config,
                }
            )

    if args.limit_runs is not None:
        runs = runs[: args.limit_runs]

    return runs


def load_existing_summary(summary_path):

    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def train_or_reuse(run, reuse_existing):

    config = run["config"]

    if reuse_existing and config.summary_path.exists() and config.checkpoint_path.exists():
        print(f"Reutilizando: {config.experiment_name}")
        return load_existing_summary(config.summary_path), True

    print(f"\n=== Treinando {config.experiment_name} ===")
    summary = train_segmentation_model(run["builder"], config)

    return summary, False


def evaluate_run(run, summary, eval_batch_size):

    config = run["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = load_model_bundle(
        run["model_key"],
        device=device,
        checkpoint_path=summary["checkpoint_path"],
    )

    metrics = evaluate_segmentation_model(
        bundle["model"],
        bundle["display_name"],
        bundle["threshold"],
        dataset_path=config.dataset_path,
        img_size=summary.get("hyperparameters", {}).get("img_size", config.img_size),
        batch_size=eval_batch_size,
        split="test",
        device=device,
        num_workers=summary.get("hyperparameters", {}).get("num_workers", config.num_workers),
    )

    return {
        "experiment_name": summary["experiment_name"],
        "model_key": run["model_key"],
        "model_name": run["model_name"],
        "checkpoint_path": summary["checkpoint_path"],
        "history_path": summary["history_path"],
        "loss_name": summary.get("hyperparameters", {}).get("loss_name", "bce_dice"),
        "img_size": metrics["img_size"],
        "best_epoch": summary["best_epoch"],
        "best_threshold": summary["best_threshold"],
        "best_val_dice": summary["best_val_dice"],
        "best_val_iou": summary.get("best_val_iou"),
        "best_selection_metric": summary.get("best_selection_metric", "dice_iou"),
        "best_selection_score": summary.get("best_selection_score"),
        "test_dice_global": summary["test_dice"],
        "test_iou_global": summary.get("test_iou"),
        "dice_binary_mean": metrics["dice"],
        "iou_binary_mean": metrics["iou"],
        "precision_binary_mean": metrics["precision"],
        "recall_binary_mean": metrics["recall"],
        "f1_binary_mean": metrics["f1"],
        "hausdorff_mean": metrics["hausdorff"],
        "fps_eval": metrics["fps"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "dataset_path": config.dataset_path,
    }


def save_reports(df, prefix):

    output_dir = PROJECT_ROOT / "results" / "segmentation_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    md_path = output_dir / f"{prefix}.md"

    df_sorted = df.sort_values(
        ["dice_binary_mean", "iou_binary_mean", "best_val_dice"],
        ascending=False,
    )

    df_sorted.to_csv(csv_path, index=False)
    df_sorted.to_json(json_path, orient="records", indent=2, force_ascii=False)

    table_columns = [
        "experiment_name",
        "model_name",
        "loss_name",
        "img_size",
        "dice_binary_mean",
        "iou_binary_mean",
        "best_threshold",
        "hausdorff_mean",
        "fps_eval",
    ]

    lines = [
        "# Quality Search",
        "",
        "Ranking por Dice e IoU no split de teste.",
        "",
    ]

    try:
        lines.append(df_sorted[table_columns].to_markdown(index=False))
    except Exception:
        lines.append(df_sorted[table_columns].to_csv(index=False))

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"CSV salvo em: {csv_path}")
    print(f"JSON salvo em: {json_path}")
    print(f"Markdown salvo em: {md_path}")


def main():

    args = parse_args()
    runs = build_runs(args)

    if not runs:
        print("Nenhuma execucao planejada.")
        return

    if args.dry_run:
        print("Execucoes planejadas:")
        for run in runs:
            config = run["config"]
            print(
                f"- {config.experiment_name} | modelo={run['model_name']} | "
                f"loss={config.loss_name} | img={config.img_size} | "
                f"batch={config.batch_size} | dataset={config.dataset_path}"
            )
        return

    rows = []
    for run in runs:
        summary, _ = train_or_reuse(run, args.reuse_existing)
        rows.append(evaluate_run(run, summary, args.eval_batch_size))

    df = pd.DataFrame(rows)
    save_reports(df, args.summary_prefix)

    best = df.sort_values(["dice_binary_mean", "iou_binary_mean"], ascending=False).iloc[0]
    print("\nMelhor experimento:")
    print(
        f"{best['experiment_name']} | "
        f"dice={best['dice_binary_mean']:.4f} | "
        f"iou={best['iou_binary_mean']:.4f} | "
        f"threshold={best['best_threshold']:.2f}"
    )


if __name__ == "__main__":
    main()

