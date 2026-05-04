import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.experiments.train_deeplab import build_model as build_deeplab
from src.segmentation.core.segmentation_evaluation import evaluate_segmentation_model
from src.segmentation.core.segmentation_training import SegmentationTrainingConfig, train_segmentation_model


DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "dataset_geral_cv" / "folds"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "segmentation_experiments"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Treina DeepLabV3 com validacao cruzada sobre os 70% de "
            "desenvolvimento do dataset_geral e avalia no holdout de 30%."
        )
    )
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--reuse-existing-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quando possivel, avalia checkpoints ja salvos em vez de treinar novamente.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args, fold_index):
    fold_name = f"fold_{fold_index:02d}"
    experiment_name = f"dataset_geral_deeplab_{args.backbone}_cv_{fold_name}"
    return SegmentationTrainingConfig(
        model_name="DeepLab",
        experiment_name=experiment_name,
        checkpoint_name=f"{experiment_name}.pth",
        dataset_path=str(args.splits_root / fold_name),
        model_kind="deeplab",
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name="adam",
        scheduler_name="plateau",
        scheduler_patience=4,
        scheduler_factor=0.5,
        min_lr=1e-6,
        early_stopping_patience=args.early_stopping,
        bce_weight=1.0,
        dice_weight=1.0,
        loss_name="bce_dice",
        threshold=0.5,
        threshold_search=True,
        seed=args.seed + fold_index,
        augment=args.augment,
        clahe=args.clahe,
        num_workers=args.num_workers,
        model_kwargs={
            "backbone": args.backbone,
            "pretrained": not args.no_pretrained,
        },
    )


def summarize_results(summaries, output_path):
    metrics = ["best_val_dice", "best_val_iou", "test_dice", "test_iou"]
    aggregate = {
        "folds": len(summaries),
        "summaries": summaries,
        "metrics": {},
    }
    for metric in metrics:
        values = [float(summary[metric]) for summary in summaries if metric in summary]
        if values:
            aggregate["metrics"][metric] = {
                "mean": round(sum(values) / len(values), 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_existing_checkpoint(config, args):
    summary_path = config.summary_path
    if summary_path.exists():
        return load_json(summary_path)

    checkpoint_path = config.checkpoint_path
    metadata_path = checkpoint_path.with_suffix(".meta.json")
    if not checkpoint_path.exists() or not metadata_path.exists():
        return None

    metadata = load_json(metadata_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_deeplab(**(config.model_kwargs or {})).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    threshold = float(metadata.get("best_threshold", config.threshold))
    val_metrics = evaluate_segmentation_model(
        model,
        display_name="DeepLab",
        threshold=threshold,
        dataset_path=config.dataset_path,
        img_size=config.img_size,
        batch_size=args.batch_size,
        split="val",
        device=device,
        num_workers=args.num_workers,
    )
    test_metrics = evaluate_segmentation_model(
        model,
        display_name="DeepLab",
        threshold=threshold,
        dataset_path=config.dataset_path,
        img_size=config.img_size,
        batch_size=args.batch_size,
        split="test",
        device=device,
        num_workers=args.num_workers,
    )

    summary = {
        "model_name": config.model_name,
        "experiment_name": config.experiment_name,
        "checkpoint_path": str(config.checkpoint_path),
        "history_path": str(config.history_path),
        "best_epoch": int(metadata.get("best_epoch", 0)),
        "best_val_dice": round(float(val_metrics["dice"]), 6),
        "best_val_iou": round(float(val_metrics["iou"]), 6),
        "best_selection_metric": config.selection_metric,
        "best_selection_score": round(
            float(0.5 * val_metrics["dice"] + 0.5 * val_metrics["iou"]),
            6,
        ),
        "best_threshold": round(threshold, 4),
        "test_loss": None,
        "test_dice": round(float(test_metrics["dice"]), 6),
        "test_iou": round(float(test_metrics["iou"]), 6),
        "test_precision": round(float(test_metrics["precision"]), 6),
        "test_recall": round(float(test_metrics["recall"]), 6),
        "test_f1": round(float(test_metrics["f1"]), 6),
        "test_hausdorff": round(float(test_metrics["hausdorff"]), 6),
        "test_fps": round(float(test_metrics["fps"]), 6),
        "resolved_pos_weight": metadata.get("resolved_pos_weight"),
        "train_samples": len(list((Path(config.dataset_path) / "train" / "image").glob("*"))),
        "val_samples": int(val_metrics["samples"]),
        "test_samples": int(test_metrics["samples"]),
        "elapsed_seconds": None,
        "model_kwargs": config.model_kwargs or {},
        "hyperparameters": metadata.get("hyperparameters", {}),
        "summary_source": "evaluated_existing_checkpoint",
    }
    save_json(summary_path, summary)
    return summary


def main():
    args = parse_args()
    fold_end = args.fold_end or args.folds
    configs = [build_config(args, fold_index) for fold_index in range(args.fold_start, fold_end + 1)]

    if args.dry_run:
        for config in configs:
            print(
                json.dumps(
                    {
                        "experiment_name": config.experiment_name,
                        "dataset_path": config.dataset_path,
                        "checkpoint_path": str(config.checkpoint_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        return

    summaries = []
    for config in configs:
        if not Path(config.dataset_path).exists():
            raise FileNotFoundError(f"Split nao encontrado: {config.dataset_path}")

        summary = None
        if args.reuse_existing_checkpoints:
            summary = evaluate_existing_checkpoint(config, args)
            if summary is not None:
                print(f"Reutilizando checkpoint existente: {config.experiment_name}")

        if summary is None:
            summary = train_segmentation_model(build_deeplab, config)

        summaries.append(summary)

    output_path = DEFAULT_RESULTS_ROOT / f"dataset_geral_deeplab_{args.backbone}_cv_summary.json"
    aggregate = summarize_results(summaries, output_path)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
