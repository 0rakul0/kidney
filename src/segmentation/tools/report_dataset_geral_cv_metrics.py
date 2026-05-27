import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.experiments.train_deeplab import build_model as build_deeplab
from src.segmentation.core.segmentation_evaluation import evaluate_segmentation_model


RESULTS_ROOT = PROJECT_ROOT / "results" / "segmentation_experiments"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "dataset_geral_cv_report"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera curvas de aprendizado e metricas detalhadas dos folds "
            "DeepLabV3 treinados no dataset_geral."
        )
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fold_name(index):
    return f"dataset_geral_deeplab_resnet50_cv_fold_{index:02d}"


def load_histories(folds):
    frames = []
    for index in range(1, folds + 1):
        path = RESULTS_ROOT / f"{fold_name(index)}_history.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "fold", index)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_learning_curve(history, output_root):
    if history.empty:
        return None

    plots_root = output_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    output_path = plots_root / "deeplab_dataset_geral_learning_curves.png"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)
    axes = axes.ravel()

    for fold, frame in history.groupby("fold"):
        label = f"fold {fold}"
        axes[0].plot(frame["epoch"], frame["train_loss"], linewidth=1.8, label=label)
        axes[1].plot(frame["epoch"], frame["val_loss"], linewidth=1.8, label=label)
        axes[2].plot(frame["epoch"], frame["train_dice"], linewidth=1.8, label=label)
        axes[3].plot(frame["epoch"], frame["val_dice"], linewidth=1.8, label=label)

    titles = [
        "Perda de treino",
        "Perda de validacao",
        "Dice de treino",
        "Dice de validacao",
    ]
    ylabels = ["loss", "loss", "Dice", "Dice"]
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("Epoca")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_validation_curve(history, output_root):
    if history.empty:
        return None

    plots_root = output_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    output_path = plots_root / "deeplab_dataset_geral_validation_iou_threshold.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    for fold, frame in history.groupby("fold"):
        label = f"fold {fold}"
        axes[0].plot(frame["epoch"], frame["val_iou"], linewidth=1.8, label=label)
        axes[1].plot(frame["epoch"], frame["selected_threshold"], linewidth=1.8, label=label)

    axes[0].set_title("IoU de validacao")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("IoU")
    axes[1].set_title("Limiar selecionado")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Threshold")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def evaluate_fold(index, args):
    name = fold_name(index)
    summary_path = RESULTS_ROOT / f"{name}_summary.json"
    metadata_path = PROJECT_ROOT / "models" / f"{name}.meta.json"
    checkpoint_path = PROJECT_ROOT / "models" / f"{name}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata nao encontrado: {metadata_path}")

    summary = read_json(summary_path) if summary_path.exists() else {}
    metadata = read_json(metadata_path)
    hyper = metadata.get("hyperparameters", summary.get("hyperparameters", {}))
    dataset_path = hyper.get("dataset_path") or str(PROJECT_ROOT / "dataset_aumentado" / "dataset_geral_cv" / "folds" / f"fold_{index:02d}")
    threshold = float(metadata.get("best_threshold", summary.get("best_threshold", 0.5)))
    model_kwargs = metadata.get("model_kwargs") or summary.get("model_kwargs") or {
        "backbone": "resnet50",
        "pretrained": True,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_deeplab(**model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_metrics = evaluate_segmentation_model(
        model,
        display_name="DeepLab",
        threshold=threshold,
        dataset_path=dataset_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        split="test",
        device=device,
        num_workers=args.num_workers,
    )

    return {
        "fold": index,
        "best_epoch": int(metadata.get("best_epoch", summary.get("best_epoch", 0))),
        "threshold": threshold,
        "val_dice": float(metadata.get("best_val_dice", summary.get("best_val_dice", 0.0))),
        "val_iou": float(metadata.get("best_val_iou", summary.get("best_val_iou", 0.0))),
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_hausdorff": test_metrics["hausdorff"],
        "test_fps": test_metrics["fps"],
        "test_samples": test_metrics["samples"],
        "checkpoint_path": str(checkpoint_path),
    }


def summarize_metric_rows(rows):
    metric_keys = [
        "val_dice",
        "val_iou",
        "test_dice",
        "test_iou",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_hausdorff",
        "test_fps",
    ]
    aggregate = {"folds": len(rows), "metrics": {}}
    for key in metric_keys:
        values = [float(row[key]) for row in rows]
        aggregate["metrics"][key] = {
            "mean": round(sum(values) / len(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    aggregate["best_fold_by_test_dice"] = max(rows, key=lambda row: row["test_dice"])["fold"]
    aggregate["best_fold_by_test_precision"] = max(rows, key=lambda row: row["test_precision"])["fold"]
    return aggregate


def save_markdown(output_root, metrics_rows, aggregate, learning_curve, validation_curve):
    path = output_root / "dataset_geral_deeplab_cv_metricas.md"
    lines = [
        "# Metricas DeepLabV3 no dataset_geral",
        "",
        "## Curvas",
        "",
    ]
    if learning_curve is not None:
        lines.append(f"- Curva de aprendizado: `{learning_curve}`")
    if validation_curve is not None:
        lines.append(f"- Curva de IoU/threshold: `{validation_curve}`")
    lines.extend(
        [
            "",
            "## Metricas no teste fixo",
            "",
            "| Fold | Dice | IoU | Precisao | Recall | F1 | Hausdorff | FPS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics_rows:
        lines.append(
            "| {fold} | {test_dice:.4f} | {test_iou:.4f} | {test_precision:.4f} | "
            "{test_recall:.4f} | {test_f1:.4f} | {test_hausdorff:.2f} | {test_fps:.2f} |".format(**row)
        )
    metrics = aggregate["metrics"]
    lines.extend(
        [
            "",
            "## Consolidado",
            "",
            f"- Dice medio: {metrics['test_dice']['mean']:.4f}.",
            f"- IoU medio: {metrics['test_iou']['mean']:.4f}.",
            f"- Precisao media: {metrics['test_precision']['mean']:.4f}.",
            f"- Recall medio: {metrics['test_recall']['mean']:.4f}.",
            f"- F1 medio: {metrics['test_f1']['mean']:.4f}.",
            f"- Melhor fold por Dice: {aggregate['best_fold_by_test_dice']}.",
            f"- Melhor fold por precisao: {aggregate['best_fold_by_test_precision']}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    history = load_histories(args.folds)
    history_path = args.output_root / "dataset_geral_deeplab_cv_history_all_folds.csv"
    if not history.empty:
        history.to_csv(history_path, index=False)

    learning_curve = save_learning_curve(history, args.output_root)
    validation_curve = save_validation_curve(history, args.output_root)

    metrics_rows = []
    if not args.skip_eval:
        for index in range(1, args.folds + 1):
            metrics_rows.append(evaluate_fold(index, args))
    else:
        for index in range(1, args.folds + 1):
            summary = read_json(RESULTS_ROOT / f"{fold_name(index)}_summary.json")
            metrics_rows.append(
                {
                    "fold": index,
                    "best_epoch": summary.get("best_epoch"),
                    "threshold": summary.get("best_threshold"),
                    "val_dice": summary.get("best_val_dice"),
                    "val_iou": summary.get("best_val_iou"),
                    "test_dice": summary.get("test_dice"),
                    "test_iou": summary.get("test_iou"),
                    "test_precision": summary.get("test_precision"),
                    "test_recall": summary.get("test_recall"),
                    "test_f1": summary.get("test_f1"),
                    "test_hausdorff": summary.get("test_hausdorff"),
                    "test_fps": summary.get("test_fps"),
                    "test_samples": summary.get("test_samples"),
                    "checkpoint_path": summary.get("checkpoint_path"),
                }
            )

    metrics_csv = args.output_root / "dataset_geral_deeplab_cv_test_metrics.csv"
    write_csv(metrics_csv, metrics_rows)
    aggregate = summarize_metric_rows(metrics_rows)
    aggregate["history_csv"] = str(history_path)
    aggregate["metrics_csv"] = str(metrics_csv)
    aggregate["learning_curve_png"] = None if learning_curve is None else str(learning_curve)
    aggregate["validation_curve_png"] = None if validation_curve is None else str(validation_curve)
    write_json(args.output_root / "dataset_geral_deeplab_cv_metrics_summary.json", aggregate)
    markdown_path = save_markdown(args.output_root, metrics_rows, aggregate, learning_curve, validation_curve)

    print(json.dumps({**aggregate, "markdown": str(markdown_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
