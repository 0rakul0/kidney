import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.experiments.run_hyperparameter_search import MODEL_REGISTRY
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.core.segmentation_evaluation import evaluate_segmentation_model
from src.segmentation.core.segmentation_training import SegmentationTrainingConfig, train_segmentation_model


DATASET_VARIANTS = {
    "original": "dataset_inicial",
    "augmented": "dataset_aumentado/expansao_pseudorrotulada",
}

EXISTING_ORIGINAL_EXPERIMENTS = {
    ("unet", "baseline"): "unet_baseline",
    ("unet", "capacity_augmented"): "unet_capacity_augmented",
    ("deeplab", "resnet50_baseline"): "deeplab_resnet50_baseline",
    ("deeplab", "resnet101_capacity"): "deeplab_resnet101_capacity",
    ("segformer", "b0_baseline"): "segformer_b0_baseline_10ep",
    ("segformer", "b2_capacity"): "segformer_b2_capacity_8ep",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Treina e compara modelos baseline e ajustados em variantes "
            "de dataset, gerando um relatorio unico de metricas."
        )
    )
    parser.add_argument("--model", choices=["all", *MODEL_REGISTRY.keys()], default="all")
    parser.add_argument(
        "--dataset-variant",
        choices=["all", *DATASET_VARIANTS.keys()],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-prefix", type=str, default="dataset_variant_comparison")
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reaproveita summaries e checkpoints existentes quando disponiveis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra as execucoes planejadas sem treinar.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Limita o numero de execucoes para smoke tests.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size usado na avaliacao detalhada apos o treino.",
    )
    return parser.parse_args()


def selected_keys(value, registry):
    if value == "all":
        return list(registry.keys())
    return [value]


def build_base_config(model_key, model_info, dataset_path, epochs, seed):
    return SegmentationTrainingConfig(
        model_name=model_info["display_name"],
        experiment_name=f"{model_key}_baseline",
        checkpoint_name=f"{model_info['checkpoint_prefix']}_best.pth",
        dataset_path=dataset_path,
        model_kind=model_info["model_kind"],
        epochs=epochs,
        seed=seed,
        threshold_search=True,
        early_stopping_patience=8,
        model_kwargs={},
    )


def load_summary_by_experiment_name(experiment_name):
    summary_path = (
        PROJECT_ROOT
        / "results"
        / "segmentation_experiments"
        / f"{experiment_name}_summary.json"
    )
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def apply_reference_hyperparameters(config, reference_summary):
    if reference_summary is None:
        return config

    hyper = reference_summary.get("hyperparameters", {})
    if not hyper:
        return config

    config.img_size = hyper.get("img_size", config.img_size)
    config.batch_size = hyper.get("batch_size", config.batch_size)
    config.epochs = hyper.get("epochs", config.epochs)
    config.learning_rate = hyper.get("learning_rate", config.learning_rate)
    config.weight_decay = hyper.get("weight_decay", config.weight_decay)
    config.optimizer_name = hyper.get("optimizer_name", config.optimizer_name)
    config.scheduler_name = hyper.get("scheduler_name", config.scheduler_name)
    config.scheduler_patience = hyper.get("scheduler_patience", config.scheduler_patience)
    config.scheduler_factor = hyper.get("scheduler_factor", config.scheduler_factor)
    config.min_lr = hyper.get("min_lr", config.min_lr)
    config.early_stopping_patience = hyper.get(
        "early_stopping_patience",
        config.early_stopping_patience,
    )
    config.bce_weight = hyper.get("bce_weight", config.bce_weight)
    config.dice_weight = hyper.get("dice_weight", config.dice_weight)
    config.loss_name = hyper.get("loss_name", config.loss_name)
    config.focal_gamma = hyper.get("focal_gamma", config.focal_gamma)
    config.focal_alpha = hyper.get("focal_alpha", config.focal_alpha)
    config.tversky_alpha = hyper.get("tversky_alpha", config.tversky_alpha)
    config.tversky_beta = hyper.get("tversky_beta", config.tversky_beta)
    config.focal_tversky_gamma = hyper.get(
        "focal_tversky_gamma",
        config.focal_tversky_gamma,
    )
    config.threshold = hyper.get("threshold", config.threshold)
    config.threshold_search = hyper.get("threshold_search", config.threshold_search)
    config.threshold_candidates = tuple(
        hyper.get("threshold_candidates", config.threshold_candidates)
    )
    config.threshold_metric = hyper.get("threshold_metric", config.threshold_metric)
    config.threshold_metric_weight = hyper.get(
        "threshold_metric_weight",
        config.threshold_metric_weight,
    )
    config.selection_metric = hyper.get("selection_metric", config.selection_metric)
    config.selection_metric_weight = hyper.get(
        "selection_metric_weight",
        config.selection_metric_weight,
    )
    config.pos_weight = hyper.get("pos_weight", config.pos_weight)
    config.auto_pos_weight = hyper.get("auto_pos_weight", config.auto_pos_weight)
    config.augment = hyper.get("augment", config.augment)
    config.clahe = hyper.get("clahe", config.clahe)
    config.num_workers = hyper.get("num_workers", config.num_workers)
    config.model_kwargs = hyper.get("model_kwargs", config.model_kwargs)
    return config


def planned_runs(args):
    runs = []
    model_keys = selected_keys(args.model, MODEL_REGISTRY)
    dataset_keys = selected_keys(args.dataset_variant, DATASET_VARIANTS)

    for dataset_key in dataset_keys:
        dataset_path = DATASET_VARIANTS[dataset_key]

        for model_key in model_keys:
            model_info = MODEL_REGISTRY[model_key]
            base_config = build_base_config(
                model_key,
                model_info,
                dataset_path=dataset_path,
                epochs=args.epochs,
                seed=args.seed,
            )

            for experiment in model_info["experiments"]:
                experiment_label = experiment["name"]
                run_name = f"{dataset_key}_{model_key}_{experiment_label}"
                checkpoint_name = f"{run_name}.pth"
                variant_type = "baseline" if "baseline" in experiment_label else "tuned"
                reference_experiment_name = EXISTING_ORIGINAL_EXPERIMENTS.get(
                    (model_key, experiment_label)
                )
                reference_summary = load_summary_by_experiment_name(reference_experiment_name)

                config_kwargs = {
                    **base_config.__dict__,
                    **experiment["overrides"],
                    "experiment_name": run_name,
                    "checkpoint_name": checkpoint_name,
                    "dataset_path": dataset_path,
                    "model_kwargs": experiment["model_kwargs"],
                }

                config = SegmentationTrainingConfig(**config_kwargs)
                if reference_summary is not None:
                    config = apply_reference_hyperparameters(config, reference_summary)
                runs.append(
                    {
                        "dataset_variant": dataset_key,
                        "dataset_path": dataset_path,
                        "model_key": model_key,
                        "model_name": model_info["display_name"],
                        "experiment_label": experiment_label,
                        "variant_type": variant_type,
                        "builder": model_info["builder"],
                        "config": config,
                        "reference_experiment_name": reference_experiment_name,
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
    summary_path = config.summary_path
    checkpoint_path = config.checkpoint_path
    reference_experiment_name = run.get("reference_experiment_name")

    if reuse_existing and summary_path.exists() and checkpoint_path.exists():
        summary = load_existing_summary(summary_path)
        print(f"Reutilizando: {config.experiment_name}")
        return summary, True

    if (
        reuse_existing
        and run["dataset_variant"] == "original"
        and reference_experiment_name is not None
    ):
        reference_summary = load_summary_by_experiment_name(reference_experiment_name)
        if reference_summary is not None and Path(reference_summary["checkpoint_path"]).exists():
            print(f"Reutilizando original existente: {reference_experiment_name}")
            return reference_summary, True

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
        dataset_path=run["dataset_path"],
        img_size=summary.get("hyperparameters", {}).get("img_size", config.img_size),
        batch_size=eval_batch_size,
        split="test",
        device=device,
        num_workers=summary.get("hyperparameters", {}).get("num_workers", config.num_workers),
    )

    return {
        "dataset_variant": run["dataset_variant"],
        "dataset_path": run["dataset_path"],
        "model_key": run["model_key"],
        "model_name": run["model_name"],
        "experiment_label": run["experiment_label"],
        "variant_type": run["variant_type"],
        "experiment_name": summary["experiment_name"],
        "checkpoint_path": summary["checkpoint_path"],
        "history_path": summary["history_path"],
        "best_epoch": summary["best_epoch"],
        "best_val_dice": summary["best_val_dice"],
        "best_threshold": summary["best_threshold"],
        "test_loss": summary["test_loss"],
        "test_dice_global": summary["test_dice"],
        "resolved_pos_weight": summary["resolved_pos_weight"],
        "train_samples": summary["train_samples"],
        "val_samples": summary["val_samples"],
        "test_samples": summary["test_samples"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "dice_binary_mean": metrics["dice"],
        "iou_binary_mean": metrics["iou"],
        "precision_binary_mean": metrics["precision"],
        "recall_binary_mean": metrics["recall"],
        "f1_binary_mean": metrics["f1"],
        "hausdorff_mean": metrics["hausdorff"],
        "fps_eval": metrics["fps"],
        "eval_batch_size": metrics["batch_size"],
        "recomputed_test_samples": metrics["samples"],
        "img_size": metrics["img_size"],
    }


def add_comparative_columns(df):
    baseline_map = (
        df[df["variant_type"] == "baseline"]
        .set_index(["dataset_variant", "model_key"])["dice_binary_mean"]
        .to_dict()
    )

    df["delta_dice_binary_vs_dataset_baseline"] = df.apply(
        lambda row: row["dice_binary_mean"] - baseline_map.get(
            (row["dataset_variant"], row["model_key"]),
            row["dice_binary_mean"],
        ),
        axis=1,
    )

    original_map = (
        df[df["dataset_variant"] == "original"]
        .set_index(["model_key", "experiment_label"])["dice_binary_mean"]
        .to_dict()
    )

    df["delta_dice_binary_vs_original_same_experiment"] = df.apply(
        lambda row: row["dice_binary_mean"] - original_map.get(
            (row["model_key"], row["experiment_label"]),
            row["dice_binary_mean"],
        ),
        axis=1,
        result_type="reduce",
    )

    df["rank_overall"] = df["dice_binary_mean"].rank(method="dense", ascending=False).astype(int)
    return df


def save_reports(df, prefix):
    output_dir = PROJECT_ROOT / "results" / "segmentation_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    markdown_path = output_dir / f"{prefix}.md"

    df_sorted = df.sort_values(
        ["dataset_variant", "model_key", "dice_binary_mean"],
        ascending=[True, True, False],
    )
    df_sorted.to_csv(csv_path, index=False)
    df_sorted.to_json(json_path, orient="records", indent=2, force_ascii=False)

    lines = [
        "# Comparative segmentation results",
        "",
        "Sorted by dataset variant, model, and Dice.",
        "",
    ]

    for dataset_variant in df_sorted["dataset_variant"].unique():
        lines.append(f"## {dataset_variant}")
        lines.append("")
        subset = df_sorted[df_sorted["dataset_variant"] == dataset_variant][
            [
                "model_name",
                "experiment_label",
                "variant_type",
                "dice_binary_mean",
                "iou_binary_mean",
                "precision_binary_mean",
                "recall_binary_mean",
                "f1_binary_mean",
                "hausdorff_mean",
                "fps_eval",
                "delta_dice_binary_vs_dataset_baseline",
                "delta_dice_binary_vs_original_same_experiment",
            ]
        ]
        try:
            lines.append(subset.to_markdown(index=False))
        except Exception:
            lines.append(subset.to_csv(index=False))
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"CSV salvo em: {csv_path}")
    print(f"JSON salvo em: {json_path}")
    print(f"Markdown salvo em: {markdown_path}")


def main():
    args = parse_args()
    runs = planned_runs(args)

    if not runs:
        print("Nenhuma execucao planejada.")
        return

    if args.dry_run:
        print("Execucoes planejadas:")
        for run in runs:
            print(
                f"- {run['config'].experiment_name} | "
                f"dataset={run['dataset_variant']} | "
                f"modelo={run['model_name']} | "
                f"tipo={run['variant_type']} | "
                f"path={run['dataset_path']}"
            )
        return

    rows = []
    for run in runs:
        summary, _ = train_or_reuse(run, reuse_existing=args.reuse_existing)
        row = evaluate_run(run, summary, eval_batch_size=args.eval_batch_size)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_comparative_columns(df)
    save_reports(df, args.summary_prefix)

    best = df.sort_values("dice_binary_mean", ascending=False).iloc[0]
    print("\nMelhor execucao:")
    print(
        f"{best['experiment_name']} | dataset={best['dataset_variant']} | "
        f"dice={best['dice_binary_mean']:.4f} | "
        f"iou={best['iou_binary_mean']:.4f} | "
        f"f1={best['f1_binary_mean']:.4f}"
    )


if __name__ == "__main__":
    main()

