import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_WEIGHTS_ROOT = Path(r"E:\weights\weights")
DEFAULT_LOCAL_RESULTS = (
    PROJECT_ROOT / "results" / "segmentation_experiments" / "dataset_variant_comparison.csv"
)
DEFAULT_OUTPUT_PREFIX = "kidneyus_reference_comparison"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consolida metricas do trabalho kidneyUS a partir dos summaries "
            "dos pesos nnUNet e compara com os resultados locais."
        )
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=DEFAULT_EXTERNAL_WEIGHTS_ROOT,
        help="Raiz onde estao os pesos e summaries do kidneyUS.",
    )
    parser.add_argument(
        "--local-results",
        type=Path,
        default=DEFAULT_LOCAL_RESULTS,
        help="CSV consolidado com os resultados locais do projeto.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefixo dos arquivos gerados em results/segmentation_experiments/.",
    )
    return parser.parse_args()


def find_external_summaries(weights_root):
    return sorted(weights_root.rglob("validation_raw_postprocessed/summary.json"))


def load_external_rows(summary_paths):
    rows = []
    for summary_path in summary_paths:
        parts = summary_path.parts
        if len(parts) < 9:
            continue

        group_name = parts[3]
        task_name = parts[4]
        trainer_name = parts[5]
        fold_name = parts[6]

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = data.get("results", {}).get("mean", {}).get("1", {})
        if not metrics:
            continue

        rows.append(
            {
                "source_type": "kidneyus_reference",
                "group_name": group_name,
                "task_name": task_name,
                "trainer_name": trainer_name,
                "fold_name": fold_name,
                "summary_path": str(summary_path),
                "dice": metrics.get("Dice"),
                "iou": metrics.get("Jaccard"),
                "precision": metrics.get("Precision"),
                "recall": metrics.get("Recall"),
                "accuracy": metrics.get("Accuracy"),
            }
        )

    return pd.DataFrame(rows)


def aggregate_external_metrics(df_external):
    if df_external.empty:
        return pd.DataFrame()

    grouped = (
        df_external.groupby(["group_name", "task_name", "trainer_name"], as_index=False)
        .agg(
            fold_count=("fold_name", "count"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_min=("dice", "min"),
            dice_max=("dice", "max"),
            iou_mean=("iou", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            accuracy_mean=("accuracy", "mean"),
        )
        .sort_values(["task_name", "dice_mean"], ascending=[True, False])
    )
    return grouped


def load_local_results(local_results_path):
    df_local = pd.read_csv(local_results_path)
    return df_local.sort_values("dice_binary_mean", ascending=False)


def build_local_best_tables(df_local):
    overall_best = df_local.head(1).copy()

    family_best = (
        df_local.sort_values("dice_binary_mean", ascending=False)
        .groupby("model_name", as_index=False)
        .head(1)
        .sort_values("dice_binary_mean", ascending=False)
        .reset_index(drop=True)
    )

    return overall_best, family_best


def build_comparison_table(df_external_agg, df_local_family_best, df_local_overall_best):
    task001 = df_external_agg[df_external_agg["task_name"] == "Task001_KidneyCapsule"].copy()
    task002 = df_external_agg[df_external_agg["task_name"] == "Task002_KidneyRegions"].copy()

    overall_best = df_local_overall_best.iloc[0]

    comparison_rows = []
    for _, row in task001.iterrows():
        comparison_rows.append(
            {
                "comparison_scope": "overall_best_vs_kidneyus_task001",
                "external_group": row["group_name"],
                "external_task": row["task_name"],
                "external_dice_mean": row["dice_mean"],
                "external_iou_mean": row["iou_mean"],
                "local_model": overall_best["model_name"],
                "local_experiment": overall_best["experiment_name"],
                "local_dataset_variant": overall_best["dataset_variant"],
                "local_dice": overall_best["dice_binary_mean"],
                "local_iou": overall_best["iou_binary_mean"],
                "delta_local_minus_external_dice": overall_best["dice_binary_mean"] - row["dice_mean"],
            }
        )

    for _, local_row in df_local_family_best.iterrows():
        mixed_task001 = task001[task001["group_name"] == "mixed"]
        mixed_task002 = task002[task002["group_name"] == "mixed"]

        reference_task001 = mixed_task001.iloc[0] if not mixed_task001.empty else None
        reference_task002 = mixed_task002.iloc[0] if not mixed_task002.empty else None

        comparison_rows.append(
            {
                "comparison_scope": "family_best_vs_kidneyus_mixed",
                "external_group": "mixed",
                "external_task": "Task001_KidneyCapsule",
                "external_dice_mean": None if reference_task001 is None else reference_task001["dice_mean"],
                "external_iou_mean": None if reference_task001 is None else reference_task001["iou_mean"],
                "local_model": local_row["model_name"],
                "local_experiment": local_row["experiment_name"],
                "local_dataset_variant": local_row["dataset_variant"],
                "local_dice": local_row["dice_binary_mean"],
                "local_iou": local_row["iou_binary_mean"],
                "delta_local_minus_external_dice": None
                if reference_task001 is None
                else local_row["dice_binary_mean"] - reference_task001["dice_mean"],
            }
        )
        comparison_rows.append(
            {
                "comparison_scope": "family_best_vs_kidneyus_mixed",
                "external_group": "mixed",
                "external_task": "Task002_KidneyRegions",
                "external_dice_mean": None if reference_task002 is None else reference_task002["dice_mean"],
                "external_iou_mean": None if reference_task002 is None else reference_task002["iou_mean"],
                "local_model": local_row["model_name"],
                "local_experiment": local_row["experiment_name"],
                "local_dataset_variant": local_row["dataset_variant"],
                "local_dice": local_row["dice_binary_mean"],
                "local_iou": local_row["iou_binary_mean"],
                "delta_local_minus_external_dice": None
                if reference_task002 is None
                else local_row["dice_binary_mean"] - reference_task002["dice_mean"],
            }
        )

    return pd.DataFrame(comparison_rows)


def save_outputs(df_external_folds, df_external_agg, df_local_family_best, df_local_overall_best, df_compare, output_prefix):
    output_dir = PROJECT_ROOT / "results" / "segmentation_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    external_folds_path = output_dir / f"{output_prefix}_external_folds.csv"
    external_agg_path = output_dir / f"{output_prefix}_external_aggregated.csv"
    family_best_path = output_dir / f"{output_prefix}_local_family_best.csv"
    compare_path = output_dir / f"{output_prefix}_comparison.csv"
    json_path = output_dir / f"{output_prefix}.json"
    markdown_path = output_dir / f"{output_prefix}.md"

    df_external_folds.to_csv(external_folds_path, index=False)
    df_external_agg.to_csv(external_agg_path, index=False)
    df_local_family_best.to_csv(family_best_path, index=False)
    df_compare.to_csv(compare_path, index=False)

    payload = {
        "external_aggregated": df_external_agg.to_dict(orient="records"),
        "local_overall_best": df_local_overall_best.to_dict(orient="records"),
        "local_family_best": df_local_family_best.to_dict(orient="records"),
        "comparison_table": df_compare.to_dict(orient="records"),
        "notes": [
            "As metricas do kidneyUS foram extraidas dos summaries de validacao postprocessed encontrados junto aos pesos nnUNet.",
            "As metricas locais foram extraidas do CSV consolidado dataset_variant_comparison.csv.",
            "A comparacao e util como referencia de trabalho relacionado, mas nao representa um benchmark estritamente identico de split, anotacao e protocolo.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def to_markdown_or_csv(df):
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    lines = [
        "# Comparacao com kidneyUS",
        "",
        "Este relatorio compara os melhores resultados locais com as metricas de validacao presentes nos artefatos do projeto kidneyUS.",
        "",
        "Importante: a comparacao e contextual e nao estritamente equivalente, porque os protocolos de split, anotacao e tarefa podem diferir.",
        "",
        "## Melhor resultado local",
        "",
    ]

    local_overall_subset = df_local_overall_best[
        ["model_name", "experiment_name", "dataset_variant", "dice_binary_mean", "iou_binary_mean", "f1_binary_mean", "hausdorff_mean", "fps_eval"]
    ]
    lines.append(to_markdown_or_csv(local_overall_subset))
    lines.append("")
    lines.append("## Melhor resultado local por familia")
    lines.append("")
    local_family_subset = df_local_family_best[
        ["model_name", "experiment_name", "dataset_variant", "dice_binary_mean", "iou_binary_mean", "f1_binary_mean", "hausdorff_mean", "fps_eval"]
    ]
    lines.append(to_markdown_or_csv(local_family_subset))
    lines.append("")
    lines.append("## Referencia kidneyUS agregada dos pesos nnUNet")
    lines.append("")
    external_subset = df_external_agg[
        ["group_name", "task_name", "fold_count", "dice_mean", "iou_mean", "precision_mean", "recall_mean", "dice_min", "dice_max"]
    ]
    lines.append(to_markdown_or_csv(external_subset))
    lines.append("")
    lines.append("## Deltas de comparacao")
    lines.append("")
    compare_subset = df_compare[
        ["comparison_scope", "external_group", "external_task", "local_model", "local_experiment", "local_dataset_variant", "local_dice", "external_dice_mean", "delta_local_minus_external_dice"]
    ]
    lines.append(to_markdown_or_csv(compare_subset))
    lines.append("")
    lines.append("## Observacoes")
    lines.append("")
    lines.append("- Task001_KidneyCapsule e a referencia mais proxima da segmentacao binaria externa do rim usada neste projeto.")
    lines.append("- Task002_KidneyRegions e uma tarefa mais rica e menos diretamente equivalente ao problema atual.")
    lines.append("- Os resultados do kidneyUS resumem validacao de folds do nnUNet e nao o mesmo split local train/val/test deste repositorio.")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "external_folds_path": external_folds_path,
        "external_agg_path": external_agg_path,
        "family_best_path": family_best_path,
        "compare_path": compare_path,
        "json_path": json_path,
        "markdown_path": markdown_path,
    }


def main():
    args = parse_args()

    if not args.weights_root.exists():
        raise FileNotFoundError(f"Pasta de pesos nao encontrada: {args.weights_root}")

    if not args.local_results.exists():
        raise FileNotFoundError(f"CSV local nao encontrado: {args.local_results}")

    summary_paths = find_external_summaries(args.weights_root)
    if not summary_paths:
        raise RuntimeError("Nenhum summary.json postprocessed do kidneyUS foi encontrado.")

    df_external_folds = load_external_rows(summary_paths)
    df_external_agg = aggregate_external_metrics(df_external_folds)

    df_local = load_local_results(args.local_results)
    df_local_overall_best, df_local_family_best = build_local_best_tables(df_local)
    df_compare = build_comparison_table(df_external_agg, df_local_family_best, df_local_overall_best)

    outputs = save_outputs(
        df_external_folds,
        df_external_agg,
        df_local_family_best,
        df_local_overall_best,
        df_compare,
        args.output_prefix,
    )

    print("Arquivos gerados:")
    for _, path in outputs.items():
        print(path)

    best_local = df_local_overall_best.iloc[0]
    best_task001 = (
        df_external_agg[df_external_agg["task_name"] == "Task001_KidneyCapsule"]
        .sort_values("dice_mean", ascending=False)
        .iloc[0]
    )
    print("\nResumo:")
    print(
        f"Melhor local: {best_local['experiment_name']} | "
        f"Dice={best_local['dice_binary_mean']:.4f}"
    )
    print(
        f"Melhor kidneyUS Task001: {best_task001['group_name']} | "
        f"Dice={best_task001['dice_mean']:.4f}"
    )
    print(
        f"Delta local - kidneyUS Task001: "
        f"{best_local['dice_binary_mean'] - best_task001['dice_mean']:.4f}"
    )


if __name__ == "__main__":
    main()

