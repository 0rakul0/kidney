import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation_experiments"


def parse_args():

    parser = argparse.ArgumentParser(
        description="Compara a melhor busca quality-first com os melhores modelos puros por familia."
    )
    parser.add_argument(
        "--quality-report",
        type=str,
        default="quality_search_augmented.csv",
        help="CSV gerado por experiments/run_quality_search.py",
    )
    parser.add_argument(
        "--pure-report",
        type=str,
        default="dataset_variant_comparison.csv",
        help="CSV consolidado dos modelos puros.",
    )
    parser.add_argument(
        "--dataset-variant",
        type=str,
        default="augmented",
        choices=["original", "augmented"],
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="quality_vs_pure_models",
    )

    return parser.parse_args()


def load_csv(filename):

    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    return pd.read_csv(path)


def best_quality_per_family(df_quality):

    ordered = df_quality.sort_values(
        ["model_key", "dice_binary_mean", "iou_binary_mean", "best_val_dice"],
        ascending=[True, False, False, False],
    )

    return ordered.groupby("model_key", as_index=False).first()


def best_pure_per_family(df_pure, dataset_variant):

    subset = df_pure[df_pure["dataset_variant"] == dataset_variant].copy()
    ordered = subset.sort_values(
        ["model_key", "dice_binary_mean", "iou_binary_mean", "best_val_dice"],
        ascending=[True, False, False, False],
    )

    return ordered.groupby("model_key", as_index=False).first()


def build_comparison(df_quality_best, df_pure_best):

    comparison = df_quality_best.merge(
        df_pure_best,
        on="model_key",
        suffixes=("_quality", "_pure"),
        how="inner",
    )

    comparison["delta_dice"] = (
        comparison["dice_binary_mean_quality"] - comparison["dice_binary_mean_pure"]
    )
    comparison["delta_iou"] = (
        comparison["iou_binary_mean_quality"] - comparison["iou_binary_mean_pure"]
    )
    comparison["delta_hausdorff"] = (
        comparison["hausdorff_mean_quality"] - comparison["hausdorff_mean_pure"]
    )

    columns = [
        "model_key",
        "model_name_quality",
        "experiment_name_quality",
        "loss_name",
        "img_size_quality",
        "dice_binary_mean_quality",
        "iou_binary_mean_quality",
        "hausdorff_mean_quality",
        "experiment_name_pure",
        "dice_binary_mean_pure",
        "iou_binary_mean_pure",
        "hausdorff_mean_pure",
        "delta_dice",
        "delta_iou",
        "delta_hausdorff",
    ]

    return comparison[columns].sort_values(
        ["delta_dice", "delta_iou"],
        ascending=False,
    )


def save_outputs(comparison, output_prefix):

    csv_path = RESULTS_DIR / f"{output_prefix}.csv"
    md_path = RESULTS_DIR / f"{output_prefix}.md"

    comparison.to_csv(csv_path, index=False)

    lines = [
        "# Quality vs Pure Models",
        "",
        "Comparacao por familia entre a melhor variante quality-first e o melhor modelo puro ja benchmarkado.",
        "",
    ]

    try:
        lines.append(comparison.to_markdown(index=False))
    except Exception:
        lines.append(comparison.to_csv(index=False))

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"CSV salvo em: {csv_path}")
    print(f"Markdown salvo em: {md_path}")


def main():

    args = parse_args()

    df_quality = load_csv(args.quality_report)
    df_pure = load_csv(args.pure_report)

    comparison = build_comparison(
        best_quality_per_family(df_quality),
        best_pure_per_family(df_pure, args.dataset_variant),
    )
    save_outputs(comparison, args.output_prefix)

    if comparison.empty:
        print("Nenhuma familia em comum encontrada para comparacao.")
        return

    best = comparison.sort_values(["delta_dice", "delta_iou"], ascending=False).iloc[0]
    print(
        "Melhor ganho:",
        f"{best['model_name_quality']} | ",
        f"delta_dice={best['delta_dice']:.4f} | ",
        f"delta_iou={best['delta_iou']:.4f}",
    )


if __name__ == "__main__":
    main()

