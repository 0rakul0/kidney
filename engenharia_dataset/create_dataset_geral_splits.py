import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dataset_geral"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset_geral_cv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cria uma divisao 70/30 do dataset_geral e folds de validacao "
            "cruzada dentro dos 70% usados para treino/desenvolvimento."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument(
        "--link-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Modo de materializacao dos arquivos nas pastas de split.",
    )
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)


def normalize_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "sim"}


def bucket_key(row):
    return (
        row.get("source_name", "unknown"),
        row.get("label_source", "unknown"),
        row.get("mask_status", "unknown"),
    )


def split_rows(rows, test_ratio, folds, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[bucket_key(row)].append(row)

    test_rows = []
    development_rows = []

    for bucket_rows in buckets.values():
        shuffled = list(bucket_rows)
        rng.shuffle(shuffled)
        test_count = int(round(len(shuffled) * test_ratio))
        if len(shuffled) > 1:
            test_count = min(max(test_count, 1), len(shuffled) - 1)
        test_rows.extend(shuffled[:test_count])
        development_rows.extend(shuffled[test_count:])

    rng.shuffle(test_rows)
    rng.shuffle(development_rows)

    fold_rows = [[] for _ in range(folds)]
    dev_buckets = defaultdict(list)
    for row in development_rows:
        dev_buckets[bucket_key(row)].append(row)

    for bucket_rows in dev_buckets.values():
        shuffled = list(bucket_rows)
        rng.shuffle(shuffled)
        for index, row in enumerate(shuffled):
            fold_rows[index % folds].append(row)

    return development_rows, test_rows, fold_rows


def materialize_file(source, destination, link_mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if link_mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def materialize_split(rows, split_dir, link_mode):
    image_dir = split_dir / "image"
    mask_dir = split_dir / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for row in rows:
        image_id = row["image_id"]
        image_source = Path(row["dataset_image_path"])
        mask_source = Path(row["dataset_mask_path"])
        image_destination = image_dir / f"{image_id}.png"
        mask_destination = mask_dir / f"{image_id}.png"
        materialize_file(image_source, image_destination, link_mode)
        materialize_file(mask_source, mask_destination, link_mode)
        copied.append(
            {
                **row,
                "split_image_path": str(image_destination),
                "split_mask_path": str(mask_destination),
            }
        )
    return copied


def summarize(rows):
    by_source = defaultdict(int)
    by_mask_status = defaultdict(int)
    by_label_source = defaultdict(int)
    for row in rows:
        by_source[row.get("source_name", "unknown")] += 1
        by_mask_status[row.get("mask_status", "unknown")] += 1
        by_label_source[row.get("label_source", "unknown")] += 1
    return {
        "total": len(rows),
        "by_source": dict(sorted(by_source.items())),
        "by_mask_status": dict(sorted(by_mask_status.items())),
        "by_label_source": dict(sorted(by_label_source.items())),
    }


def main():
    args = parse_args()
    if args.clear_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.input_root / "manifest.csv"
    rows = read_manifest(manifest_path)
    eligible_rows = [
        row
        for row in rows
        if normalize_bool(row.get("has_mask")) and row.get("dataset_mask_path")
    ]

    development_rows, test_rows, fold_rows = split_rows(
        eligible_rows,
        test_ratio=args.test_ratio,
        folds=args.folds,
        seed=args.seed,
    )

    split_manifest_rows = []
    test_materialized = materialize_split(test_rows, args.output_root / "holdout_test", args.link_mode)
    write_csv(args.output_root / "holdout_test" / "manifest.csv", test_materialized)

    for fold_index in range(args.folds):
        fold_name = f"fold_{fold_index + 1:02d}"
        val_rows = fold_rows[fold_index]
        train_rows = [
            row
            for current_index, rows_for_fold in enumerate(fold_rows)
            if current_index != fold_index
            for row in rows_for_fold
        ]

        fold_root = args.output_root / "folds" / fold_name
        train_materialized = materialize_split(train_rows, fold_root / "train", args.link_mode)
        val_materialized = materialize_split(val_rows, fold_root / "val", args.link_mode)
        test_materialized_for_fold = materialize_split(test_rows, fold_root / "test", args.link_mode)

        for split_name, materialized_rows in (
            ("train", train_materialized),
            ("val", val_materialized),
            ("test", test_materialized_for_fold),
        ):
            for row in materialized_rows:
                split_manifest_rows.append(
                    {
                        "fold": fold_name,
                        "split": split_name,
                        **row,
                    }
                )
            write_csv(fold_root / split_name / "manifest.csv", materialized_rows)

    write_csv(args.output_root / "manifest.csv", split_manifest_rows)

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "folds": args.folds,
        "eligible_images_with_masks": len(eligible_rows),
        "development_70_percent": summarize(development_rows),
        "holdout_test_30_percent": summarize(test_rows),
        "folds_summary": {
            f"fold_{index + 1:02d}": {
                "train": summarize(
                    [
                        row
                        for current_index, rows_for_fold in enumerate(fold_rows)
                        if current_index != index
                        for row in rows_for_fold
                    ]
                ),
                "val": summarize(fold_rows[index]),
                "test": summarize(test_rows),
            }
            for index in range(args.folds)
        },
    }
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
