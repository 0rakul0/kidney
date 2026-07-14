import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1_deduplicated"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_oof_5fold"
)
EXAM_PATTERN = re.compile(r"(IM-\d{4})", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cria folds out-of-fold da capsula agrupados por exame."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def exam_id(filename):
    match = EXAM_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Identificador de exame ausente: {filename}")
    return match.group(1).upper()


def collect_records(source):
    records = []
    for split in ("train", "val", "test"):
        for image_path in sorted((source / split / "image").glob("*.png")):
            mask_path = source / split / "mask" / image_path.name
            records.append(
                {
                    "filename": image_path.name,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "exam_id": exam_id(image_path.name),
                }
            )
    return records


def balanced_group_folds(records, folds, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["exam_id"]].append(record)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    assigned = [[] for _ in range(folds)]
    counts = [0] * folds
    for _, group_records in items:
        target = min(range(folds), key=lambda index: counts[index])
        assigned[target].extend(group_records)
        counts[target] += len(group_records)
    return assigned


def split_train_validation(records, ratio, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["exam_id"]].append(record)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    target = max(1, round(len(records) * ratio))
    validation = []
    training = []
    count = 0
    for _, group_records in items:
        if count < target:
            validation.extend(group_records)
            count += len(group_records)
        else:
            training.extend(group_records)
    return training, validation


def link_record(record, destination, split):
    image_dir = destination / split / "image"
    mask_dir = destination / split / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (record["image_path"], image_dir / record["filename"]),
        (record["mask_path"], mask_dir / record["filename"]),
    ):
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def main():
    args = parse_args()
    if args.clear_output and args.output.exists():
        resolved = args.output.resolve()
        if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
            raise RuntimeError(f"Saida insegura: {resolved}")
        shutil.rmtree(resolved)
    args.output.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.source)
    test_folds = balanced_group_folds(records, args.folds, args.seed)
    rows = []
    fold_summaries = []
    all_filenames = {record["filename"] for record in records}

    for fold_index, test_records in enumerate(test_folds, 1):
        test_names = {record["filename"] for record in test_records}
        remaining = [
            record for record in records if record["filename"] not in test_names
        ]
        train_records, val_records = split_train_validation(
            remaining,
            args.validation_ratio,
            args.seed + fold_index,
        )
        fold_root = args.output / f"fold_{fold_index:02d}"
        for split, split_records in (
            ("train", train_records),
            ("val", val_records),
            ("test", test_records),
        ):
            for record in split_records:
                link_record(record, fold_root, split)
                rows.append(
                    {
                        "fold": fold_index,
                        "split": split,
                        "filename": record["filename"],
                        "exam_id": record["exam_id"],
                    }
                )
        fold_summaries.append(
            {
                "fold": fold_index,
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
                "train_exams": len({r["exam_id"] for r in train_records}),
                "val_exams": len({r["exam_id"] for r in val_records}),
                "test_exams": len({r["exam_id"] for r in test_records}),
            }
        )

    test_occurrences = defaultdict(int)
    for row in rows:
        if row["split"] == "test":
            test_occurrences[row["filename"]] += 1
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "folds": args.folds,
        "unique_images": len(records),
        "unique_exams": len({record["exam_id"] for record in records}),
        "all_images_once_as_test": (
            set(test_occurrences) == all_filenames
            and all(value == 1 for value in test_occurrences.values())
        ),
        "fold_summary": fold_summaries,
    }
    with (args.output / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
