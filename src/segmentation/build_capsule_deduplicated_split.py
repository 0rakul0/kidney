import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1_deduplicated"
)
EXAM_PATTERN = re.compile(r"(IM-\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deduplica a base kidneyUS Capsule por hash e refaz os splits "
            "agrupando todas as imagens do mesmo exame."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def file_sha1(path):
    digest = hashlib.sha1()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Máscara ilegível: {path}")
    return mask > 0


def dice(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    denominator = mask_a.sum() + mask_b.sum()
    return 1.0 if denominator == 0 else float(2.0 * intersection / denominator)


def extract_exam_id(filename):
    match = EXAM_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Identificador de exame não encontrado: {filename}")
    return match.group(1).upper()


def load_rows(input_root):
    manifest_path = input_root / "manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    loaded = []
    for row in rows:
        image_path = Path(row["split_image_path"])
        mask_path = Path(row["split_mask_path"])
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Par ausente: {image_path} / {mask_path}")
        loaded.append(
            {
                **row,
                "image_path": image_path,
                "mask_path": mask_path,
                "image_sha1": file_sha1(image_path),
                "mask_sha1": file_sha1(mask_path),
                "exam_id": extract_exam_id(row["filename"]),
                "original_split": row["split"],
            }
        )
    return loaded


def choose_mask_medoid(group):
    if len(group) == 1:
        return group[0], 1.0

    masks = [read_mask(row["mask_path"]) for row in group]
    mean_scores = []
    for index, mask in enumerate(masks):
        scores = [
            dice(mask, other)
            for other_index, other in enumerate(masks)
            if other_index != index
        ]
        mean_scores.append(float(np.mean(scores)))

    best_index = max(
        range(len(group)),
        key=lambda index: (mean_scores[index], -int(group[index]["filename"].split("_", 1)[0])),
    )
    return group[best_index], mean_scores[best_index]


def deduplicate(rows):
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["image_sha1"]].append(row)

    unique_rows = []
    duplicate_report = []
    for image_sha1, group in sorted(by_hash.items()):
        selected, mean_mask_dice = choose_mask_medoid(group)
        unique_rows.append(
            {
                **selected,
                "duplicate_count": len(group),
                "selected_mask_mean_dice": mean_mask_dice,
                "duplicate_filenames": ";".join(sorted(row["filename"] for row in group)),
                "duplicate_original_splits": ";".join(
                    sorted(set(row["original_split"] for row in group))
                ),
            }
        )
        if len(group) > 1:
            for row in group:
                duplicate_report.append(
                    {
                        "image_sha1": image_sha1,
                        "exam_id": selected["exam_id"],
                        "selected_filename": selected["filename"],
                        "selected_mask_sha1": selected["mask_sha1"],
                        "selected_mask_mean_dice": f"{mean_mask_dice:.6f}",
                        "candidate_filename": row["filename"],
                        "candidate_mask_sha1": row["mask_sha1"],
                        "candidate_original_split": row["original_split"],
                        "selected": str(row["filename"] == selected["filename"]).lower(),
                    }
                )
    return unique_rows, duplicate_report


def assign_grouped_splits(rows, seed, train_ratio, val_ratio):
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("As proporções devem ser positivas e somar 1.")

    by_exam = defaultdict(list)
    for row in rows:
        by_exam[row["exam_id"]].append(row)

    rng = random.Random(seed)
    groups = list(by_exam.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    total = len(rows)
    targets = {
        "train": round(total * train_ratio),
        "val": round(total * val_ratio),
    }
    targets["test"] = total - targets["train"] - targets["val"]
    assigned = {"train": [], "val": [], "test": []}

    for exam_id, group in groups:
        size = len(group)
        split = max(
            assigned,
            key=lambda name: (
                (targets[name] - len(assigned[name])) / max(targets[name], 1),
                -len(assigned[name]),
            ),
        )
        for row in group:
            assigned[split].append({**row, "split": split})

    return assigned, targets


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_output(output_root, assigned):
    manifest_rows = []
    for split, rows in assigned.items():
        image_dir = output_root / split / "image"
        mask_dir = output_root / split / "mask"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        split_rows = []
        for row in sorted(rows, key=lambda value: value["filename"]):
            output_image = image_dir / row["filename"]
            output_mask = mask_dir / row["filename"]
            shutil.copy2(row["image_path"], output_image)
            shutil.copy2(row["mask_path"], output_mask)
            exported = {
                "filename": row["filename"],
                "exam_id": row["exam_id"],
                "split": split,
                "image_sha1": row["image_sha1"],
                "mask_sha1": row["mask_sha1"],
                "source_image_path": row["source_image_path"],
                "source_mask_path": str(row["mask_path"]),
                "split_image_path": str(output_image),
                "split_mask_path": str(output_mask),
                "original_split": row["original_split"],
                "duplicate_count": row["duplicate_count"],
                "duplicate_filenames": row["duplicate_filenames"],
                "duplicate_original_splits": row["duplicate_original_splits"],
                "selected_mask_mean_dice": f"{row['selected_mask_mean_dice']:.6f}",
            }
            split_rows.append(exported)
            manifest_rows.append(exported)
        write_csv(output_root / split / "manifest.csv", split_rows)
    write_csv(output_root / "manifest.csv", manifest_rows)
    return manifest_rows


def audit(manifest_rows):
    hash_splits = defaultdict(set)
    exam_splits = defaultdict(set)
    for row in manifest_rows:
        hash_splits[row["image_sha1"]].add(row["split"])
        exam_splits[row["exam_id"]].add(row["split"])
    cross_hash = {key: sorted(value) for key, value in hash_splits.items() if len(value) > 1}
    cross_exam = {key: sorted(value) for key, value in exam_splits.items() if len(value) > 1}
    if cross_hash or cross_exam:
        raise RuntimeError(
            f"Vazamento detectado: hashes={len(cross_hash)}, exames={len(cross_exam)}"
        )
    return {
        "unique_hashes": len(hash_splits),
        "unique_exams": len(exam_splits),
        "cross_split_hashes": 0,
        "cross_split_exams": 0,
    }


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.clear_output and output_root.exists():
        if not str(output_root).startswith(str(PROJECT_ROOT.resolve())):
            raise RuntimeError(f"Saída insegura para remoção: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = load_rows(args.input_root.resolve())
    unique_rows, duplicate_report = deduplicate(source_rows)
    assigned, targets = assign_grouped_splits(
        unique_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    manifest_rows = build_output(output_root, assigned)
    audit_result = audit(manifest_rows)

    write_csv(output_root / "duplicate_resolution.csv", duplicate_report)
    summary = {
        "input_root": str(args.input_root.resolve()),
        "output_root": str(output_root),
        "seed": args.seed,
        "source_records": len(source_rows),
        "unique_images": len(unique_rows),
        "duplicate_records_removed": len(source_rows) - len(unique_rows),
        "duplicate_hash_groups": len(
            {row["image_sha1"] for row in duplicate_report}
        ),
        "mask_selection": "medoid_by_mean_pairwise_dice",
        "grouping_key": "IM-#### extracted from filename",
        "target_splits": targets,
        "actual_splits": {split: len(rows) for split, rows in assigned.items()},
        **audit_result,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
