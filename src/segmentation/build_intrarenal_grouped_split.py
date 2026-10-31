"""Cria um split intrarrenal agrupado por identificador de exame.

O conjunto intrarrenal original possui imagens do mesmo identificador ``IM-*``
em divisões diferentes. Isso pode produzir vazamento entre treino e teste em
uma avaliação de segmentação. Este script reorganiza imagens, máscaras de
região e máscaras renais mantendo todas as imagens do mesmo exame na mesma
divisão.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1_grouped"
)
EXAM_PATTERN = re.compile(r"(IM-\d+)", re.IGNORECASE)
SPLITS = ("train", "val", "test")


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_exam_id(filename: str) -> str:
    match = EXAM_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Identificador IM-* não encontrado em {filename}")
    return match.group(1).upper()


def load_records(input_root: Path) -> list[dict]:
    records = []
    seen_hashes: dict[str, Path] = {}
    for source_split in SPLITS:
        split_root = input_root / source_split
        image_dir = split_root / "image"
        mask_dir = split_root / "mask"
        kidney_dir = split_root / "kidney_mask"
        if not image_dir.exists() or not mask_dir.exists() or not kidney_dir.exists():
            raise FileNotFoundError(
                f"Estrutura incompleta em {split_root}; são necessários image, mask e kidney_mask."
            )
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            kidney_path = kidney_dir / image_path.name
            if not mask_path.exists() or not kidney_path.exists():
                raise FileNotFoundError(f"Par incompleto para {image_path.name} em {split_root}")
            image_sha1 = file_sha1(image_path)
            previous = seen_hashes.get(image_sha1)
            if previous is not None:
                raise ValueError(f"Imagem duplicada entre o conjunto: {previous} e {image_path}")
            seen_hashes[image_sha1] = image_path
            records.append(
                {
                    "filename": image_path.name,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "kidney_mask_path": kidney_path,
                    "source_split": source_split,
                    "exam_id": extract_exam_id(image_path.name),
                    "image_sha1": image_sha1,
                }
            )
    if not records:
        raise ValueError(f"Nenhuma imagem encontrada em {input_root}")
    return records


def assign_grouped_splits(records: list[dict], seed: int, train_ratio: float, val_ratio: float):
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("As proporções devem ser positivas e somar 1.")

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["exam_id"]].append(record)

    total = len(records)
    targets = {
        "train": round(total * train_ratio),
        "val": round(total * val_ratio),
    }
    targets["test"] = total - targets["train"] - targets["val"]

    rng = random.Random(seed)
    grouped = list(groups.items())
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: len(item[1]), reverse=True)

    assigned = {split: [] for split in SPLITS}
    for exam_id, group in grouped:
        destination = max(
            SPLITS,
            key=lambda split: (
                (targets[split] - len(assigned[split])) / max(targets[split], 1),
                -len(assigned[split]),
            ),
        )
        for record in group:
            assigned[destination].append({**record, "split": destination})
    return assigned, targets


def write_output(output_root: Path, assigned: dict[str, list[dict]], input_root: Path, seed: int, targets: dict):
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for split in SPLITS:
        split_root = output_root / split
        for kind in ("image", "mask", "kidney_mask"):
            (split_root / kind).mkdir(parents=True, exist_ok=True)
        for record in sorted(assigned[split], key=lambda item: item["filename"]):
            shutil.copy2(record["image_path"], split_root / "image" / record["filename"])
            shutil.copy2(record["mask_path"], split_root / "mask" / record["filename"])
            shutil.copy2(record["kidney_mask_path"], split_root / "kidney_mask" / record["filename"])
            manifest_rows.append(
                {
                    "split": split,
                    "exam_id": record["exam_id"],
                    "filename": record["filename"],
                    "source_split": record["source_split"],
                    "image_sha1": record["image_sha1"],
                }
            )

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_root / "split_metadata.txt").write_text(
        "\n".join(
            [
                "split_strategy=grouped_by_exam_id",
                f"source_root={input_root.resolve()}",
                f"seed={seed}",
                f"target_sizes={targets}",
                *(f"{split}_size={len(assigned[split])}" for split in SPLITS),
                *(f"{split}_exam_ids={len({row['exam_id'] for row in assigned[split]})}" for split in SPLITS),
            ]
        ),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.clear_output:
            raise FileExistsError(f"A saída já existe e não está vazia: {output_root}")
        shutil.rmtree(output_root)
    records = load_records(input_root)
    assigned, targets = assign_grouped_splits(records, args.seed, args.train_ratio, args.val_ratio)
    write_output(output_root, assigned, input_root, args.seed, targets)
    print({split: len(rows) for split, rows in assigned.items()})
    print(f"exam_ids={len({row['exam_id'] for row in records})}")
    print(f"output={output_root}")


if __name__ == "__main__":
    main()
