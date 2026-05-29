import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTRARENAL_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal"
DEFAULT_INPUT_ROOT = INTRARENAL_ROOT / "intermediario" / "kidneyus_regions"
DEFAULT_OUTPUT_ROOT = INTRARENAL_ROOT / "supervisionado" / "regions_multiclass_annotator_1"
CLASS_TO_LABEL = {
    "background": 0,
    "cortex": 1,
    "medulla": 2,
    "central_echo_complex": 3,
}
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cria splits por paciente para segmentacao intrarrenal multiclasse "
            "dentro da ROI renal: fundo, cortex, medulla e central echo complex."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--annotator", choices=["annotator_1", "annotator_2"], default="annotator_1")
    parser.add_argument(
        "--required-classes",
        default="cortex,medulla,central_echo_complex",
        help="Classes internas que precisam estar presentes para evitar tratar ausencia de anotacao como fundo.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["copy", "hardlink"], default="hardlink")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def patient_id(filename):
    return filename.split("_IM-", 1)[0]


def grouped_splits(rows, test_ratio, val_ratio, seed):
    groups = {}
    for row in rows:
        groups.setdefault(patient_id(row["filename"]), []).append(row)
    rng = random.Random(seed)
    patient_groups = list(groups.values())
    rng.shuffle(patient_groups)
    targets = {
        "test": round(len(rows) * test_ratio),
        "val": round(len(rows) * val_ratio),
    }
    splits = {"train": [], "val": [], "test": []}
    for split in ("test", "val"):
        while patient_groups and len(splits[split]) < targets[split]:
            splits[split].extend(patient_groups.pop())
    for group in patient_groups:
        splits["train"].extend(group)
    return splits


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Nao foi possivel ler mascara: {path}")
    return mask > 0


def build_label_mask(input_root, annotator, filename):
    roi_root = input_root / "roi" / annotator
    capsule = load_mask(roi_root / "capsule_mask" / filename)
    label = np.zeros(capsule.shape, dtype=np.uint8)
    for class_name in ("cortex", "medulla", "central_echo_complex"):
        class_mask = load_mask(roi_root / f"{class_name}_mask" / filename)
        label[class_mask & capsule] = CLASS_TO_LABEL[class_name]
    label[~capsule] = CLASS_TO_LABEL["background"]
    return label


def save_label(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), label):
        raise RuntimeError(f"Nao foi possivel salvar mascara multiclasse: {path}")


def eligible_rows(manifest, annotator, required_classes):
    required_fields = [f"has_{class_name}" for class_name in required_classes]
    rows = []
    for row in manifest:
        if row["annotator"] != annotator or row["has_capsule"] != "true":
            continue
        if all(row.get(field) == "true" for field in required_fields):
            rows.append(row)
    return rows


def materialize_split(rows, split, args):
    output_rows = []
    roi_root = args.input_root / "roi" / args.annotator
    for row in rows:
        filename = row["filename"]
        image_source = roi_root / "image" / filename
        kidney_source = roi_root / "capsule_mask" / filename
        image_destination = args.output_root / split / "image" / filename
        kidney_destination = args.output_root / split / "kidney_mask" / filename
        mask_destination = args.output_root / split / "mask" / filename
        if not image_source.exists() or not kidney_source.exists():
            raise FileNotFoundError(f"ROI renal incompleta: {filename}")
        materialize_file(image_source, image_destination, args.link_mode)
        materialize_file(kidney_source, kidney_destination, args.link_mode)
        label = build_label_mask(args.input_root, args.annotator, filename)
        save_label(mask_destination, label)
        output_rows.append(
            {
                **row,
                "split": split,
                "patient_id": patient_id(filename),
                "split_image_path": str(image_destination),
                "split_mask_path": str(mask_destination),
                "split_kidney_mask_path": str(kidney_destination),
            }
        )
    write_csv(args.output_root / split / "manifest.csv", output_rows)
    return output_rows


def class_pixel_counts(rows):
    counts = {name: 0 for name in CLASS_TO_LABEL}
    for row in rows:
        label = cv2.imread(row["split_mask_path"], cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue
        for class_name, class_id in CLASS_TO_LABEL.items():
            counts[class_name] += int((label == class_id).sum())
    return counts


def main():
    args = parse_args()
    if args.test_ratio < 0 or args.val_ratio < 0 or args.test_ratio + args.val_ratio >= 1:
        raise ValueError("As proporcoes de val/test devem ser positivas e somar menos de 1.")
    if args.clear_output and args.output_root.exists():
        resolved = args.output_root.resolve()
        expected_parent = (PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal").resolve()
        if expected_parent not in resolved.parents:
            raise ValueError(f"Recusa remover saida fora de dataset_aumentado/dataset_intrarrenal: {resolved}")
        shutil.rmtree(resolved)

    required_classes = [item.strip() for item in args.required_classes.split(",") if item.strip()]
    unknown = sorted(set(required_classes) - (set(CLASS_TO_LABEL) - {"background"}))
    if unknown:
        raise ValueError(f"Classes desconhecidas em --required-classes: {unknown}")

    manifest = read_manifest(args.input_root / "manifest.csv")
    eligible = eligible_rows(manifest, args.annotator, required_classes)
    split_rows = grouped_splits(eligible, args.test_ratio, args.val_ratio, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split in ("train", "val", "test"):
        all_rows.extend(materialize_split(split_rows[split], split, args))
    write_csv(args.output_root / "manifest.csv", all_rows)

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "annotator": args.annotator,
        "required_classes": required_classes,
        "classes": LABEL_TO_CLASS,
        "seed": args.seed,
        "total_eligible_images": len(eligible),
        "splits": {split: len(rows) for split, rows in split_rows.items()},
        "class_pixel_counts": class_pixel_counts(all_rows),
        "note": (
            "Dataset para a etapa 2: DeepLab multiclasse dentro da ROI renal. "
            "Cada paciente aparece em apenas um split. A classe 0 e fundo "
            "dentro do recorte; classes internas ausentes por falta de anotacao "
            "sao excluidas pelo filtro required_classes."
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
