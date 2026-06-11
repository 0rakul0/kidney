import argparse
import csv
import json
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTRARENAL_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal"
DEFAULT_INPUT_ROOT = INTRARENAL_ROOT / "intermediario" / "kidneyus_regions"
DEFAULT_OUTPUT_ROOT = INTRARENAL_ROOT / "supervisionado" / "capsule_annotator_1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cria splits supervisionados image/mask para segmentacao externa do rim "
            "usando a anotacao Capsule do kidneyUS."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--annotator", choices=["annotator_1", "annotator_2"], default="annotator_1")
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


def patient_id_from_filename(filename):
    return filename.split("_IM-", 1)[0]


def grouped_splits(rows, test_ratio, val_ratio, rng):
    groups = {}
    for row in rows:
        groups.setdefault(patient_id_from_filename(row["filename"]), []).append(row)

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


def materialize_split(rows, split, args):
    output_rows = []
    for row in rows:
        filename = row["filename"]
        image_source = args.input_root / "images" / filename
        mask_source = args.input_root / "full_masks" / args.annotator / "capsule" / filename
        if not image_source.exists() or not mask_source.exists():
            raise FileNotFoundError(f"Par image/Capsule ausente: {filename}")

        image_destination = args.output_root / split / "image" / filename
        mask_destination = args.output_root / split / "mask" / filename
        materialize_file(image_source, image_destination, args.link_mode)
        materialize_file(mask_source, mask_destination, args.link_mode)
        output_rows.append(
            {
                **row,
                "split": split,
                "split_image_path": str(image_destination),
                "split_mask_path": str(mask_destination),
            }
        )
    write_csv(args.output_root / split / "manifest.csv", output_rows)
    return output_rows


def main():
    args = parse_args()
    if args.test_ratio < 0 or args.val_ratio < 0 or args.test_ratio + args.val_ratio >= 1:
        raise ValueError("As proporcoes de val/test devem somar menos de 1.")
    if args.clear_output and args.output_root.exists():
        resolved = args.output_root.resolve()
        expected_parent = INTRARENAL_ROOT.resolve()
        if expected_parent not in resolved.parents:
            raise ValueError(f"Recusa remover saida fora de dataset_aumentado/dataset_intrarrenal: {resolved}")
        shutil.rmtree(resolved)

    manifest = read_manifest(args.input_root / "manifest.csv")
    eligible = [
        row
        for row in manifest
        if row["annotator"] == args.annotator and row["has_capsule"] == "true"
    ]
    split_rows = grouped_splits(
        eligible,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        rng=random.Random(args.seed),
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split in ("train", "val", "test"):
        all_rows.extend(materialize_split(split_rows[split], split, args))
    write_csv(args.output_root / "manifest.csv", all_rows)

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "task": "kidney_external_segmentation",
        "mask_class": "Capsule",
        "annotator": args.annotator,
        "seed": args.seed,
        "total_eligible_images": len(eligible),
        "splits": {split: len(rows) for split, rows in split_rows.items()},
        "note": (
            "Base canonica para treinar o segmentador externo do rim. "
            "As imagens vem do kidneyUS e a mascara binaria vem da classe Capsule. "
            "Cada paciente aparece em apenas um split para reduzir vazamento."
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
