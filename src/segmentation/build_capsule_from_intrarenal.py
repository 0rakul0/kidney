"""Materializa um conjunto binário de cápsula a partir das kidney_mask.

Este conjunto é usado somente para validar a cascata no mesmo domínio das
estruturas intrarrenais. A máscara ``kidney_mask`` passa a ser a máscara
binária ``mask`` esperada pelos treinadores de U-Net e DeepLabV3.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1_grouped"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_from_regions_grouped"
)
SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clear-output", action="store_true")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.clear_output:
            raise FileExistsError(f"A saída já existe e não está vazia: {output_root}")
        shutil.rmtree(output_root)

    rows = []
    for split in SPLITS:
        source = input_root / split
        image_dir = source / "image"
        kidney_dir = source / "kidney_mask"
        if not image_dir.exists() or not kidney_dir.exists():
            raise FileNotFoundError(f"Estrutura incompleta em {source}")
        destination = output_root / split
        (destination / "image").mkdir(parents=True, exist_ok=True)
        (destination / "mask").mkdir(parents=True, exist_ok=True)
        for image_path in sorted(image_dir.glob("*.png")):
            kidney_path = kidney_dir / image_path.name
            if not kidney_path.exists():
                raise FileNotFoundError(f"kidney_mask ausente para {image_path.name}")
            shutil.copy2(image_path, destination / "image" / image_path.name)
            shutil.copy2(kidney_path, destination / "mask" / image_path.name)
            rows.append({"split": split, "filename": image_path.name, "source_kidney_mask": str(kidney_path)})

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "source_metadata.txt").write_text(
        "\n".join(
            [
                "task=binary_kidney_capsule",
                f"source_root={input_root}",
                "mask_source=kidney_mask_from_intrarenal_dataset",
                *(f"{split}_size={sum(row['split'] == split for row in rows)}" for split in SPLITS),
            ]
        ),
        encoding="utf-8",
    )
    print({split: sum(row["split"] == split for row in rows) for split in SPLITS})
    print(f"output={output_root}")


if __name__ == "__main__":
    main()
