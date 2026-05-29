import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_AUMENTADO = PROJECT_ROOT / "dataset_aumentado"
DEFAULT_DATASET_MANIFEST = DATASET_AUMENTADO / "dataset_geral" / "manifest.csv"
DEFAULT_REGIONS_MANIFEST = (
    DATASET_AUMENTADO / "dataset_intrarrenal" / "intermediario" / "kidneyus_regions" / "manifest.csv"
)
DEFAULT_MEDULLA_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "medulla_predictions_consensus_v1_dataset_geral"
    / "manifest.csv"
)
DEFAULT_OUTPUT = DATASET_AUMENTADO / "curadoria" / "curadoria_mascaras.csv"
FIELDS = ["imagem", "anot1", "anot2", "cl1", "cl2", "fibrose"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera a tabela-base para curadoria humana das mascaras renais e intrarrenais."
    )
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--regions-manifest", type=Path, default=DEFAULT_REGIONS_MANIFEST)
    parser.add_argument("--medulla-predictions", type=Path, default=DEFAULT_MEDULLA_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def existing_path(value):
    if not value:
        return ""
    path = Path(value)
    return str(path.resolve()) if path.exists() else ""


def build_manual_medulla_index(regions_manifest):
    medulla_dir = regions_manifest.parent / "full_masks" / "annotator_1" / "medulla"
    index = {}
    for row in read_rows(regions_manifest):
        if row.get("annotator") != "annotator_1" or row.get("has_medulla", "").lower() != "true":
            continue
        path = medulla_dir / row["filename"]
        if path.exists():
            index[row["filename"]] = str(path.resolve())
    return index


def build_predicted_medulla_index(predictions_manifest):
    index = {}
    for row in read_rows(predictions_manifest):
        path = existing_path(row.get("predicted_medulla_mask_path", ""))
        if path:
            index[row["image_id"]] = path
    return index


def main():
    args = parse_args()
    manual_medulla = build_manual_medulla_index(args.regions_manifest)
    predicted_medulla = build_predicted_medulla_index(args.medulla_predictions)
    dataset_rows = read_rows(args.dataset_manifest)

    rows = []
    for row in dataset_rows:
        image_path = existing_path(row.get("dataset_image_path", ""))
        kidney_mask = existing_path(row.get("dataset_mask_path", "")) if row.get("has_mask", "").lower() == "true" else ""
        filename = Path(row.get("original_image_path", "")).name
        medulla_mask = manual_medulla.get(filename) or predicted_medulla.get(row["image_id"], "")
        rows.append(
            {
                "imagem": image_path,
                "anot1": kidney_mask,
                "anot2": medulla_mask,
                "cl1": "",
                "cl2": "",
                "fibrose": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with_kidney = sum(bool(row["anot1"]) for row in rows)
    with_medulla = sum(bool(row["anot2"]) for row in rows)
    print(f"rows={len(rows)} kidney_masks={with_kidney} intrarenal_masks={with_medulla}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
