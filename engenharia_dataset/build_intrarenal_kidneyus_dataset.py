import argparse
import ast
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "dataset_aumentado" / "fontes" / "kidneyUS_images_25_june_2025"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal" / "intermediario" / "kidneyus_regions"
ANNOTATOR_FILES = {
    "annotator_1": "reviewed_labels_1.csv",
    "annotator_2": "reviewed_labels_2.csv",
}
ANATOMY_TO_SLUG = {
    "Capsule": "capsule",
    "Cortex": "cortex",
    "Medulla": "medulla",
    "Central Echo Complex": "central_echo_complex",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Converte anotacoes poligonais kidneyUS em mascaras supervisionadas "
            "e ROIs renais para segmentacao intrarrenal."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pad-ratio", type=float, default=0.12)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def parse_json_value(value):
    value = str(value or "").strip()
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def read_annotations(source_dir, file_name):
    records = defaultdict(list)
    path = source_dir / file_name
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            anatomy = parse_json_value(row.get("region_attributes", "")).get("Anatomy", "")
            if anatomy not in ANATOMY_TO_SLUG:
                continue
            shape = parse_json_value(row.get("region_shape_attributes", ""))
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if shape.get("name") != "polygon" or len(xs) < 3 or len(xs) != len(ys):
                continue
            records[row["filename"]].append((anatomy, xs, ys))
    return records


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def build_masks(image_shape, regions):
    masks = {
        slug: np.zeros(image_shape, dtype=np.uint8)
        for slug in ANATOMY_TO_SLUG.values()
    }
    for anatomy, xs, ys in regions:
        points = np.array(list(zip(xs, ys)), dtype=np.int32)
        cv2.fillPoly(masks[ANATOMY_TO_SLUG[anatomy]], [points], 255)
    return masks


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Nao foi possivel salvar imagem: {path}")


def roi_bounds(capsule_mask, pad_ratio):
    ys, xs = np.where(capsule_mask > 0)
    if xs.size == 0:
        return None
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    pad = max(4, int(round(max(width, height) * pad_ratio)))
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(capsule_mask.shape[1], int(xs.max()) + pad + 1)
    y2 = min(capsule_mask.shape[0], int(ys.max()) + pad + 1)
    return x1, y1, x2, y2


def crop(image, bounds):
    x1, y1, x2, y2 = bounds
    return image[y1:y2, x1:x2]


def mask_dice(left, right):
    left = left > 0
    right = right > 0
    denominator = int(left.sum() + right.sum())
    if denominator == 0:
        return None
    return float(2 * np.logical_and(left, right).sum() / denominator)


def make_preview(image, masks, bounds, label):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    full_overlay = image_rgb.copy()
    full_overlay[masks["capsule"] > 0] = (0, 180, 255)
    full_overlay[masks["medulla"] > 0] = (0, 0, 255)
    full_overlay = cv2.addWeighted(image_rgb, 0.70, full_overlay, 0.30, 0)

    roi_image = crop(image_rgb, bounds)
    roi_overlay = roi_image.copy()
    roi_medulla = crop(masks["medulla"], bounds)
    roi_overlay[roi_medulla > 0] = (0, 0, 255)
    roi_overlay = cv2.addWeighted(roi_image, 0.65, roi_overlay, 0.35, 0)
    roi_mask = cv2.cvtColor(roi_medulla, cv2.COLOR_GRAY2BGR)

    tiles = []
    for tile in (full_overlay, roi_overlay, roi_mask):
        tiles.append(cv2.resize(tile, (280, 220), interpolation=cv2.INTER_AREA))
    panel = cv2.hconcat(tiles)
    cv2.putText(
        panel,
        label[:100],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(output_root, clear_output):
    if clear_output and output_root.exists():
        resolved = output_root.resolve()
        intended_parent = (PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal").resolve()
        if intended_parent not in resolved.parents:
            raise ValueError(f"Recusa remover saida fora de dataset_aumentado/dataset_intrarrenal: {resolved}")
        shutil.rmtree(resolved)
    output_root.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    prepare_output(args.output_root, args.clear_output)
    annotations = {
        annotator: read_annotations(args.source_dir, csv_name)
        for annotator, csv_name in ANNOTATOR_FILES.items()
    }
    file_names = sorted(path.name for path in args.source_dir.glob("*.png"))
    manifest_rows = []
    masks_by_annotator = defaultdict(dict)
    preview_written = defaultdict(int)

    for file_name in file_names:
        source_path = args.source_dir / file_name
        if not source_path.exists():
            continue
        image = read_image(source_path)
        save_image(args.output_root / "images" / file_name, image)

        for annotator, records in annotations.items():
            masks = build_masks(image.shape, records.get(file_name, []))
            masks_by_annotator[annotator][file_name] = masks
            for slug, mask in masks.items():
                save_image(args.output_root / "full_masks" / annotator / slug / file_name, mask)

            bounds = roi_bounds(masks["capsule"], args.pad_ratio)
            has_capsule = bool(masks["capsule"].any())
            has_medulla = bool(masks["medulla"].any())
            eligible = has_capsule and has_medulla
            roi_path = ""
            if bounds is not None:
                masked = image.copy()
                masked[masks["capsule"] == 0] = 0
                save_image(args.output_root / "roi" / annotator / "image" / file_name, crop(image, bounds))
                save_image(args.output_root / "roi" / annotator / "masked_image" / file_name, crop(masked, bounds))
                for slug, mask in masks.items():
                    save_image(args.output_root / "roi" / annotator / f"{slug}_mask" / file_name, crop(mask, bounds))
                if has_medulla:
                    medulla_image = image.copy()
                    medulla_image[masks["medulla"] == 0] = 0
                    save_image(
                        args.output_root / "roi" / annotator / "medulla_image" / file_name,
                        crop(medulla_image, bounds),
                    )
                roi_path = str(args.output_root / "roi" / annotator / "image" / file_name)

                if eligible and preview_written[annotator] < args.preview_count:
                    preview_written[annotator] += 1
                    panel = make_preview(image, masks, bounds, f"{annotator} | {file_name} | vermelho=medulla")
                    save_image(
                        args.output_root / "previews" / f"{annotator}_{preview_written[annotator]:02d}_{file_name}",
                        panel,
                    )

            manifest_rows.append(
                {
                    "filename": file_name,
                    "annotator": annotator,
                    "source_image_path": str(source_path),
                    "roi_image_path": roi_path,
                    "has_capsule": str(has_capsule).lower(),
                    "has_cortex": str(bool(masks["cortex"].any())).lower(),
                    "has_medulla": str(has_medulla).lower(),
                    "has_central_echo_complex": str(bool(masks["central_echo_complex"].any())).lower(),
                    "eligible_medulla_training": str(eligible).lower(),
                    "capsule_pixels": int((masks["capsule"] > 0).sum()),
                    "medulla_pixels": int((masks["medulla"] > 0).sum()),
                    "roi_bounds": "" if bounds is None else ",".join(str(value) for value in bounds),
                }
            )

    agreement_rows = []
    for file_name in file_names:
        first = masks_by_annotator["annotator_1"].get(file_name)
        second = masks_by_annotator["annotator_2"].get(file_name)
        if first is None or second is None:
            continue
        for slug in ANATOMY_TO_SLUG.values():
            dice = mask_dice(first[slug], second[slug])
            if dice is None:
                continue
            agreement_rows.append(
                {
                    "filename": file_name,
                    "anatomy": slug,
                    "annotator_1_pixels": int((first[slug] > 0).sum()),
                    "annotator_2_pixels": int((second[slug] > 0).sum()),
                    "both_annotators_labeled": str(
                        bool(first[slug].any()) and bool(second[slug].any())
                    ).lower(),
                    "dice": f"{dice:.6f}",
                }
            )

    write_csv(args.output_root / "manifest.csv", manifest_rows)
    write_csv(args.output_root / "interannotator_agreement.csv", agreement_rows)

    summary = {
        "source_dir": str(args.source_dir),
        "output_root": str(args.output_root),
        "source_png_images": len(file_names),
        "images_with_supported_annotations": sum(
            any(records.get(file_name, []) for records in annotations.values())
            for file_name in file_names
        ),
        "images_without_supported_annotations": sum(
            not any(records.get(file_name, []) for records in annotations.values())
            for file_name in file_names
        ),
        "records_by_annotator": {},
        "agreement": {},
        "recommended_target": "medulla",
        "note": (
            "Medulla e o alvo supervisionado inicial para piramides renais. "
            "As ROIs deste dataset usam a capsula manual; experimentos em cascata "
            "devem substituir a ROI pela mascara prevista pelo modelo 2."
        ),
    }
    for annotator in ANNOTATOR_FILES:
        subset = [row for row in manifest_rows if row["annotator"] == annotator]
        summary["records_by_annotator"][annotator] = {
            "images": len(subset),
            "with_capsule": sum(row["has_capsule"] == "true" for row in subset),
            "with_medulla": sum(row["has_medulla"] == "true" for row in subset),
            "eligible_medulla_training": sum(row["eligible_medulla_training"] == "true" for row in subset),
        }
    for slug in ANATOMY_TO_SLUG.values():
        values = [float(row["dice"]) for row in agreement_rows if row["anatomy"] == slug]
        both_values = [
            float(row["dice"])
            for row in agreement_rows
            if row["anatomy"] == slug and row["both_annotators_labeled"] == "true"
        ]
        summary["agreement"][slug] = {
            "images_labeled_by_either_annotator": len(values),
            "images_labeled_by_both_annotators": len(both_values),
            "images_labeled_by_only_one_annotator": len(values) - len(both_values),
            "mean_dice_including_presence_disagreement": (
                round(float(np.mean(values)), 6) if values else None
            ),
            "mean_dice_when_both_labeled": (
                round(float(np.mean(both_values)), 6) if both_values else None
            ),
        }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
