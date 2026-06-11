import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTRARENAL_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal"
DEFAULT_BASE_ROOT = INTRARENAL_ROOT / "supervisionado" / "medulla_annotator_1"
DEFAULT_SELECTED_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "medulla_consensus_review"
    / "selected_for_review.csv"
)
DEFAULT_OUTPUT_ROOT = INTRARENAL_ROOT / "pseudo_expandido" / "medulla_expanded_consensus_v1"
DEFAULT_REVIEW_ROOT = (
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "medulla_consensus_review"
    / "audit_packet_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cria dataset expandido de Medulla com pseudo-rotulos selecionados "
            "por consenso; preserva validacao e teste manuais."
        )
    )
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--selected-manifest", type=Path, default=DEFAULT_SELECTED_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--min-dice", type=float, default=0.75)
    parser.add_argument("--min-training-dice", type=float, default=0.78)
    parser.add_argument("--min-training-ratio", type=float, default=0.10)
    parser.add_argument("--max-training-components", type=int, default=3)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.80)
    parser.add_argument(
        "--kidney-roi-provenance",
        default="existing_kidney_mask",
        help="Origem da mascara renal usada para recortar a Medulla.",
    )
    parser.add_argument("--pad-ratio", type=float, default=0.12)
    parser.add_argument("--sheet-size", type=int, default=16)
    parser.add_argument("--link-mode", choices=["copy", "hardlink"], default="hardlink")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def load_mask(path, shape=None):
    mask = read_gray(path)
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def md5_file(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def roi_bounds(mask, pad_ratio):
    ys, xs = np.where(mask > 0)
    if not xs.size:
        return None
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    pad = max(4, int(round(max(width, height) * pad_ratio)))
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(mask.shape[1], int(xs.max()) + pad + 1),
        min(mask.shape[0], int(ys.max()) + pad + 1),
    )


def crop(array, bounds):
    x1, y1, x2, y2 = bounds
    return array[y1:y2, x1:x2]


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Nao foi possivel salvar imagem: {path}")


def materialize(source, destination, link_mode):
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


def clear_allowed(path, expected_parent):
    resolved = path.resolve()
    if expected_parent.resolve() not in resolved.parents:
        raise ValueError(f"Recusa remover saida fora de {expected_parent}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def manual_hashes(base_root):
    hashes = {}
    for split in ("train", "val", "test"):
        for row in read_csv(base_root / split / "manifest.csv"):
            image_path = base_root / split / "image" / row["filename"]
            hashes.setdefault(md5_file(image_path), set()).add(split)
    return hashes


def copy_manual_split(split, args):
    output_rows = []
    for row in read_csv(args.base_root / split / "manifest.csv"):
        filename = row["filename"]
        image_dst = args.output_root / split / "image" / filename
        mask_dst = args.output_root / split / "mask" / filename
        kidney_dst = args.output_root / split / "kidney_mask" / filename
        materialize(args.base_root / split / "image" / filename, image_dst, args.link_mode)
        materialize(args.base_root / split / "mask" / filename, mask_dst, args.link_mode)
        materialize(args.base_root / split / "kidney_mask" / filename, kidney_dst, args.link_mode)
        output_rows.append(
            {
                **row,
                "label_provenance": "manual_annotator_1",
                "review_decision": "manual_reference",
                "split_image_path": str(image_dst),
                "split_mask_path": str(mask_dst),
                "split_kidney_mask_path": str(kidney_dst),
            }
        )
    return output_rows


def mask_metrics(deeplab, roi_unet, kidney):
    intersection = np.logical_and(deeplab > 0, roi_unet > 0)
    union = np.logical_or(deeplab > 0, roi_unet > 0)
    components, _, stats, _ = cv2.connectedComponentsWithStats(deeplab.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.array([], dtype=np.int32)
    total = int(deeplab.sum())
    return {
        "positive_pixels": total,
        "kidney_pixels": int(kidney.sum()),
        "positive_to_kidney_ratio": float(total / max(int(kidney.sum()), 1)),
        "components": max(0, int(components - 1)),
        "largest_component_fraction": float(areas.max() / max(total, 1)) if areas.size else 0.0,
        "agreement_pixels": int(intersection.sum()),
        "disagreement_pixels": int(union.sum() - intersection.sum()),
    }


def build_pseudo_row(row, manual_image_hashes, args):
    image_path = Path(row["dataset_image_path"])
    kidney_path = Path(row["dataset_kidney_mask_path"])
    deeplab_path = Path(row["deeplab_mask_path"])
    roi_unet_path = Path(row["roi_unet_mask_path"])
    digest = md5_file(image_path)
    overlap = sorted(manual_image_hashes.get(digest, set()))
    if overlap:
        return None, {
            "image_id": row["image_id"],
            "review_decision": "excluded_manual_split_overlap",
            "manual_overlap_splits": ",".join(overlap),
        }
    image = read_gray(image_path)
    kidney = load_mask(kidney_path, image.shape)
    deeplab = load_mask(deeplab_path, image.shape)
    roi_unet = load_mask(roi_unet_path, image.shape)
    bounds = roi_bounds(kidney, args.pad_ratio)
    if bounds is None:
        return None, {
            "image_id": row["image_id"],
            "review_decision": "excluded_empty_kidney_roi",
            "manual_overlap_splits": "",
        }
    info = mask_metrics(deeplab, roi_unet, kidney)
    flags = []
    if float(row["model_dice"]) < args.min_training_dice:
        flags.append("consensus_dice_below_training_threshold")
    if info["positive_to_kidney_ratio"] < args.min_training_ratio:
        flags.append("small_medulla_ratio")
    if info["components"] > args.max_training_components:
        flags.append("fragmented_prediction")
    if info["largest_component_fraction"] < args.min_largest_component_fraction:
        flags.append("low_largest_component_fraction")
    included_in_train = not flags
    filename = f"pseudo_{row['image_id']}.png"
    image_dst = args.output_root / "train" / "image" / filename
    mask_dst = args.output_root / "train" / "mask" / filename
    kidney_dst = args.output_root / "train" / "kidney_mask" / filename
    if included_in_train:
        save_image(image_dst, crop(image, bounds))
        save_image(mask_dst, crop(deeplab * 255, bounds))
        save_image(kidney_dst, crop(kidney * 255, bounds))
    pseudo_row = {
        "filename": filename,
        "annotator": "pseudo_consensus",
        "source_image_path": str(image_path),
        "roi_image_path": str(image_dst),
        "has_capsule": "true",
        "has_cortex": "false",
        "has_medulla": "true",
        "has_central_echo_complex": "false",
        "eligible_medulla_training": "true",
        "capsule_pixels": info["kidney_pixels"],
        "medulla_pixels": info["positive_pixels"],
        "roi_bounds": ",".join(str(value) for value in bounds),
        "split": "train",
        "split_image_path": str(image_dst),
        "split_mask_path": str(mask_dst),
        "split_kidney_mask_path": str(kidney_dst),
        "label_provenance": "pseudo_consensus_deeplab_target",
        "kidney_roi_provenance": args.kidney_roi_provenance,
        "review_decision": "accepted_by_strict_consensus_pending_clinical_audit",
        "consensus_dice": f"{float(row['model_dice']):.6f}",
        "consensus_iou": f"{float(row['model_iou']):.6f}",
    }
    audit_row = {
        "image_id": row["image_id"],
        "source_name": row["source_name"],
        "source_image_path": str(image_path),
        "kidney_mask_path": str(kidney_path),
        "training_target_mask_path": str(deeplab_path),
        "control_mask_path": str(roi_unet_path),
        "kidney_roi_provenance": args.kidney_roi_provenance,
        "model_dice": f"{float(row['model_dice']):.6f}",
        "model_iou": f"{float(row['model_iou']):.6f}",
        "medulla_to_kidney_ratio": f"{info['positive_to_kidney_ratio']:.6f}",
        "components": info["components"],
        "largest_component_fraction": f"{info['largest_component_fraction']:.6f}",
        "quality_flags": ";".join(flags),
        "review_decision": (
            "pending_clinical_audit_priority"
            if flags
            else "pending_clinical_audit_strict_consensus"
        ),
        "reviewer": "",
        "review_notes": "",
        "included_in_pseudo_train": str(included_in_train).lower(),
    }
    return (pseudo_row if included_in_train else None), audit_row


def overlay(base, mask, color):
    colored = base.copy()
    colored[mask > 0] = color
    return cv2.addWeighted(base, 0.70, colored, 0.30, 0)


def audit_tile(audit_row, args):
    image = read_gray(Path(audit_row["source_image_path"]))
    kidney = load_mask(Path(audit_row["kidney_mask_path"]), image.shape)
    deeplab = load_mask(Path(audit_row["training_target_mask_path"]), image.shape)
    roi_unet = load_mask(Path(audit_row["control_mask_path"]), image.shape)
    bounds = roi_bounds(kidney, args.pad_ratio)
    base = cv2.cvtColor(crop(image, bounds), cv2.COLOR_GRAY2BGR)
    deep = overlay(base, crop(deeplab, bounds), (0, 0, 255))
    control = overlay(base, crop(roi_unet, bounds), (255, 0, 255))
    agreement = overlay(base, crop((deeplab & roi_unet), bounds), (0, 255, 0))
    tiles = [
        cv2.resize(tile, (160, 126), interpolation=cv2.INTER_AREA)
        for tile in (base, deep, control, agreement)
    ]
    panel = cv2.hconcat(tiles)
    title = (
        f"{audit_row['image_id'][:44]} | D={float(audit_row['model_dice']):.3f} "
        f"| R={float(audit_row['medulla_to_kidney_ratio']):.3f}"
    )
    cv2.putText(panel, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def write_audit_sheets(audit_rows, args):
    sheet_dir = args.review_root / "sheets"
    individual_dir = args.review_root / "individual"
    for index, row in enumerate(audit_rows, 1):
        panel = audit_tile(row, args)
        save_image(individual_dir / f"{index:04d}_{row['image_id']}.png", panel)
    for offset in range(0, len(audit_rows), args.sheet_size):
        batch = audit_rows[offset: offset + args.sheet_size]
        panels = [audit_tile(row, args) for row in batch]
        blank = np.zeros_like(panels[0])
        while len(panels) < args.sheet_size:
            panels.append(blank)
        rows = [cv2.hconcat(panels[i:i + 2]) for i in range(0, args.sheet_size, 2)]
        sheet = cv2.vconcat(rows)
        page = offset // args.sheet_size + 1
        save_image(sheet_dir / f"review_sheet_{page:03d}.png", sheet)


def main():
    args = parse_args()
    if args.clear_output:
        clear_allowed(args.output_root, PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal")
        clear_allowed(args.review_root, PROJECT_ROOT / "results")

    selected = [
        row
        for row in read_csv(args.selected_manifest)
        if float(row["model_dice"]) >= args.min_dice
    ]
    manual_image_hashes = manual_hashes(args.base_root)
    train_rows = copy_manual_split("train", args)
    val_rows = copy_manual_split("val", args)
    test_rows = copy_manual_split("test", args)
    audit_rows = []
    excluded_rows = []
    audit_only_rows = []
    pseudo_rows = []
    for row in selected:
        pseudo_row, audit_row = build_pseudo_row(row, manual_image_hashes, args)
        if audit_row["review_decision"] == "excluded_manual_split_overlap" or audit_row["review_decision"] == "excluded_empty_kidney_roi":
            excluded_rows.append(audit_row)
        elif pseudo_row is None:
            audit_only_rows.append(audit_row)
            audit_rows.append(audit_row)
        else:
            pseudo_rows.append(pseudo_row)
            audit_rows.append(audit_row)
    train_rows.extend(pseudo_rows)
    write_csv(args.output_root / "train" / "manifest.csv", train_rows)
    write_csv(args.output_root / "val" / "manifest.csv", val_rows)
    write_csv(args.output_root / "test" / "manifest.csv", test_rows)
    write_csv(args.output_root / "manifest.csv", train_rows + val_rows + test_rows)
    write_csv(args.review_root / "review_queue.csv", audit_rows + excluded_rows)
    write_audit_sheets(audit_rows, args)

    source_counts = {}
    for row in audit_rows:
        if row["included_in_pseudo_train"] == "true":
            source_counts[row["source_name"]] = source_counts.get(row["source_name"], 0) + 1
    dice_values = np.array(
        [float(row["model_dice"]) for row in audit_rows if row["included_in_pseudo_train"] == "true"],
        dtype=np.float32,
    )
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_dataset": str(args.base_root),
        "selected_manifest": str(args.selected_manifest),
        "selection_rule": f"model_dice >= {args.min_dice:.2f}",
        "automatic_training_rule": {
            "model_dice_at_least": args.min_training_dice,
            "medulla_to_kidney_ratio_at_least": args.min_training_ratio,
            "components_at_most": args.max_training_components,
            "largest_component_fraction_at_least": args.min_largest_component_fraction,
        },
        "training_target_rule": "DeepLab prediction, gated by agreement with MedullaROIUNet",
        "kidney_roi_provenance": args.kidney_roi_provenance,
        "manual_splits_unchanged": {
            "train": len(train_rows) - len(pseudo_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "pseudo_consensus_candidates_received": len(selected),
        "pseudo_added_to_train": len(pseudo_rows),
        "pseudo_flagged_for_priority_clinical_audit": len(audit_only_rows),
        "pseudo_excluded_for_manual_overlap_or_invalid_roi": len(excluded_rows),
        "expanded_train_total": len(train_rows),
        "pseudo_sources_added_to_train": source_counts,
        "consensus_dice_added": {
            "mean": round(float(dice_values.mean()), 6) if dice_values.size else None,
            "minimum": round(float(dice_values.min()), 6) if dice_values.size else None,
            "maximum": round(float(dice_values.max()), 6) if dice_values.size else None,
        },
        "review_packet": {
            "queue": str(args.review_root / "review_queue.csv"),
            "individual_panels": len(audit_rows),
            "sheets": int(np.ceil(len(audit_rows) / args.sheet_size)) if audit_rows else 0,
        },
        "output_root": str(args.output_root),
        "provenance_warning": (
            "As novas mascaras sao pseudo-rotulos filtrados por consenso entre modelos "
            f"para experimento de expansao; a ROI renal foi marcada como {args.kidney_roi_provenance}. "
            "Todas permanecem pending_clinical_audit "
            "ate revisao humana; casos sinalizados nao entram no treino automatico."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.review_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
