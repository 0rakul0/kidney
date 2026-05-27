import argparse
import csv
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_loader import load_model_bundle


DEFAULT_MODEL = "segformer"
DEFAULT_CHECKPOINT = "models/segformer_b2_capacity_8ep.pth"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera pseudo-mascaras a partir de dataset_aumentado/fontes/dataset_loader/ e monta "
            "um dataset aumentado em uma pasta separada."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Gera pseudo-mascaras filtradas a partir das imagens do dataset_loader.",
    )
    add_shared_generation_args(generate)
    generate.add_argument("--loader-dir", type=str, default="dataset_aumentado/fontes/dataset_loader")
    generate.add_argument("--dataset-root", type=str, default="dataset_inicial")
    generate.add_argument("--output-root", type=str, default="dataset_aumentado/pseudo_labels")
    generate.add_argument("--include-existing", action="store_true")
    generate.add_argument("--img-size", type=int, default=256)
    generate.add_argument("--min-confidence", type=float, default=0.88)
    generate.add_argument("--min-foreground-pixels", type=int, default=800)
    generate.add_argument("--min-area-ratio", type=float, default=0.03)
    generate.add_argument("--max-area-ratio", type=float, default=0.75)
    generate.add_argument("--morph-kernel", type=int, default=5)

    build = subparsers.add_parser(
        "build-augmented",
        help=(
            "Cria um dataset aumentado com o dataset atual + pseudo-mascaras "
            "aceitas, em uma nova pasta."
        ),
    )
    build.add_argument("--dataset-root", type=str, default="dataset_inicial")
    build.add_argument("--pseudo-root", type=str, default="dataset_aumentado/pseudo_labels")
    build.add_argument("--output-root", type=str, default="dataset_aumentado/expansao_pseudorrotulada")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--clear-output", action="store_true")

    full = subparsers.add_parser(
        "full",
        help="Executa generate e build-augmented em sequencia.",
    )
    add_shared_generation_args(full)
    full.add_argument("--loader-dir", type=str, default="dataset_aumentado/fontes/dataset_loader")
    full.add_argument("--dataset-root", type=str, default="dataset_inicial")
    full.add_argument("--pseudo-root", type=str, default="dataset_aumentado/pseudo_labels")
    full.add_argument("--output-root", type=str, default="dataset_aumentado/expansao_pseudorrotulada")
    full.add_argument("--include-existing", action="store_true")
    full.add_argument("--img-size", type=int, default=256)
    full.add_argument("--min-confidence", type=float, default=0.88)
    full.add_argument("--min-foreground-pixels", type=int, default=800)
    full.add_argument("--min-area-ratio", type=float, default=0.03)
    full.add_argument("--max-area-ratio", type=float, default=0.75)
    full.add_argument("--morph-kernel", type=int, default=5)
    full.add_argument("--seed", type=int, default=42)
    full.add_argument("--clear-output", action="store_true")

    return parser.parse_args()


def add_shared_generation_args(parser):
    parser.add_argument(
        "--model",
        choices=["unet", "unetplusplus", "deeplab", "segformer"],
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101"],
        help="Backbone para checkpoint DeepLab sem metadata.",
    )
    parser.add_argument(
        "--segformer-backbone",
        type=str,
        help="Backbone Hugging Face para checkpoint SegFormer sem metadata.",
    )


def list_existing_dataset_names(dataset_root):
    dataset_root = Path(dataset_root)
    names = set()
    counts = {}

    for split in SPLITS:
        image_dir = dataset_root / split / "image"
        files = sorted(path.name for path in image_dir.glob("*") if path.is_file())
        counts[split] = len(files)
        names.update(files)

    return names, counts


def prepare_image_tensor(image, img_size):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    stacked = np.stack([normalized, normalized, normalized], axis=0)
    tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    return resized, tensor


def predict_probability(bundle, tensor, img_size):
    model = bundle["model"]
    display_name = bundle["display_name"]

    with torch.no_grad():
        if display_name == "SegFormer":
            logits = model(pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
        elif display_name == "DeepLab":
            logits = model(tensor)["out"]
        else:
            logits = model(tensor)

    return torch.sigmoid(logits).cpu().numpy()[0, 0]


def postprocess_mask(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def ensure_structure(root):
    root = Path(root)
    accepted_image_dir = root / "accepted" / "image"
    accepted_mask_dir = root / "accepted" / "mask"
    rejected_dir = root / "rejected"
    root.mkdir(parents=True, exist_ok=True)
    accepted_image_dir.mkdir(parents=True, exist_ok=True)
    accepted_mask_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    return accepted_image_dir, accepted_mask_dir, rejected_dir


def write_report(report_path, rows):
    fieldnames = [
        "file_name",
        "status",
        "mean_confidence",
        "foreground_pixels",
        "area_ratio",
        "threshold",
        "checkpoint",
        "reason",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_dir = Path(args.loader_dir)
    dataset_root = Path(args.dataset_root)
    pseudo_root = getattr(args, "pseudo_root", None)
    output_root = Path(pseudo_root if pseudo_root is not None else args.output_root)

    accepted_image_dir, accepted_mask_dir, rejected_dir = ensure_structure(output_root)

    existing_names, split_counts = list_existing_dataset_names(dataset_root)

    bundle = load_model_bundle(
        args.model,
        device=device,
        checkpoint_path=args.checkpoint,
        deeplab_backbone=args.backbone,
        segformer_backbone=args.segformer_backbone,
    )
    threshold = bundle["threshold"]

    candidate_paths = sorted(
        path
        for path in loader_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not args.include_existing:
        candidate_paths = [path for path in candidate_paths if path.name not in existing_names]

    report_rows = []
    accepted_count = 0
    rejected_count = 0

    for image_path in candidate_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected_count += 1
            report_rows.append(
                build_report_row(
                    image_path.name,
                    "rejected",
                    threshold,
                    bundle["checkpoint_path"],
                    reason="unable_to_read",
                )
            )
            continue

        resized, tensor = prepare_image_tensor(image, args.img_size)
        tensor = tensor.to(device)
        probability = predict_probability(bundle, tensor, args.img_size)
        mask = (probability >= threshold).astype(np.uint8)
        mask = postprocess_mask(mask, args.morph_kernel)

        foreground_pixels = int(mask.sum())
        area_ratio = float(foreground_pixels / mask.size)
        mean_confidence = float(probability[mask == 1].mean()) if foreground_pixels > 0 else 0.0

        reason = None
        if foreground_pixels == 0:
            reason = "empty_mask"
        elif foreground_pixels < args.min_foreground_pixels:
            reason = "too_small"
        elif area_ratio < args.min_area_ratio:
            reason = "area_ratio_too_small"
        elif area_ratio > args.max_area_ratio:
            reason = "area_ratio_too_large"
        elif mean_confidence < args.min_confidence:
            reason = "low_confidence"

        if reason is None:
            cv2.imwrite(str(accepted_image_dir / image_path.name), resized)
            cv2.imwrite(str(accepted_mask_dir / image_path.name), (mask * 255).astype(np.uint8))
            accepted_count += 1
            status = "accepted"
        else:
            cv2.imwrite(str(rejected_dir / image_path.name), resized)
            rejected_count += 1
            status = "rejected"

        report_rows.append(
            build_report_row(
                image_path.name,
                status,
                threshold,
                bundle["checkpoint_path"],
                mean_confidence=mean_confidence,
                foreground_pixels=foreground_pixels,
                area_ratio=area_ratio,
                reason=reason or "accepted",
            )
        )

    report_path = output_root / "pseudo_label_report.csv"
    write_report(report_path, report_rows)

    print(f"Modelo usado: {bundle['display_name']}")
    print(f"Checkpoint: {bundle['checkpoint_path']}")
    print(f"Threshold de mascara: {threshold:.2f}")
    print(f"Imagens ja existentes no dataset: {len(existing_names)}")
    print(f"Candidatas processadas: {len(candidate_paths)}")
    print(f"Aceitas: {accepted_count}")
    print(f"Rejeitadas: {rejected_count}")
    print(f"Split atual: {split_counts}")
    print(f"Relatorio salvo em: {report_path}")


def build_report_row(
    file_name,
    status,
    threshold,
    checkpoint_path,
    mean_confidence=0.0,
    foreground_pixels=0,
    area_ratio=0.0,
    reason="",
):
    return {
        "file_name": file_name,
        "status": status,
        "mean_confidence": round(float(mean_confidence), 6),
        "foreground_pixels": int(foreground_pixels),
        "area_ratio": round(float(area_ratio), 6),
        "threshold": round(float(threshold), 4),
        "checkpoint": str(checkpoint_path),
        "reason": reason,
    }


def prepare_output_split_dirs(output_root):
    for split in SPLITS:
        (output_root / split / "image").mkdir(parents=True, exist_ok=True)
        (output_root / split / "mask").mkdir(parents=True, exist_ok=True)


def copy_current_dataset(dataset_root, output_root):
    for split in SPLITS:
        for kind in ("image", "mask"):
            source_dir = dataset_root / split / kind
            target_dir = output_root / split / kind
            for source_path in source_dir.glob("*"):
                if source_path.is_file():
                    shutil.copy2(source_path, target_dir / source_path.name)


def split_counts_with_largest_remainder(total_new, base_counts):
    base_total = sum(base_counts.values())
    raw = {
        split: (base_counts[split] / base_total) * total_new
        for split in SPLITS
    }
    assigned = {split: math.floor(value) for split, value in raw.items()}
    remainder = total_new - sum(assigned.values())

    order = sorted(
        SPLITS,
        key=lambda split: (raw[split] - assigned[split], base_counts[split]),
        reverse=True,
    )

    for split in order[:remainder]:
        assigned[split] += 1

    return assigned


def run_build_augmented(dataset_root, pseudo_root, output_root, seed, clear_output=False):
    dataset_root = Path(dataset_root)
    pseudo_root = Path(pseudo_root)
    output_root = Path(output_root)

    accepted_image_dir = pseudo_root / "accepted" / "image"
    accepted_mask_dir = pseudo_root / "accepted" / "mask"
    if not accepted_image_dir.exists() or not accepted_mask_dir.exists():
        raise FileNotFoundError(
            "Pseudo-mascaras aceitas nao encontradas. Execute o comando generate primeiro."
        )

    existing_names, base_counts = list_existing_dataset_names(dataset_root)
    pseudo_names = sorted(
        path.name for path in accepted_image_dir.glob("*") if path.is_file()
    )
    new_names = [name for name in pseudo_names if name not in existing_names]

    if clear_output and output_root.exists():
        shutil.rmtree(output_root)

    prepare_output_split_dirs(output_root)
    copy_current_dataset(dataset_root, output_root)

    random.Random(seed).shuffle(new_names)
    allocation = split_counts_with_largest_remainder(len(new_names), base_counts)

    offset = 0
    manifest_rows = []

    for split in SPLITS:
        count = allocation[split]
        split_names = new_names[offset:offset + count]
        offset += count

        for name in split_names:
            shutil.copy2(accepted_image_dir / name, output_root / split / "image" / name)
            shutil.copy2(accepted_mask_dir / name, output_root / split / "mask" / name)
            manifest_rows.append({"split": split, "file_name": name, "source": "pseudo_label"})

    manifest_path = output_root / "pseudo_label_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["split", "file_name", "source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Dataset base copiado para: {output_root}")
    print(f"Pseudo-rotulos novos incorporados: {len(new_names)}")
    print(f"Distribuicao adicionada: {allocation}")
    print(f"Manifesto salvo em: {manifest_path}")


def main():
    args = parse_args()

    if args.command == "generate":
        run_generate(args)
        return

    if args.command == "build-augmented":
        run_build_augmented(
            dataset_root=args.dataset_root,
            pseudo_root=args.pseudo_root,
            output_root=args.output_root,
            seed=args.seed,
            clear_output=args.clear_output,
        )
        return

    if args.command == "full":
        run_generate(args)
        run_build_augmented(
            dataset_root=args.dataset_root,
            pseudo_root=Path(args.pseudo_root),
            output_root=Path(args.output_root),
            seed=args.seed,
            clear_output=args.clear_output,
        )


if __name__ == "__main__":
    main()
