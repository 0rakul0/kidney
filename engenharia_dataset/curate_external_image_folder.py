import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Curadoria generica de uma pasta externa de imagens: copia apenas "
            "ultrassom B-mode/escala de cinza para dataset_aumentado/fontes/external_data/processed."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_aumentado") / "fontes" / "external_data" / "processed",
    )
    parser.add_argument("--min-width", type=int, default=128)
    parser.add_argument("--min-height", type=int, default=128)
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Converte RGB para cinza. Por padrao RGB/colorido e rejeitado.",
    )
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def safe_name(value):
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def iter_images(input_dir):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def is_nearly_grayscale(image, tolerance=3.0):
    if image.ndim != 3 or image.shape[2] < 3:
        return True
    b, g, r = cv2.split(image[:, :, :3])
    diff = (
        np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
        + np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
        + np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32)))
    ) / 3.0
    return diff <= tolerance


def load_as_bmode(path, include_rgb, min_width, min_height):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, "unreadable"

    if raw.ndim == 2:
        gray = raw
    elif raw.ndim == 3:
        if not include_rgb and not is_nearly_grayscale(raw):
            return None, "rgb_or_color_skipped"
        gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        return None, f"unsupported_dimensions:{raw.ndim}"

    if gray.shape[0] < min_height or gray.shape[1] < min_width:
        return None, "too_small"

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray, "accepted"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    dataset_slug = safe_name(args.dataset_name)
    output_dir = args.output_root / dataset_slug
    image_dir = output_dir / "images"

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    accepted = 0
    rejected = 0

    for path in iter_images(args.input_dir):
        gray, status = load_as_bmode(
            path,
            include_rgb=args.include_rgb,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        relative = path.relative_to(args.input_dir)
        output_path = ""

        if gray is not None:
            output_name = safe_name(str(relative.with_suffix(""))) + ".png"
            output_path = image_dir / output_name
            cv2.imwrite(str(output_path), gray)
            accepted += 1
            status = "accepted"
        else:
            rejected += 1
            if gray is not None:
                status = "too_small"

        rows.append(
            {
                "dataset_name": args.dataset_name,
                "original_path": str(path),
                "relative_path": str(relative),
                "status": status,
                "output_path": str(output_path),
            }
        )

    write_csv(output_dir / "manifest.csv", rows)
    summary = {
        "dataset_name": args.dataset_name,
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "accepted_images": accepted,
        "rejected_images": rejected,
        "include_rgb": args.include_rgb,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
