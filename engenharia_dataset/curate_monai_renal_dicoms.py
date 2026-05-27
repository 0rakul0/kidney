import argparse
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP_DIR = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "fontes"
    / "external_data"
    / "raw"
    / "MONAI_ClinicalUltrasoundRepository"
    / "per-study-zips"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "dataset_aumentado" / "fontes" / "external_data" / "processed" / "monai_renal_png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert selected MONAI/NVIDIA renal ultrasound DICOM archives into "
            "a lighter PNG dataset plus provenance manifests."
        )
    )
    parser.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames-per-dicom", type=int, default=3)
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Include RGB DICOMs. By default they are skipped to avoid Doppler/color frames.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the processed output directory before writing new PNGs.",
    )
    return parser.parse_args()


def normalize_to_uint8(array):
    array = np.asarray(array)

    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)

    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return None

    values = array[finite]
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return None

    clipped = np.clip(array, lo, hi)
    normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return normalized


def selected_frame_indices(frame_count, max_frames):
    if frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames <= 1:
        return [frame_count // 2]
    return sorted(set(np.linspace(0, frame_count - 1, max_frames, dtype=int).tolist()))


def safe_stem(name):
    return Path(name).stem.replace(" ", "_")


def read_dataset_from_zip(zip_file, member):
    with zip_file.open(member) as file:
        data = file.read()
    return pydicom.dcmread(io.BytesIO(data), force=True)


def classify_dataset(ds, include_rgb, min_width, min_height):
    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    photo = str(getattr(ds, "PhotometricInterpretation", ""))

    if rows < min_height or cols < min_width:
        return False, "too_small"

    if samples > 1 and not include_rgb:
        return False, "rgb_or_color_skipped"

    if not hasattr(ds, "PixelData"):
        return False, "no_pixel_data"

    if photo.upper() not in {"MONOCHROME1", "MONOCHROME2", "RGB", "YBR_FULL", "YBR_FULL_422"}:
        return False, f"unsupported_photometric:{photo}"

    return True, "accepted"


def write_metadata_member(zip_file, member, metadata_dir):
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = metadata_dir / Path(member).name
    if output_path.exists():
        return output_path
    with zip_file.open(member) as src, output_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output_path


def process_dicom(zip_path, zip_file, member, output_image_dir, args):
    rows = []

    try:
        ds = read_dataset_from_zip(zip_file, member)
    except Exception as exc:
        return [
            base_manifest_row(zip_path, member)
            | {"status": "rejected", "reason": f"dicom_read_error:{exc}"}
        ]

    ok, reason = classify_dataset(
        ds,
        include_rgb=args.include_rgb,
        min_width=args.min_width,
        min_height=args.min_height,
    )

    base = base_manifest_row(zip_path, member)
    base.update(
        {
            "rows": getattr(ds, "Rows", ""),
            "columns": getattr(ds, "Columns", ""),
            "samples_per_pixel": getattr(ds, "SamplesPerPixel", ""),
            "photometric_interpretation": getattr(ds, "PhotometricInterpretation", ""),
            "number_of_frames": getattr(ds, "NumberOfFrames", "1"),
            "modality": getattr(ds, "Modality", ""),
            "study_description": getattr(ds, "StudyDescription", ""),
            "series_description": getattr(ds, "SeriesDescription", ""),
        }
    )

    if not ok:
        base.update({"status": "rejected", "reason": reason})
        return [base]

    try:
        pixels = ds.pixel_array
    except Exception as exc:
        base.update({"status": "rejected", "reason": f"pixel_decode_error:{exc}"})
        return [base]

    if pixels.ndim == 2:
        frame_arrays = [pixels]
        frame_indices = [0]
    else:
        frame_count = int(getattr(ds, "NumberOfFrames", 1) or 1)
        if frame_count > 1:
            frame_indices = selected_frame_indices(
                frame_count, max(1, args.max_frames_per_dicom)
            )
            frame_arrays = [pixels[index] for index in frame_indices]
        else:
            frame_arrays = [pixels]
            frame_indices = [0]

    for frame_index, frame in zip(frame_indices, frame_arrays):
        image = normalize_to_uint8(frame)
        row = dict(base)
        if image is None:
            row.update({"status": "rejected", "reason": "normalization_failed"})
            rows.append(row)
            continue

        study_ref = zip_path.stem
        dicom_stem = safe_stem(member)
        output_name = f"{study_ref}__{dicom_stem}__frame{frame_index:03d}.png"
        output_path = output_image_dir / output_name
        cv2.imwrite(str(output_path), image)

        row.update(
            {
                "status": "accepted",
                "reason": "converted_to_png",
                "frame_index": frame_index,
                "output_path": str(output_path),
                "output_width": image.shape[1],
                "output_height": image.shape[0],
            }
        )
        rows.append(row)

    return rows


def base_manifest_row(zip_path, member):
    return {
        "source": "MONAI Clinical Ultrasound Image Repository",
        "license": "CC-BY-NC 4.0",
        "zip_path": str(zip_path),
        "study_ref": zip_path.stem,
        "dicom_member": member,
        "status": "",
        "reason": "",
        "frame_index": "",
        "output_path": "",
        "output_width": "",
        "output_height": "",
        "rows": "",
        "columns": "",
        "samples_per_pixel": "",
        "photometric_interpretation": "",
        "number_of_frames": "",
        "modality": "",
        "study_description": "",
        "series_description": "",
    }


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(base_manifest_row(Path("study.zip"), "member").keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_manifest(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_summary(path, rows, output_dir):
    accepted = [row for row in rows if row["status"] == "accepted"]
    reasons = {}
    for row in rows:
        key = f"{row['status']}:{row['reason']}"
        reasons[key] = reasons.get(key, 0) + 1

    total_png_bytes = sum(path.stat().st_size for path in output_dir.glob("images/*.png"))
    summary = {
        "source": "MONAI Clinical Ultrasound Image Repository",
        "license": "CC-BY-NC 4.0",
        "manifest": str(output_dir / "manifest.csv"),
        "total_manifest_rows": len(rows),
        "accepted_png_images": len(accepted),
        "reasons": reasons,
        "processed_png_bytes": total_png_bytes,
        "processed_png_megabytes": round(total_png_bytes / 1024**2, 2),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    output_image_dir = args.output_dir / "images"
    metadata_dir = args.output_dir / "metadata"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.csv"
    existing_rows = [] if args.clear_output else load_existing_manifest(manifest_path)
    processed_studies = {
        row["study_ref"]
        for row in existing_rows
        if row.get("study_ref")
    }
    new_rows = []
    zip_paths = sorted(args.zip_dir.glob("*.zip"))

    for idx, zip_path in enumerate(zip_paths, 1):
        if zip_path.stem in processed_studies:
            print(f"[{idx}/{len(zip_paths)}] {zip_path.name} ja processado; pulando")
            continue
        print(f"[{idx}/{len(zip_paths)}] {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zip_file:
            members = zip_file.namelist()
            for member in members:
                lower = member.lower()
                if lower.endswith(".json") or lower.endswith(".csv"):
                    write_metadata_member(zip_file, member, metadata_dir / zip_path.stem)
                elif lower.endswith(".dcm"):
                    new_rows.extend(
                        process_dicom(zip_path, zip_file, member, output_image_dir, args)
                    )

    rows = existing_rows + new_rows
    write_manifest(manifest_path, rows)
    write_summary(args.output_dir / "summary.json", rows, args.output_dir)

    accepted = sum(1 for row in new_rows if row["status"] == "accepted")
    rejected = len(new_rows) - accepted
    print(f"New accepted PNG images: {accepted}")
    print(f"New rejected rows: {rejected}")
    print(f"Total manifest rows: {len(rows)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
