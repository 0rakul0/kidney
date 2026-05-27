import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = "https://clinical-ultrasound-image-repository.s3.amazonaws.com"
RAW_ROOT = PROJECT_ROOT / "dataset_aumentado" / "fontes" / "external_data" / "raw" / "MONAI_ClinicalUltrasoundRepository"
INDEX_ROOT = PROJECT_ROOT / "dataset_aumentado" / "fontes" / "external_data" / "indices"
PATTERNS = ("RENAL", "RETROPERITONEAL", "KIDNEY")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download MONAI/NVIDIA Clinical Ultrasound metadata, find renal "
            "candidate studies, estimate archive sizes, and optionally download "
            "a size-capped subset."
        )
    )
    parser.add_argument("--max-gb", type=float, default=2.0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help=(
            "Inclui estudos que ja aparecem no manifesto processado. Por padrao, "
            "eles sao pulados para permitir baixar o MONAI em lotes."
        ),
    )
    return parser.parse_args()


def download(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, output_path)


def download_metadata():
    meta_dir = RAW_ROOT / "meta-only"
    download(f"{HOST}/archives/meta-only/all-meta.json", meta_dir / "all-meta.json")
    download(f"{HOST}/archives/meta-only/all-meta.csv", meta_dir / "all-meta.csv")
    return meta_dir / "all-meta.json"


def find_renal_candidates(meta_json_path):
    with meta_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []
    for patient, item in data.items():
        ref = item.get("_ref", "")
        text = json.dumps(item, ensure_ascii=False).upper() + " " + ref.upper()
        if ref.startswith("A") and any(pattern in text for pattern in PATTERNS):
            demo = item.get("demo", {}) or {}
            exam = item.get("exam", {}) or {}
            rows.append(
                {
                    "patient": patient,
                    "ref": ref,
                    "archive_url": f"{HOST}/archives/per-study/{ref}.zip",
                    "description": ref,
                    "sex": demo.get("sex") or demo.get("Sex") or "",
                    "age": demo.get("age") or demo.get("Age") or "",
                    "exam_keys": ";".join(sorted(exam.keys()))
                    if isinstance(exam, dict)
                    else "",
                }
            )

    out_path = INDEX_ROOT / "monai_clinical_ultrasound_renal_studies.csv"
    write_csv(out_path, rows)
    return rows


def get_content_length(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length", "")
    return int(value) if value.isdigit() else None


def estimate_sizes(rows, sleep_seconds):
    out_path = INDEX_ROOT / "monai_renal_study_download_sizes.csv"
    existing = {}
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as file:
            existing = {row["ref"]: row for row in csv.DictReader(file)}

    results = []
    for idx, row in enumerate(rows, 1):
        if row["ref"] in existing and existing[row["ref"]].get("content_length_bytes"):
            results.append(existing[row["ref"]])
            continue

        length = ""
        error = ""
        try:
            current_length = get_content_length(row["archive_url"])
            length = str(current_length) if current_length is not None else ""
        except Exception as exc:
            error = str(exc)

        results.append(
            {
                "ref": row["ref"],
                "archive_url": row["archive_url"],
                "content_length_bytes": length,
                "error": error,
            }
        )
        if idx % 25 == 0:
            print(f"Checked {idx}/{len(rows)}")
        time.sleep(sleep_seconds)

    write_csv(out_path, results)
    return results


def load_processed_studies():
    manifest_path = (
        PROJECT_ROOT
        / "dataset_aumentado"
        / "fontes"
        / "external_data"
        / "processed"
        / "monai_renal_png"
        / "manifest.csv"
    )
    if not manifest_path.exists():
        return set()

    with manifest_path.open(newline="", encoding="utf-8") as file:
        return {
            row["study_ref"]
            for row in csv.DictReader(file)
            if row.get("study_ref")
        }


def download_size_capped_subset(size_rows, max_gb, include_processed=False):
    selected = []
    running = 0
    cap = int(max_gb * 1024**3)
    processed_studies = set() if include_processed else load_processed_studies()

    sized = [
        {
            **row,
            "content_length_bytes": int(row["content_length_bytes"]),
        }
        for row in size_rows
        if str(row["content_length_bytes"]).isdigit()
    ]
    sized.sort(key=lambda row: row["content_length_bytes"])

    for row in sized:
        if row["ref"] in processed_studies:
            continue
        if running + row["content_length_bytes"] > cap:
            break
        selected.append(row)
        running += row["content_length_bytes"]

    out_dir = RAW_ROOT / "per-study-zips"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for idx, row in enumerate(selected, 1):
        dest = out_dir / f"{row['ref']}.zip"
        status = (
            "existing"
            if dest.exists() and dest.stat().st_size == row["content_length_bytes"]
            else "downloaded"
        )
        if status == "downloaded":
            print(
                f"[{idx}/{len(selected)}] {row['ref']} "
                f"{row['content_length_bytes'] / 1024**2:.1f} MB"
            )
            download(row["archive_url"], dest)

        manifest_rows.append(
            {
                "source": "MONAI Clinical Ultrasound Image Repository",
                "ref": row["ref"],
                "archive_url": row["archive_url"],
                "local_path": str(dest),
                "content_length_bytes": row["content_length_bytes"],
                "status": status,
                "license": "CC-BY-NC 4.0",
                "notes": (
                    "Renal/retroperitoneal candidate selected from smallest "
                    "archives for initial local pseudo-labeling triage."
                ),
            }
        )

    manifest_path = INDEX_ROOT / "monai_downloaded_renal_subset_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    print(f"Skipped already processed studies: {len(processed_studies)}")
    print(f"Selected {len(selected)} archives, {running / 1024**3:.2f} GB")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    meta_json_path = download_metadata()
    rows = find_renal_candidates(meta_json_path)
    print(f"Renal candidate studies: {len(rows)}")

    size_rows = estimate_sizes(rows, sleep_seconds=args.sleep)
    if not args.metadata_only:
        download_size_capped_subset(
            size_rows,
            max_gb=args.max_gb,
            include_processed=args.include_processed,
        )


if __name__ == "__main__":
    main()
