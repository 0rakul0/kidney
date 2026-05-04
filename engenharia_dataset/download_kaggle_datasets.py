import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "kaggle_datasets.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "external_data" / "raw"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download selected Kaggle datasets listed in config/kaggle_datasets.csv."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every dataset in the config, including lower-priority CT datasets.",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        help="Download one specific Kaggle slug. Can be passed more than once.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Keep Kaggle zip files instead of extracting them.",
    )
    return parser.parse_args()


def require_kaggle_cli():
    kaggle = shutil.which("kaggle")
    if kaggle:
        return kaggle

    local_executable = Path(sys.executable).resolve().parent / "kaggle.exe"
    if local_executable.exists():
        return str(local_executable)

    raise SystemExit(
        "Kaggle CLI was not found. Install it with:\n"
        "  python -m pip install kaggle\n\n"
        "Then create a Kaggle API token and place kaggle.json in:\n"
        f"  {Path.home() / '.kaggle' / 'kaggle.json'}"
    )


def require_kaggle_token():
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    env_ready = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if token_path.exists() or env_ready:
        return

    raise SystemExit(
        "Kaggle credentials were not found. Create an API token in Kaggle "
        "Account settings and place kaggle.json in:\n"
        f"  {token_path}\n"
        "Alternatively set KAGGLE_USERNAME and KAGGLE_KEY."
    )


def read_targets(config_path, include_all, explicit_slugs):
    with config_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if explicit_slugs:
        wanted = set(explicit_slugs)
        selected = [row for row in rows if row["slug"] in wanted]
        missing = wanted - {row["slug"] for row in selected}
        if missing:
            raise SystemExit(f"Slug(s) not found in config: {', '.join(sorted(missing))}")
        return selected

    if include_all:
        return rows

    return [row for row in rows if row["include_by_default"].lower() == "true"]


def slug_to_dirname(slug):
    return slug.replace("/", "__")


def download_dataset(kaggle, row, output_dir, unzip):
    slug = row["slug"]
    dataset_dir = output_dir / slug_to_dirname(slug)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    command = [
        kaggle,
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(dataset_dir),
    ]
    if unzip:
        command.append("--unzip")

    print(f"Downloading {slug} -> {dataset_dir}")
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    kaggle = require_kaggle_cli()
    require_kaggle_token()

    targets = read_targets(args.config, args.all, args.slug)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for row in targets:
        download_dataset(kaggle, row, args.output_dir, unzip=not args.no_unzip)

    print(f"Downloaded {len(targets)} dataset(s) into {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
