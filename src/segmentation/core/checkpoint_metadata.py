import json
from pathlib import Path


def get_checkpoint_metadata_path(checkpoint_path):

    path = Path(checkpoint_path)

    return path.with_suffix(".meta.json")


def save_checkpoint_metadata(checkpoint_path, metadata):

    metadata_path = get_checkpoint_metadata_path(checkpoint_path)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return metadata_path


def load_checkpoint_metadata(checkpoint_path):

    metadata_path = get_checkpoint_metadata_path(checkpoint_path)

    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)

