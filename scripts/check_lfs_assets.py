"""Check whether Git LFS assets are available as real files.

This check never downloads models. Use ``git lfs pull`` explicitly when the
report identifies missing or pointer-only files.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


POINTER_MARKER = b"version https://git-lfs.github.com/spec/v1"
ASSET_PATTERNS = ("*.pth", "*.pt", "*.ckpt", "*.bin", "*.safetensors")


def tracked_assets(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", *ASSET_PATTERNS],
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def classify(path: Path) -> str:
    if not path.exists():
        return "ausente"
    try:
        with path.open("rb") as handle:
            header = handle.read(256)
    except OSError:
        return "ilegivel"
    if POINTER_MARKER in header:
        return "ponteiro_lfs"
    return "conteudo_presente"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="retorna erro se algum asset não estiver disponível")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = tracked_assets(root)
    counts = {"conteudo_presente": 0, "ponteiro_lfs": 0, "ausente": 0, "ilegivel": 0}
    for path in assets:
        status = classify(path)
        counts[status] += 1
        if status != "conteudo_presente":
            print(f"{status}: {path.relative_to(root)}")
    print(f"Assets LFS rastreados: {len(assets)}")
    print(f"Conteúdo presente: {counts['conteudo_presente']}")
    print(f"Ponteiros LFS: {counts['ponteiro_lfs']}")
    print(f"Ausentes/ilegíveis: {counts['ausente'] + counts['ilegivel']}")
    if args.strict and any(counts[key] for key in ("ponteiro_lfs", "ausente", "ilegivel")):
        print("Execute 'git lfs install' e 'git lfs pull' antes de executar os modelos.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
