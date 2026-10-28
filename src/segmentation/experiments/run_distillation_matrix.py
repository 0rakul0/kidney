"""Executa as combinações da destilação no mesmo protocolo, sem alterar a Curadoria Web.

Uso: python src/segmentation/experiments/run_distillation_matrix.py --epochs 12 --batch-size 16
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "src/segmentation/experiments/train_distillation_combinations.py"
OUT = ROOT / "results/segmentation_experiments/capsule_distillation_combinations"
COMBINATIONS = ["baseline", "unet+deeplab", "unet+segformer", "deeplab+segformer", "unet+deeplab+segformer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--early-stopping", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--only", choices=COMBINATIONS, nargs="+")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    combinations = args.only or COMBINATIONS
    rows = []
    for combination in combinations:
        cmd = [sys.executable, str(SCRIPT), "--teachers", combination,
               "--epochs", str(args.epochs), "--early-stopping", str(args.early_stopping),
               "--batch-size", str(args.batch_size), "--base-channels", str(args.base_channels)]
        print("Executando:", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        slug = combination.replace("+", "_")
        summary_path = OUT / slug / f"capsule_distillation_{slug}_summary.json"
        row = {"teachers": combination, "returncode": result.returncode, "dice": "",
               "iou": "", "threshold": "", "checkpoint": ""}
        if result.returncode == 0 and summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            test = data["student_metrics"]["test"]
            row.update(dice=test["dice"], iou=test["iou"], threshold=test["threshold"],
                       checkpoint=str(ROOT / "models" / f"kidneyus_capsule_student_{slug}.pth"))
        rows.append(row)
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)
    print("\nComparação salva em:", OUT / "comparison.csv")


if __name__ == "__main__":
    main()
