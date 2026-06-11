import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.dataset import KidneyDataset
from src.segmentation.tools.benchmark_models import evaluate_model
from src.segmentation.core.model_loader import load_model_bundle


DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "segmentation_experiments" / "kidneyus_capsule_benchmark"
CHECKPOINTS = {
    "unet": PROJECT_ROOT / "models" / "kidneyus_capsule_unet.pth",
    "unetplusplus": PROJECT_ROOT / "models" / "kidneyus_capsule_unetplusplus.pth",
    "deeplab": PROJECT_ROOT / "models" / "kidneyus_capsule_deeplab.pth",
    "segformer": PROJECT_ROOT / "models" / "kidneyus_capsule_segformer.pth",
}


def main():
    img_size = 256
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = KidneyDataset(DATASET_ROOT / "test", img_size=img_size, augment=False, clahe=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    rows = []
    for model_name, checkpoint in CHECKPOINTS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint ausente: {checkpoint}")
        bundle = load_model_bundle(
            model_name,
            device=device,
            checkpoint_path=checkpoint,
            model_dir=PROJECT_ROOT / "models",
        )
        metrics = evaluate_model(
            bundle["model"],
            bundle["display_name"],
            bundle["threshold"],
            test_loader,
            test_dataset,
            device,
            img_size,
        )
        rows.append(
            {
                "model_key": model_name,
                "model": bundle["display_name"],
                "checkpoint": str(checkpoint),
                "threshold": float(bundle["threshold"]),
                **{key.lower(): float(value) for key, value in metrics.items()},
            }
        )

    df = pd.DataFrame(rows).sort_values(["dice", "iou"], ascending=False)
    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"
    summary_path = output_dir / "summary.json"
    plot_path = output_dir / "dice_comparison.png"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["dice"])
    plt.ylabel("Dice Score")
    plt.title("kidneyUS Capsule Benchmark")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "test_samples": len(test_dataset),
        "device": device,
        "img_size": img_size,
        "batch_size": batch_size,
        "clahe": True,
        "best_model": df.iloc[0].to_dict(),
        "results_csv": str(csv_path),
        "results_json": str(json_path),
        "plot": str(plot_path),
        "note": (
            "Benchmark dos quatro modelos treinados novamente na base canonica "
            "kidneyUS Capsule, usando o mesmo split de teste por paciente."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
