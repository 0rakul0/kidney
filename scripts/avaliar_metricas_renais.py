"""Avalia Dice, IoU, HD95 e ASD em teste manual, com inferencia real.

Exemplos:
 python scripts/avaliar_metricas_renais.py --task intrarenal --model unet
 python scripts/avaliar_metricas_renais.py --task intrarenal --model deeplab
 python scripts/avaliar_metricas_renais.py --task capsule --model unet
 Requer scipy, opencv-python, numpy, torch, torchvision.
 HD95/ASD em pixels do grid de avaliacao (256x256), salvo se --spacing Y X for informado.
 """
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.segmentation.experiments.train_unet import build_model as capsule_unet
from src.segmentation.experiments.train_unetplusplus import build_model as capsule_unetpp
from src.segmentation.experiments.train_deeplab import build_model as capsule_deeplab
from src.segmentation.experiments.train_segformer import build_model as capsule_segformer
from src.segmentation.core.dataset import KidneyDataset
from src.segmentation.experiments.train_inner_unet import build_model as inner_unet
from src.segmentation.experiments.train_inner_deeplab import (
    build_model as inner_deeplab,
    IntrarenalMulticlassDataset,
)


def scores(pred, ref, spacing):
    p, r = np.asarray(pred, bool), np.asarray(ref, bool)
    tp = int(np.logical_and(p, r).sum())
    n1, n2 = int(p.sum()), int(r.sum())
    dice = 2 * tp / (n1 + n2) if n1 + n2 else 1.0
    union = int(np.logical_or(p, r).sum())
    iou = tp / union if union else 1.0
    # Ambas vazias: metricas perfeitas. Apenas uma vazia: superficie indefinida.
    if not n1 and not n2:
        return dice, iou, 0.0, 0.0, "both_empty"
    if not n1 or not n2:
        return dice, iou, float("nan"), float("nan"), "one_empty"
    structure = np.ones((3, 3), dtype=bool)
    sp = p & ~binary_erosion(p, structure=structure, border_value=0)
    sr = r & ~binary_erosion(r, structure=structure, border_value=0)
    dp = distance_transform_edt(~sp, sampling=spacing)
    dr = distance_transform_edt(~sr, sampling=spacing)
    distances = np.concatenate([dr[sp], dp[sr]])
    return dice, iou, float(np.percentile(distances, 95)), float(distances.mean()), "ok"


def load_model(task, name, checkpoint, device):
    if task == "intrarenal":
        if name == "unet":
            model = inner_unet()
        elif name == "deeplab":
            model = inner_deeplab(backbone="resnet50", pretrained=False)
        else:
            raise ValueError("Modelo intrarrenal disponivel: unet, deeplab")
    else:
        if name == "unet":
            model = capsule_unet()
        elif name == "unetplusplus":
            model = capsule_unetpp()
        elif name == "deeplab":
            model = capsule_deeplab(backbone="resnet50", pretrained=False)
        elif name == "segformer":
            model = capsule_segformer()
        else:
            raise ValueError(name)
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["intrarenal", "capsule"], required=True)
    parser.add_argument("--model", choices=["unet", "unetplusplus", "deeplab", "segformer"], required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True,
                        help="Aplicar CLAHE como no treinamento de referencia; consulte metadata.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--spacing", type=float, nargs=2, metavar=("Y", "X"), default=(1., 1.))
    args = parser.parse_args()
    if args.task == "intrarenal" and args.model not in ("unet", "deeplab"):
        parser.error("Intrarrenal disponivel apenas para unet/deeplab.")
    stem = ("intrarenal_unet_multiclass_annotator1" if args.model == "unet" else
            "intrarenal_deeplab_resnet50_multiclass_annotator1") if args.task == "intrarenal" else {
                "unet": "unet_best", "unetplusplus": "unetplusplus_best",
                "deeplab": "deeplab_best", "segformer": "segformer_best"}[args.model]
    root = args.dataset_root or (ROOT / "dataset_aumentado/dataset_intrarrenal/supervisionado" /
        ("regions_multiclass_annotator_1" if args.task == "intrarenal" else "capsule_annotator_1"))
    ckpt = args.checkpoint or ROOT / "models" / (stem + ".pth")
    output = args.output or ROOT / "results" / "metricas_renais" / (args.task + "_" + args.model)
    test = root / "test"
    images = sorted((test / "image").glob("*.png"))
    if not images or not ckpt.is_file():
        raise FileNotFoundError(f"Teste ou checkpoint ausente: {test}, {ckpt}")
    if ckpt.stat().st_size < 5000:
        raise RuntimeError(f"Checkpoint parece ponteiro Git LFS: {ckpt}. Execute git lfs pull.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.task, args.model, ckpt, device)
    inner_ds = IntrarenalMulticlassDataset(test, img_size=args.img_size, augment=False, clahe=args.clahe) if args.task == "intrarenal" else None
    capsule_ds = KidneyDataset(test, img_size=args.img_size, augment=False, clahe=args.clahe) if args.task == "capsule" else None
    if inner_ds is not None:
        images = [test / "image" / name for name in inner_ds.names]
    rows = []
    threshold = args.threshold if args.threshold is not None else (0.35 if args.model == "unet" else 0.5)
    with torch.inference_mode():
        for idx, f in enumerate(images):
            if inner_ds is not None:
                x, y = inner_ds[idx]
                x = x[None].to(device)
                gt = y.numpy()
            else:
                x, y = capsule_ds[idx]
                x = x[None].to(device)
                gt = (y.numpy() > 0).astype(np.uint8)
            prediction = model(x)
            if hasattr(prediction, "logits"):
                prediction = prediction.logits
            elif isinstance(prediction, dict):
                prediction = prediction["out"]
            if prediction.shape[-2:] != gt.shape:
                prediction = torch.nn.functional.interpolate(prediction, size=gt.shape, mode="bilinear", align_corners=False)
            if args.task == "intrarenal":
                pred = prediction.argmax(1)[0].cpu().numpy()
                classes = [(1, "cortex"), (2, "medulla"), (3, "central_echo_complex")]
            else:
                pred = (prediction.sigmoid()[0, 0].cpu().numpy() >= threshold).astype(np.uint8)
                classes = [(1, "capsule")]
            for value, label in classes:
                d, iou, hd, asd, status = scores(pred == value, gt == value, args.spacing)
                rows.append({"image": f.name, "class": label, "dice": d, "iou": iou,
                             "hd95": hd, "asd": asd, "status": status})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "por_imagem.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"task": args.task, "model": args.model, "checkpoint": str(ckpt),
               "dataset": str(test), "n_images": len(images),
               "spacing_yx": args.spacing, "distance_units": "pixels" if args.spacing == (1., 1.) else "spacing_units",
               "empty_policy": "both empty: Dice/IoU=1, distances=0; one empty: Dice/IoU=0, distances=NaN",
               "per_class": {}}
    for _, label in classes:
        sub = [row for row in rows if row["class"] == label]
        summary["per_class"][label] = {
            metric: {"mean": float(np.nanmean([r[metric] for r in sub])),
                     "valid_n": sum(np.isfinite(r[metric]) for r in sub)}
            for metric in ("dice", "iou", "hd95", "asd")
        }
        summary["per_class"][label]["one_empty_n"] = sum(r["status"] == "one_empty" for r in sub)
    (output / "resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
