import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch

from scipy.spatial.distance import directed_hausdorff


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.inference import predict as nnunet_predict


DEFAULT_WEIGHTS_ROOT = Path(r"E:\weights\weights")
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "segmentation_experiments" / "kidneyus_nnunet_runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Executa inferencia nnUNet com os pesos do kidneyUS no mesmo estilo "
            "metodologico do projeto de referencia."
        )
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=DEFAULT_WEIGHTS_ROOT,
        help="Raiz onde estao os pesos do kidneyUS.",
    )
    parser.add_argument(
        "--group",
        choices=["mixed", "annotator_1", "annotator_2"],
        default="mixed",
        help="Conjunto de pesos a usar.",
    )
    parser.add_argument(
        "--task",
        choices=["Task001_KidneyCapsule", "Task002_KidneyRegions"],
        default="Task001_KidneyCapsule",
        help="Tarefa nnUNet a usar.",
    )
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainerV2",
        help="Trainer nnUNet usado no checkpoint.",
    )
    parser.add_argument(
        "--plans",
        default="nnUNetPlansv2.1",
        help="Identificador dos planos nnUNet.",
    )
    parser.add_argument(
        "--model",
        default="2d",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        help="Tipo de modelo nnUNet.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Raiz do dataset local com os splits train/val/test.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split local a ser usado quando --image-dir nao for informado.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Pasta com PNGs para inferencia. Se informada, substitui --dataset-path/--split.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Pasta com mascaras PNG para avaliacao opcional.",
    )
    parser.add_argument(
        "--source-image-root",
        type=Path,
        help=(
            "Pasta alternativa com imagens originais em maior resolucao. "
            "Quando informada, os nomes do split local sao procurados nela."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Raiz onde os resultados serao gravados.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Nome opcional da execucao. Se omitido, um nome descritivo e criado automaticamente.",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        help="Lista de folds a usar. Se omitido, detecta todos os folds disponiveis.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limita o numero de imagens para teste rapido.",
    )
    parser.add_argument(
        "--num-threads-preprocessing",
        type=int,
        default=1,
        help="Numero de processos de preprocessamento do nnUNet.",
    )
    parser.add_argument(
        "--num-threads-nifti-save",
        type=int,
        default=1,
        help="Numero de processos para salvar nifti.",
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Desabilita test-time augmentation.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Habilita mixed precision na inferencia.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve uma pasta de execucao ja existente.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Salva paineis comparando imagem, predicao e mascara de referencia.",
    )
    return parser.parse_args()


def patch_torch_load_for_legacy_nnunet_checkpoints():
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load


def patch_nnunet_preprocessing_for_windows():
    def preprocess_singlethreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
        if segs_from_prev_stage is None:
            segs_from_prev_stage = [None] * len(list_of_lists)

        classes = list(range(1, trainer.num_classes))

        for i, input_files in enumerate(list_of_lists):
            output_file = output_files[i]
            print("preprocessing", output_file)
            data, _, properties = trainer.preprocess_patient(input_files)

            if segs_from_prev_stage[i] is not None:
                seg_prev_path = segs_from_prev_stage[i]
                if not os.path.isfile(seg_prev_path) or not seg_prev_path.endswith(".nii.gz"):
                    raise FileNotFoundError(
                        "segs_from_prev_stage deve apontar para um arquivo .nii.gz valido"
                    )

                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(seg_prev_path))
                input_image = sitk.GetArrayFromImage(sitk.ReadImage(input_files[0]))

                if any(a != b for a, b in zip(seg_prev.shape, input_image.shape)):
                    raise ValueError(
                        "A segmentacao do estagio anterior nao tem o mesmo shape da imagem"
                    )

                seg_prev = seg_prev.transpose(trainer.plans["transpose_forward"])
                seg_reshaped = nnunet_predict.resize_segmentation(seg_prev, data.shape[1:], order=1)
                seg_reshaped = nnunet_predict.to_one_hot(seg_reshaped, classes)
                data = np.vstack((data, seg_reshaped)).astype(np.float32)

            if np.prod(data.shape) > (2e9 / 4 * 0.85):
                print("Entrada grande demais para memoria compartilhada; salvando temporariamente em disco")
                np.save(output_file[:-7] + ".npy", data)
                data = output_file[:-7] + ".npy"

            yield output_file, (data, properties)

    nnunet_predict.preprocess_multithreaded = preprocess_singlethreaded


def detect_folds(model_folder: Path):
    folds = []
    for child in sorted(model_folder.glob("fold_*")):
        checkpoint_path = child / "model_final_checkpoint.model"
        if not checkpoint_path.exists():
            continue
        try:
            folds.append(int(child.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not folds:
        raise FileNotFoundError(f"Nenhum fold encontrado em {model_folder}")
    return tuple(folds)


def resolve_model_folder(args):
    model_folder = (
        args.weights_root
        / args.group
        / args.task
        / f"{args.trainer}__{args.plans}"
    )
    if not model_folder.exists():
        raise FileNotFoundError(f"Pasta do modelo nao encontrada: {model_folder}")
    return model_folder


def resolve_input_paths(args):
    if args.image_dir:
        image_dir = args.image_dir
        mask_dir = args.mask_dir
    else:
        image_dir = args.dataset_path / args.split / "image"
        mask_dir = args.mask_dir or (args.dataset_path / args.split / "mask")

    if not image_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {image_dir}")
    if mask_dir is not None and not mask_dir.exists():
        mask_dir = None

    image_paths = sorted(image_dir.glob("*.png"))
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem PNG encontrada em {image_dir}")

    return image_paths, mask_dir


def build_run_name(args):
    if args.run_name:
        return args.run_name

    source_name = args.split if not args.image_dir else args.image_dir.name
    return f"{args.group}_{args.task}_{args.model}_{source_name}"


def prepare_run_dirs(output_root: Path, run_name: str, overwrite: bool):
    run_dir = output_root / run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"A pasta de execucao ja existe: {run_dir}. Use --overwrite para sobrescrever."
            )
        shutil.rmtree(run_dir)

    nifti_input_dir = run_dir / "nnunet_input"
    nifti_output_dir = run_dir / "nnunet_output"
    png_output_dir = run_dir / "png_masks"
    overlay_dir = run_dir / "overlays"

    for folder in [nifti_input_dir, nifti_output_dir, png_output_dir, overlay_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "nifti_input_dir": nifti_input_dir,
        "nifti_output_dir": nifti_output_dir,
        "png_output_dir": png_output_dir,
        "overlay_dir": overlay_dir,
    }


def find_source_image(image_path: Path, source_image_root: Path | None):
    if source_image_root is None:
        return image_path

    candidate = source_image_root / image_path.name
    return candidate if candidate.exists() else image_path


def write_nnunet_input_nifti(source_image_path: Path, output_path: Path):
    image = cv2.imread(str(source_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Falha ao ler imagem: {source_image_path}")

    volume = image[np.newaxis, :, :].astype(np.uint8)
    itk_image = sitk.GetImageFromArray(volume)
    itk_image.SetSpacing((1.0, 1.0, 999.0))
    sitk.WriteImage(itk_image, str(output_path))

    return image.shape[1], image.shape[0]


def load_binary_png(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Falha ao ler mascara: {mask_path}")
    return (mask > 0).astype(np.uint8)


def load_predicted_mask_from_nifti(nifti_path: Path):
    image = sitk.ReadImage(str(nifti_path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3:
        array = array[0]
    return (array > 0).astype(np.uint8)


def maybe_resize_mask(mask: np.ndarray, target_shape: tuple[int, int]):
    if mask.shape == target_shape:
        return mask

    return cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def dice(pred, target):
    intersection = float((pred * target).sum())
    return (2.0 * intersection) / (float(pred.sum()) + float(target.sum()) + 1e-8)


def iou(pred, target):
    intersection = float((pred * target).sum())
    union = float(pred.sum()) + float(target.sum()) - intersection
    return intersection / (union + 1e-8)


def precision(pred, target):
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    return tp / (tp + fp + 1e-8)


def recall(pred, target):
    tp = float((pred * target).sum())
    fn = float(((1 - pred) * target).sum())
    return tp / (tp + fn + 1e-8)


def f1_score(pred, target):
    p = precision(pred, target)
    r = recall(pred, target)
    return 2.0 * (p * r) / (p + r + 1e-8)


def hausdorff(pred, target):
    pred_points = np.argwhere(pred == 1)
    target_points = np.argwhere(target == 1)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float("nan")

    d1 = directed_hausdorff(pred_points, target_points)[0]
    d2 = directed_hausdorff(target_points, pred_points)[0]
    return float(max(d1, d2))


def compute_metrics(pred_mask: np.ndarray, target_mask: np.ndarray):
    return {
        "dice": float(dice(pred_mask, target_mask)),
        "iou": float(iou(pred_mask, target_mask)),
        "precision": float(precision(pred_mask, target_mask)),
        "recall": float(recall(pred_mask, target_mask)),
        "f1": float(f1_score(pred_mask, target_mask)),
        "hausdorff": float(hausdorff(pred_mask, target_mask)),
    }


def prepare_display_image(image: np.ndarray):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    background = np.all(image_rgb == 0, axis=2)
    image_rgb[background] = 255
    return image_rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]):
    canvas = prepare_display_image(image)
    overlay = canvas.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(canvas, 0.78, overlay, 0.22, 0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(blended, contours, -1, color, 2)
    return blended


def create_labeled_tile(image: np.ndarray, label: str):
    title_height = 34
    padding = 12
    height, width = image.shape[:2]
    tile = np.full(
        (height + title_height + (2 * padding), width + (2 * padding), 3),
        255,
        dtype=np.uint8,
    )

    top = padding + title_height
    left = padding
    tile[top : top + height, left : left + width] = image

    cv2.putText(
        tile,
        label,
        (padding, padding + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        tile,
        (left - 1, top - 1),
        (left + width, top + height),
        (215, 215, 215),
        1,
    )
    return tile


def save_overlay_panel(image: np.ndarray, pred_mask: np.ndarray, output_path: Path, target_mask=None):
    tiles = [
        create_labeled_tile(prepare_display_image(image), "Imagem"),
        create_labeled_tile(overlay_mask(image, pred_mask, (0, 140, 255)), "Predicao nnUNet"),
    ]

    if target_mask is not None:
        tiles.append(
            create_labeled_tile(overlay_mask(image, target_mask, (0, 200, 0)), "Mascara de referencia")
        )
        tiles.append(
            create_labeled_tile(
                overlay_mask(prepare_display_image(image)[:, :, 0], pred_mask ^ target_mask, (255, 0, 255)),
                "Diferencas",
            )
        )

    panel = cv2.hconcat(tiles)
    cv2.imwrite(str(output_path), panel)


def summarise_metrics(rows):
    metrics = ["dice", "iou", "precision", "recall", "f1", "hausdorff"]
    summary = {"samples": len(rows)}
    for metric in metrics:
        values = [row[metric] for row in rows if row.get(metric) is not None and not np.isnan(row[metric])]
        summary[f"{metric}_mean"] = None if not values else float(np.mean(values))
        summary[f"{metric}_std"] = None if not values else float(np.std(values))
    return summary


def main():
    args = parse_args()
    patch_torch_load_for_legacy_nnunet_checkpoints()
    patch_nnunet_preprocessing_for_windows()

    model_folder = resolve_model_folder(args)
    image_paths, mask_dir = resolve_input_paths(args)
    run_name = build_run_name(args)
    run_dirs = prepare_run_dirs(args.output_root, run_name, args.overwrite)

    folds = detect_folds(model_folder) if not args.folds else tuple(int(value) for value in args.folds)
    manifest = []

    for image_path in image_paths:
        source_image_path = find_source_image(image_path, args.source_image_root)
        nifti_name = f"{image_path.stem}_0000.nii.gz"
        nifti_path = run_dirs["nifti_input_dir"] / nifti_name
        width, height = write_nnunet_input_nifti(source_image_path, nifti_path)
        manifest.append(
            {
                "image_name": image_path.name,
                "input_png": str(image_path),
                "source_image": str(source_image_path),
                "input_nifti": str(nifti_path),
                "source_width": width,
                "source_height": height,
            }
        )

    nnunet_predict.predict_from_folder(
        model=str(model_folder),
        input_folder=str(run_dirs["nifti_input_dir"]),
        output_folder=str(run_dirs["nifti_output_dir"]),
        folds=folds,
        save_npz=False,
        num_threads_preprocessing=args.num_threads_preprocessing,
        num_threads_nifti_save=args.num_threads_nifti_save,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=not args.disable_tta,
        mixed_precision=args.mixed_precision,
        overwrite_existing=True,
        mode="normal",
        overwrite_all_in_gpu=False,
        step_size=0.5,
        checkpoint_name="model_final_checkpoint",
    )

    metric_rows = []
    for item in manifest:
        image_name = item["image_name"]
        pred_nifti_path = run_dirs["nifti_output_dir"] / f"{Path(image_name).stem}.nii.gz"
        pred_mask = load_predicted_mask_from_nifti(pred_nifti_path)
        png_mask = (pred_mask * 255).astype(np.uint8)
        png_mask_path = run_dirs["png_output_dir"] / image_name
        cv2.imwrite(str(png_mask_path), png_mask)

        row = {
            "image_name": image_name,
            "source_image": item["source_image"],
            "prediction_nifti": str(pred_nifti_path),
            "prediction_png": str(png_mask_path),
        }

        target_mask = None
        if mask_dir is not None:
            target_path = mask_dir / image_name
            if target_path.exists():
                target_mask = load_binary_png(target_path)
                pred_for_eval = maybe_resize_mask(pred_mask, target_mask.shape)
                row.update(compute_metrics(pred_for_eval, target_mask))
                row["target_mask"] = str(target_path)

        if args.save_overlays:
            source_img = cv2.imread(item["source_image"], cv2.IMREAD_GRAYSCALE)
            if source_img is not None:
                pred_for_panel = pred_mask
                target_for_panel = target_mask
                if target_for_panel is not None and source_img.shape != target_for_panel.shape:
                    target_for_panel = maybe_resize_mask(target_for_panel, source_img.shape)
                overlay_path = run_dirs["overlay_dir"] / image_name
                save_overlay_panel(source_img, pred_for_panel, overlay_path, target_for_panel)
                row["overlay"] = str(overlay_path)

        metric_rows.append(row)

    manifest_path = run_dirs["run_dir"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = run_dirs["run_dir"] / "metrics.csv"
    if metric_rows:
        fieldnames = sorted({key for row in metric_rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    summary = {
        "run_name": run_name,
        "weights_root": str(args.weights_root),
        "group": args.group,
        "task": args.task,
        "trainer": args.trainer,
        "plans": args.plans,
        "model": args.model,
        "folds": list(folds),
        "image_count": len(manifest),
        "image_dir": str(args.image_dir) if args.image_dir else str(args.dataset_path / args.split / "image"),
        "mask_dir": None if mask_dir is None else str(mask_dir),
        "source_image_root": None if args.source_image_root is None else str(args.source_image_root),
        "metrics_summary": summarise_metrics(metric_rows),
        "outputs": {
            "run_dir": str(run_dirs["run_dir"]),
            "manifest_json": str(manifest_path),
            "metrics_csv": str(csv_path),
            "nifti_input_dir": str(run_dirs["nifti_input_dir"]),
            "nifti_output_dir": str(run_dirs["nifti_output_dir"]),
            "png_output_dir": str(run_dirs["png_output_dir"]),
            "overlay_dir": str(run_dirs["overlay_dir"]),
        },
    }

    summary_path = run_dirs["run_dir"] / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Execucao concluida: {run_dirs['run_dir']}")
    print(f"Resumo salvo em: {summary_path}")
    if mask_dir is not None:
        print("Metricas agregadas:")
        for key, value in summary["metrics_summary"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

