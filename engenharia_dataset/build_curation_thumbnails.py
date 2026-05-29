import csv
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_AUMENTADO = PROJECT_ROOT / "dataset_aumentado"
CURATION_ROOT = DATASET_AUMENTADO / "curadoria"
INPUT_CSV = CURATION_ROOT / "curadoria_mascaras.csv"
REGIONS_ROOT = DATASET_AUMENTADO / "dataset_intrarrenal" / "intermediario" / "kidneyus_regions"
OUTPUT_ROOT = CURATION_ROOT / "miniaturas_completas"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest.csv"
CORTEX_PREDICTIONS = (
    PROJECT_ROOT / "results" / "intrarenal_model3" / "cortex_roi_unet_predictions_dataset_geral" / "manifest.csv"
)
MULTICLASS_PREDICTIONS = (
    PROJECT_ROOT / "results" / "intrarenal_model3" / "intrarenal_multiclass_predictions_dataset_geral" / "manifest.csv"
)
THUMB_SIZE = (240, 170)


def predicted_cortex_index():
    if not CORTEX_PREDICTIONS.exists():
        return {}
    with CORTEX_PREDICTIONS.open("r", newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        return {
            row["image_id"]: Path(row["predicted_cortex_mask_path"])
            for row in rows
            if row.get("prediction_status", "").startswith("candidate_")
            and row.get("predicted_cortex_mask_path")
            and Path(row["predicted_cortex_mask_path"]).exists()
        }


def predicted_multiclass_index():
    if not MULTICLASS_PREDICTIONS.exists():
        return {}
    index = {}
    with MULTICLASS_PREDICTIONS.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if not row.get("prediction_status", "").startswith("candidate_"):
                continue
            paths = {
                "cortex": row.get("predicted_cortex_mask_path", ""),
                "medulla": row.get("predicted_medulla_mask_path", ""),
                "central_echo_complex": row.get("predicted_central_echo_complex_mask_path", ""),
            }
            existing = {name: Path(path) for name, path in paths.items() if path and Path(path).exists()}
            if existing:
                index[row["image_id"]] = existing
    return index


def same_size(image_path, mask_path):
    if not mask_path:
        return True
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        return image.size == mask.size


def manual_visual_sources(filename):
    source_image = REGIONS_ROOT / "images" / filename
    capsule = REGIONS_ROOT / "full_masks" / "annotator_1" / "capsule" / filename
    cortex = REGIONS_ROOT / "full_masks" / "annotator_1" / "cortex" / filename
    medulla = REGIONS_ROOT / "full_masks" / "annotator_1" / "medulla" / filename
    central = REGIONS_ROOT / "full_masks" / "annotator_1" / "central_echo_complex" / filename
    if source_image.exists() and medulla.exists():
        return {
            "imagem_visual": source_image,
            "mascara_rim_visual": capsule if capsule.exists() else None,
            "mascara_cortex_visual": cortex if cortex.exists() else None,
            "mascara_medulla_visual": medulla,
            "mascara_central_echo_complex_visual": central if central.exists() else None,
            "origem_visual": "manual_full_resolution",
        }
    return None


def visual_sources(row, predicted_cortex, predicted_multiclass):
    filename = Path(row["imagem"]).name.split("__", 1)[-1]
    manual = manual_visual_sources(filename)
    if manual and row["anot2"] and "full_masks" in row["anot2"]:
        return manual
    image_id = Path(row["imagem"]).stem
    multiclass = predicted_multiclass.get(image_id, {})
    return {
        "imagem_visual": Path(row["imagem"]),
        "mascara_rim_visual": Path(row["anot1"]) if row["anot1"] else None,
        "mascara_cortex_visual": multiclass.get("cortex") or predicted_cortex.get(image_id),
        "mascara_medulla_visual": Path(row["anot2"]) if row["anot2"] else None,
        "mascara_central_echo_complex_visual": multiclass.get("central_echo_complex"),
        "origem_visual": "dataset_geral_prediction_space",
    }


def load_canvas(path):
    image = Image.open(path).convert("L")
    thumb = ImageOps.contain(image, THUMB_SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", THUMB_SIZE, (0, 0, 0))
    position = ((THUMB_SIZE[0] - thumb.width) // 2, (THUMB_SIZE[1] - thumb.height) // 2)
    canvas.paste(thumb.convert("RGB"), position)
    return canvas, position


def overlay_contour(image_path, mask_path, color):
    canvas, position = load_canvas(image_path)
    if not mask_path:
        return canvas
    if not same_size(image_path, mask_path):
        raise ValueError(f"Dimensoes incompatíveis para overlay: imagem={image_path} mascara={mask_path}")
    mask = Image.open(mask_path).convert("L").point(lambda value: 255 if value > 0 else 0)
    resized = ImageOps.contain(mask, THUMB_SIZE, Image.Resampling.NEAREST)
    boundary = ImageChops.difference(
        resized.filter(ImageFilter.MaxFilter(5)),
        resized.filter(ImageFilter.MinFilter(5)),
    )
    boundary_canvas = Image.new("L", THUMB_SIZE, 0)
    boundary_canvas.paste(boundary, position)
    solid = Image.new("RGB", THUMB_SIZE, color)
    return Image.composite(solid, canvas, boundary_canvas)


def save_jpeg(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=68, optimize=True)


def main():
    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    predicted_cortex = predicted_cortex_index()
    predicted_multiclass = predicted_multiclass_index()
    original_dir = OUTPUT_ROOT / "imagem"
    rim_dir = OUTPUT_ROOT / "overlay_rim"
    cortex_dir = OUTPUT_ROOT / "overlay_cortex"
    medulla_dir = OUTPUT_ROOT / "overlay_medulla"
    central_dir = OUTPUT_ROOT / "overlay_central_echo_complex"
    manifest_rows = []

    for index, row in enumerate(rows, start=1):
        image_id = Path(row["imagem"]).stem
        sources = visual_sources(row, predicted_cortex, predicted_multiclass)
        image_path = sources["imagem_visual"]
        rim_mask = sources["mascara_rim_visual"]
        cortex_mask = sources["mascara_cortex_visual"]
        medulla_mask = sources["mascara_medulla_visual"]
        central_mask = sources["mascara_central_echo_complex_visual"]
        original_path = original_dir / f"{image_id}.jpg"
        rim_path = rim_dir / f"{image_id}.jpg"
        cortex_path = cortex_dir / f"{image_id}.jpg"
        medulla_path = medulla_dir / f"{image_id}.jpg"
        central_path = central_dir / f"{image_id}.jpg"

        original, _ = load_canvas(image_path)
        save_jpeg(original, original_path)
        if rim_mask:
            save_jpeg(overlay_contour(image_path, rim_mask, (255, 55, 55)), rim_path)
        if cortex_mask:
            save_jpeg(overlay_contour(image_path, cortex_mask, (0, 220, 255)), cortex_path)
        if medulla_mask:
            save_jpeg(overlay_contour(image_path, medulla_mask, (255, 220, 30)), medulla_path)
        if central_mask:
            save_jpeg(overlay_contour(image_path, central_mask, (255, 145, 0)), central_path)

        manifest_rows.append(
            {
                "image_id": image_id,
                "origem_visual": sources["origem_visual"],
                "imagem_visual": str(image_path.resolve()),
                "mascara_rim_visual": str(rim_mask.resolve()) if rim_mask else "",
                "mascara_cortex_visual": str(cortex_mask.resolve()) if cortex_mask else "",
                "mascara_medulla_visual": str(medulla_mask.resolve()) if medulla_mask else "",
                "mascara_central_echo_complex_visual": str(central_mask.resolve()) if central_mask else "",
                "imagem_thumb": str(original_path.resolve()),
                "rim_overlay_thumb": str(rim_path.resolve()) if rim_mask else "",
                "cortex_overlay_thumb": str(cortex_path.resolve()) if cortex_mask else "",
                "medulla_overlay_thumb": str(medulla_path.resolve()) if medulla_mask else "",
                "central_echo_complex_overlay_thumb": str(central_path.resolve()) if central_mask else "",
            }
        )
        if index % 500 == 0:
            print(f"miniaturas={index}/{len(rows)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(
        f"rows={len(manifest_rows)} predicted_cortex={len(predicted_cortex)} "
        f"predicted_multiclass={len(predicted_multiclass)} "
        f"manifest={OUTPUT_MANIFEST.resolve()}"
    )


if __name__ == "__main__":
    main()
