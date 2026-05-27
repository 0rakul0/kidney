import csv
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATION_ROOT = PROJECT_ROOT / "dataset_aumentado" / "curadoria"
INPUT_CSV = CURATION_ROOT / "curadoria_mascaras.csv"
OUTPUT_ROOT = CURATION_ROOT / "miniaturas_completas"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest.csv"
THUMB_SIZE = (240, 170)


def load_canvas(path):
    image = Image.open(path).convert("L")
    thumb = ImageOps.contain(image, THUMB_SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", THUMB_SIZE, (0, 0, 0))
    position = ((THUMB_SIZE[0] - thumb.width) // 2, (THUMB_SIZE[1] - thumb.height) // 2)
    canvas.paste(thumb.convert("RGB"), position)
    return image, canvas, position, thumb.size


def overlay_contour(image, mask_path, color):
    _, canvas, position, thumb_size = load_canvas(image)
    if not mask_path:
        return canvas
    mask_source = Path(mask_path)
    if not mask_source.exists():
        return canvas
    mask = Image.open(mask_source).convert("L").point(lambda value: 255 if value > 0 else 0)
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

    original_dir = OUTPUT_ROOT / "imagem"
    rim_dir = OUTPUT_ROOT / "overlay_rim"
    medulla_dir = OUTPUT_ROOT / "overlay_medulla"
    manifest_rows = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["imagem"])
        image_id = image_path.stem
        original_path = original_dir / f"{image_id}.jpg"
        rim_path = rim_dir / f"{image_id}.jpg"
        medulla_path = medulla_dir / f"{image_id}.jpg"

        _, original, _, _ = load_canvas(image_path)
        save_jpeg(original, original_path)
        save_jpeg(overlay_contour(image_path, row["anot1"], (255, 55, 55)), rim_path)
        if row["anot2"]:
            save_jpeg(overlay_contour(image_path, row["anot2"], (255, 220, 30)), medulla_path)

        manifest_rows.append(
            {
                "image_id": image_id,
                "imagem_thumb": str(original_path.resolve()),
                "rim_overlay_thumb": str(rim_path.resolve()),
                "medulla_overlay_thumb": str(medulla_path.resolve()) if row["anot2"] else "",
            }
        )
        if index % 500 == 0:
            print(f"miniaturas={index}/{len(rows)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_id", "imagem_thumb", "rim_overlay_thumb", "medulla_overlay_thumb"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"rows={len(manifest_rows)} manifest={OUTPUT_MANIFEST.resolve()}")


if __name__ == "__main__":
    main()
