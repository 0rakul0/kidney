import os
from pathlib import Path
import csv

from PIL import Image, ImageOps, ImageDraw, ImageFont


BASE_DIR = Path(r"D:\kidney")
SOURCE_DIR = BASE_DIR / "results" / "qualitative_comparison"
OUTPUT_DIR = BASE_DIR / "artigo" / "figures"


CASES = [
    {
        "source": SOURCE_DIR / "segmenter_comparison_7_IM-0622-0020_anon.png",
        "metrics": SOURCE_DIR / "segmenter_comparison_7_IM-0622-0020_anon.csv",
        "output": OUTPUT_DIR / "quality_good_comparison.png",
    },
    {
        "source": SOURCE_DIR / "segmenter_comparison_1_IM-0001-0059_anon.png",
        "metrics": SOURCE_DIR / "segmenter_comparison_1_IM-0001-0059_anon.csv",
        "output": OUTPUT_DIR / "quality_bad_comparison.png",
    },
]


def trim_whitespace(image):
    inverted = ImageOps.invert(image.convert("RGB"))
    bbox = inverted.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def crop_main_row(image):
    width, height = image.size

    # The comparison sheets have four panels on the first row that already
    # contain the exact visual evidence needed in the article: original image,
    # ground truth, kidneyUS, and DeepLab.
    left = int(width * 0.02)
    right = int(width * 0.98)
    top = int(height * 0.06)
    bottom = int(height * 0.50)

    cropped = image.crop((left, top, right, bottom))
    return trim_whitespace(cropped)


def split_into_panels(image):
    width, height = image.size
    panels = []
    panel_top_crop = max(18, int(height * 0.06))

    for idx in range(4):
        left = int(round(idx * width / 4))
        right = int(round((idx + 1) * width / 4))
        panel = trim_whitespace(image.crop((left, panel_top_crop, right, height)))
        panels.append(panel)

    return panels


def load_metrics(csv_path):
    metrics = {}

    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics[row["model_name"]] = float(row["dice_on_selected_image"])

    return metrics


def load_title_font():
    font_candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
        Path(r"C:\Windows\Fonts\DejaVuSans.ttf"),
    ]

    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), 30)

    return ImageFont.load_default()


def build_panel_canvas(panels, metrics):
    titles = [
        "Imagem original",
        "Ground truth",
        f"kidneyUS (concorrente)\nDice={metrics['kidneyUS']:.4f}",
        f"DeepLab (nosso)\nDice={metrics['DeepLab']:.4f}",
    ]

    target_width = max(panel.size[0] for panel in panels)
    target_height = max(panel.size[1] for panel in panels)
    title_font = load_title_font()
    title_height = 112
    padding = 18
    gap = 18

    canvas_width = (4 * (target_width + (2 * padding))) + (3 * gap)
    canvas_height = target_height + title_height + (2 * padding)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, panel in enumerate(panels):
        x0 = idx * ((target_width + (2 * padding)) + gap)
        y0 = 0

        title_bbox = draw.multiline_textbbox(
            (0, 0),
            titles[idx],
            font=title_font,
            spacing=6,
            align="center",
        )
        title_width = title_bbox[2] - title_bbox[0]
        title_height_text = title_bbox[3] - title_bbox[1]
        title_x = x0 + padding + ((target_width - title_width) / 2)
        title_y = max(8, (title_height - title_height_text) / 2 - 4)

        draw.multiline_text(
            (title_x, title_y),
            titles[idx],
            fill=(30, 30, 30),
            font=title_font,
            spacing=6,
            align="center",
        )

        panel_x = x0 + padding + (target_width - panel.size[0]) // 2
        panel_y = y0 + title_height + (target_height - panel.size[1]) // 2
        canvas.paste(panel, (panel_x, panel_y))

        draw.rectangle(
            [
                (x0 + padding - 1, title_height - 1),
                (x0 + padding + target_width, title_height + target_height),
            ],
            outline=(210, 210, 210),
            width=2,
        )

    draw.rectangle(
        [(6, 6), (canvas.size[0] - 7, canvas.size[1] - 7)],
        outline=(210, 210, 210),
        width=2,
    )

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for case in CASES:
        image = Image.open(case["source"]).convert("RGB")
        metrics = load_metrics(case["metrics"])
        main_row = crop_main_row(image)
        panels = split_into_panels(main_row)
        final = build_panel_canvas(panels, metrics)
        final.save(case["output"], quality=95)
        print(f"Saved {case['output']}")


if __name__ == "__main__":
    main()
