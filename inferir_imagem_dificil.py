from pathlib import Path

import cv2
import numpy as np
import torch

from src.segmentation.build_dataset_geral import (
    keep_largest_component,
    predict_probability,
    prepare_tensor,
)
from src.segmentation.core.model_loader import load_model_bundle


PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_PATH = PROJECT_ROOT / "imagem_dificil.png"

OUTPUT_DIR = PROJECT_ROOT / "output_imagem_dificil"

CHECKPOINTS = {
    "unet": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_unet.pth",
    "deeplab": PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_deeplab.pth",
}


def dice(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    denom = a.sum() + b.sum()

    if denom == 0:
        return 1.0

    return 2 * np.logical_and(a, b).sum() / denom


def predict_mask(bundle, image, device, img_size=256):
    # Mesmo pré-processamento usado no consenso do projeto
    original = prepare_tensor(
        image,
        img_size,
        device,
        clahe=True,
    )

    flipped = prepare_tensor(
        cv2.flip(image, 1),
        img_size,
        device,
        clahe=True,
    )

    with torch.no_grad():
        probability = predict_probability(
            bundle,
            original,
            img_size,
        )

        flipped_probability = np.fliplr(
            predict_probability(
                bundle,
                flipped,
                img_size,
            )
        )

    # TTA: média original + flip
    probability = (
        probability + flipped_probability
    ) / 2.0

    # Retorna para a resolução original
    probability = cv2.resize(
        probability,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    threshold = float(bundle["threshold"])

    mask = probability >= threshold

    mask = keep_largest_component(mask)

    return mask.astype(np.uint8), probability


def draw_contour(image, mask, color):
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        rgb,
        contours,
        -1,
        color,
        3,
    )

    return rgb


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Dispositivo: {device}")

    image = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Imagem não encontrada: {IMAGE_PATH}"
        )

    bundles = {
        name: load_model_bundle(
            name,
            device=device,
            checkpoint_path=checkpoint,
        )
        for name, checkpoint in CHECKPOINTS.items()
    }

    print(
        "Threshold U-Net:",
        bundles["unet"]["threshold"],
    )

    print(
        "Threshold DeepLab:",
        bundles["deeplab"]["threshold"],
    )

    unet_mask, unet_prob = predict_mask(
        bundles["unet"],
        image,
        device,
    )

    deeplab_mask, deeplab_prob = predict_mask(
        bundles["deeplab"],
        image,
        device,
    )

    consensus = dice(
        unet_mask,
        deeplab_mask,
    )

    print(
        f"Consenso U-Net × DeepLabV3: "
        f"{consensus:.4f}"
    )

    # Salva imagem original
    cv2.imwrite(
        str(OUTPUT_DIR / "original.png"),
        image,
    )

    # Salva máscaras
    cv2.imwrite(
        str(OUTPUT_DIR / "mask_unet.png"),
        unet_mask * 255,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "mask_deeplab.png"),
        deeplab_mask * 255,
    )

    # Mapas de probabilidade
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "probabilidade_unet.png"
        ),
        (unet_prob * 255).clip(
            0, 255
        ).astype(np.uint8),
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "probabilidade_deeplab.png"
        ),
        (deeplab_prob * 255).clip(
            0, 255
        ).astype(np.uint8),
    )

    # Contornos separados
    unet_overlay = draw_contour(
        image,
        unet_mask,
        (255, 0, 0),
    )

    deeplab_overlay = draw_contour(
        image,
        deeplab_mask,
        (0, 140, 255),
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "overlay_unet.png"
        ),
        unet_overlay,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "overlay_deeplab.png"
        ),
        deeplab_overlay,
    )

    # Sobreposição conjunta
    overlay = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    contours_unet, _ = cv2.findContours(
        unet_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours_deep, _ = cv2.findContours(
        deeplab_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours_unet,
        -1,
        (255, 0, 0),
        3,
    )

    cv2.drawContours(
        overlay,
        contours_deep,
        -1,
        (0, 140, 255),
        3,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "overlay_consenso.png"
        ),
        overlay,
    )

    print()
    print("Arquivos gerados:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()