from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\kidney")
FIGURE = (
    ROOT
    / "artigo"
    / "SBBD_2026___Jefferson"
    / "figures"
    / "capsule_quality_comparison.png"
)

FAILED_IMAGE = (
    ROOT
    / "dataset_aumentado"
    / "dataset_geral_v2"
    / "imagens"
    / "monai_renal_png__A105609_US_RETROPERITONEAL_COMPLETE_RENAL_ARTERY_DOPPLER_3999566583046568__A105609-1-1024-1gwum7h__frame000.png"
)
SUCCESS_ID = (
    "monai_renal_png__A99505_US_RETROPERITONEAL_COMPLETE_RENAL_ARTERY_DOPPLER_"
    "2229110238275594__A99505-1-2-1g3gq79__frame000.png"
)
SUCCESS_IMAGE = (
    ROOT / "dataset_aumentado" / "dataset_geral_v2" / "imagens" / SUCCESS_ID
)
UNET_MASK = (
    ROOT / "dataset_aumentado" / "dataset_geral_v2" / "mascaras" / SUCCESS_ID
)
DEEPLAB_MASK = (
    ROOT
    / "results"
    / "segmentation_experiments"
    / "capsule_unet_deeplab_consensus"
    / "deeplab_masks"
    / SUCCESS_ID
)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def contour(mask: np.ndarray) -> list[np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


failed = read_gray(FAILED_IMAGE)
success = read_gray(SUCCESS_IMAGE)
unet = read_gray(UNET_MASK)
deeplab = read_gray(DEEPLAB_MASK)

overlay = cv2.cvtColor(success, cv2.COLOR_GRAY2RGB)
cv2.drawContours(overlay, contour(unet), -1, (0, 230, 80), 4)
cv2.drawContours(overlay, contour(deeplab), -1, (255, 70, 60), 3)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15))

axes[0].imshow(failed, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("(a) Sem segmentação", fontweight="bold")

axes[1].imshow(success, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("(b) Imagem segmentada", fontweight="bold")

axes[2].imshow(overlay)
axes[2].set_title("(c) Contornos dos modelos", fontweight="bold")
axes[2].plot([], [], color=(0, 0.9, 0.31), linewidth=3, label="U-Net")
axes[2].plot([], [], color=(1, 0.27, 0.24), linewidth=3, label="DeepLabV3")
axes[2].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    framealpha=0.86,
    fontsize=8,
)

for axis in axes:
    axis.axis("off")

fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.01, wspace=0.025)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, dpi=220, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(FIGURE)
