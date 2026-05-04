import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import random

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256
DATASET_PATH = "dataset/test"

DEEPLAB_MODEL = "models/deeplab_best.pth"
UNET_MODEL = "models/unet_best.pth"

N_SAMPLES = 10


# =========================
# U-NET ARCHITECTURE
# =========================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        bottleneck = self.bottleneck(self.pool(d4))

        up4 = self.up4(bottleneck)
        up4 = torch.cat([up4, d4], dim=1)
        up4 = self.conv4(up4)

        up3 = self.up3(up4)
        up3 = torch.cat([up3, d3], dim=1)
        up3 = self.conv3(up3)

        up2 = self.up2(up3)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.conv2(up2)

        up1 = self.up1(up2)
        up1 = torch.cat([up1, d1], dim=1)
        up1 = self.conv1(up1)

        return self.final(up1)


# =========================
# LOAD MODELS
# =========================

# DeepLab
deeplab = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=None,
    aux_loss=True
)

deeplab.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

deeplab.load_state_dict(torch.load(DEEPLAB_MODEL, map_location=DEVICE))

deeplab = deeplab.to(DEVICE)
deeplab.eval()


# UNet
unet = UNet()

unet.load_state_dict(torch.load(UNET_MODEL, map_location=DEVICE))

unet = unet.to(DEVICE)
unet.eval()


# =========================
# DATA
# =========================

img_dir = os.path.join(DATASET_PATH, "image")
mask_dir = os.path.join(DATASET_PATH, "mask")

images = sorted(os.listdir(img_dir))

samples = random.sample(images, N_SAMPLES)


# =========================
# FUNÃ‡Ã•ES
# =========================

def preparar_imagem(img):

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img_3c = np.stack([img, img, img], axis=0)

    tensor = torch.tensor(img_3c, dtype=torch.float32).unsqueeze(0)

    return tensor


def overlay_mask(img, mask, color):

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    mask_rgb = np.zeros_like(img_color)

    mask_rgb[mask == 1] = color

    overlay = cv2.addWeighted(img_color, 1.0, mask_rgb, 0.5, 0)

    return overlay


def criar_heatmap(img, prob_map):

    heatmap = (prob_map * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(img_color, 0.6, heatmap, 0.4, 0)

    return overlay

# =========================
# VISUALIZAÃ‡ÃƒO
# =========================

for img_name in samples:

    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    mask = (mask > 0).astype(np.uint8)

    img_tensor = preparar_imagem(img).to(DEVICE)

    with torch.no_grad():

        pred_deeplab = deeplab(img_tensor)["out"]
        pred_unet = unet(img_tensor)

    prob_deeplab = torch.sigmoid(pred_deeplab).cpu().numpy()[0,0]
    prob_unet = torch.sigmoid(pred_unet).cpu().numpy()[0,0]

    heatmap_deeplab = criar_heatmap(img, prob_deeplab)
    heatmap_unet = criar_heatmap(img, prob_unet)

    plt.figure(figsize=(14,4))

    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Ground Truth")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("DeepLab Heatmap")
    plt.imshow(heatmap_deeplab)
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("UNet Heatmap")
    plt.imshow(heatmap_unet)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

