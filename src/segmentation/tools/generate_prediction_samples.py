import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader
from src.segmentation.core.dataset import KidneyDataset


# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256
N_SAMPLES = 10

DATASET_PATH = str(
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
)
MODELS_PATH = "models"

SAVE_PATH = "results/prediction_samples"
os.makedirs(SAVE_PATH, exist_ok=True)


# =========================
# DATASET
# =========================

dataset = KidneyDataset(
    os.path.join(DATASET_PATH, "test"),
    IMG_SIZE
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)


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

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

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

def load_deeplab():

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None
    )

    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    model.load_state_dict(
        torch.load(os.path.join(MODELS_PATH, "deeplab_best.pth"), map_location=DEVICE)
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def load_unet():

    model = UNet()

    model.load_state_dict(
        torch.load(os.path.join(MODELS_PATH, "unet_best.pth"), map_location=DEVICE)
    )

    model = model.to(DEVICE)
    model.eval()

    return model


deeplab = load_deeplab()
unet = load_unet()


# =========================
# GENERATE SAMPLES
# =========================

count = 0

for img, mask in loader:

    if count >= N_SAMPLES:
        break

    img = img.to(DEVICE)

    with torch.no_grad():

        pred_deeplab = deeplab(img)["out"]
        pred_unet = unet(img)

    pred_deeplab = (torch.sigmoid(pred_deeplab) > 0.5).cpu().numpy()[0,0]
    pred_unet = (torch.sigmoid(pred_unet) > 0.5).cpu().numpy()[0,0]

    img_np = img.cpu().numpy()[0,0]
    mask_np = mask.numpy()[0]

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Ground Truth")
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("DeepLab")
    plt.imshow(pred_deeplab, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("UNet")
    plt.imshow(pred_unet, cmap="gray")
    plt.axis("off")

    save_file = os.path.join(SAVE_PATH, f"sample_{count}.png")

    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

    count += 1


print("Samples saved in:", SAVE_PATH)

