import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.segmentation_training import (
    add_training_args,
    build_training_config,
    train_segmentation_model,
)


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

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.down1 = DoubleConv(in_channels, c1)
        self.down2 = DoubleConv(c1, c2)
        self.down3 = DoubleConv(c2, c3)
        self.down4 = DoubleConv(c3, c4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c4, c5)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.conv4 = DoubleConv(c4 * 2, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.conv3 = DoubleConv(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.conv2 = DoubleConv(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.conv1 = DoubleConv(c1 * 2, c1)

        self.final = nn.Conv2d(c1, out_channels, kernel_size=1)

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


def build_model(base_channels=64):
    return UNet(in_channels=3, out_channels=1, base_channels=base_channels)


def parse_args():

    parser = argparse.ArgumentParser(description="Treino configuravel da U-Net para segmentacao renal.")
    add_training_args(parser, default_model_name="UNet", default_checkpoint_name="unet_best.pth", default_experiment_name="unet_baseline")
    parser.add_argument("--base-channels", type=int, default=64)

    return parser.parse_args()


def main():

    args = parse_args()

    config = build_training_config(
        args,
        model_kind="plain",
        model_kwargs={"base_channels": args.base_channels}
    )

    train_segmentation_model(build_model, config)


if __name__ == "__main__":
    main()
