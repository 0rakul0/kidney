import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.segmentation_training import (
    add_training_args,
    build_training_config,
    train_segmentation_model,
)


# =========================
# BUILDING BLOCK
# =========================

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =========================
# U-NET++ ARCHITECTURE
# =========================

class UNetPlusPlus(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):

        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, c1)
        self.conv1_0 = DoubleConv(c1, c2)
        self.conv2_0 = DoubleConv(c2, c3)
        self.conv3_0 = DoubleConv(c3, c4)

        self.conv0_1 = DoubleConv(c1 + c2, c1)
        self.conv1_1 = DoubleConv(c2 + c3, c2)
        self.conv2_1 = DoubleConv(c3 + c4, c3)

        self.conv0_2 = DoubleConv(c1 * 2 + c2, c1)
        self.conv1_2 = DoubleConv(c2 * 2 + c3, c2)

        self.conv0_3 = DoubleConv(c1 * 3 + c2, c1)

        self.final = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        return self.final(x0_3)


def build_model(base_channels=64):
    return UNetPlusPlus(in_channels=3, out_channels=1, base_channels=base_channels)


def parse_args():

    parser = argparse.ArgumentParser(description="Treino configuravel da U-Net++ para segmentacao renal.")
    add_training_args(
        parser,
        default_model_name="UNet++",
        default_checkpoint_name="unetplusplus_best.pth",
        default_experiment_name="unetplusplus_baseline"
    )
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

