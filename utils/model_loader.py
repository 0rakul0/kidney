from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from transformers import SegformerForSemanticSegmentation

from experiments.train_unet import UNet
from experiments.train_unetplusplus import UNetPlusPlus
from utils.checkpoint_metadata import load_checkpoint_metadata


DEFAULT_CHECKPOINTS = {
    "unet": "unet_best.pth",
    "unetplusplus": "unetplusplus_best.pth",
    "deeplab": "deeplab_best.pth",
    "segformer": "segformer_best.pth",
}

DISPLAY_NAMES = {
    "unet": "UNet",
    "unetplusplus": "UNet++",
    "deeplab": "DeepLab",
    "segformer": "SegFormer",
}

ALIASES = {
    "unet": "unet",
    "u-net": "unet",
    "unetplusplus": "unetplusplus",
    "unet++": "unetplusplus",
    "deeplab": "deeplab",
    "deeplabv3": "deeplab",
    "segformer": "segformer",
}


def normalize_model_name(name):
    key = name.strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Modelo nao suportado: {name}")
    return ALIASES[key]


def list_supported_models():
    return list(DEFAULT_CHECKPOINTS.keys())


def get_display_name(name):
    return DISPLAY_NAMES[normalize_model_name(name)]


def resolve_checkpoint_path(model_name, checkpoint_path=None, model_dir="models"):
    if checkpoint_path:
        return Path(checkpoint_path)

    normalized = normalize_model_name(model_name)
    return Path(model_dir) / DEFAULT_CHECKPOINTS[normalized]


def load_model_bundle(
    model_name,
    device,
    checkpoint_path=None,
    model_dir="models",
    deeplab_backbone=None,
    segformer_backbone=None,
):
    normalized = normalize_model_name(model_name)
    resolved_checkpoint = resolve_checkpoint_path(
        normalized,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
    )
    metadata = load_checkpoint_metadata(resolved_checkpoint)
    model_kwargs = metadata.get("model_kwargs", {})

    if normalized == "unet":
        model = UNet(**model_kwargs)

    elif normalized == "unetplusplus":
        model = UNetPlusPlus(**model_kwargs)

    elif normalized == "deeplab":
        backbone = deeplab_backbone or model_kwargs.get("backbone", "resnet50")
        builders = {
            "resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
            "resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
        }
        if backbone not in builders:
            raise ValueError(f"Backbone DeepLab nao suportado: {backbone}")

        model = builders[backbone](weights=None)
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    else:
        backbone_name = segformer_backbone or model_kwargs.get(
            "backbone_name",
            "nvidia/segformer-b0-finetuned-ade-512-512",
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

    state_dict = torch.load(resolved_checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    threshold = metadata.get("best_threshold", 0.5)

    return {
        "model_key": normalized,
        "display_name": DISPLAY_NAMES[normalized],
        "model": model.to(device).eval(),
        "threshold": threshold,
        "checkpoint_path": resolved_checkpoint,
        "metadata": metadata,
    }
