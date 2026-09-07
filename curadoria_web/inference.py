"""Inferência local da cascata renal para imagens enviadas pela interface web."""

from __future__ import annotations

import base64
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter


LAYER_COLORS = {
    "rim": (243, 79, 86, 235),
    "cortex": (65, 203, 208, 235),
    "medulla": (255, 219, 59, 245),
    "central_echo_complex": (255, 145, 0, 245),
}


class InferenceService:
    """Mantém os pesos carregados e salva resultados locais auditáveis."""

    def __init__(self, project_root: Path, output_dir: Path, brightness_meter):
        self.project_root = project_root
        self.output_dir = output_dir
        self.brightness_meter = brightness_meter
        self.results = {}
        self._models = None

    def _load_models(self):
        if self._models is not None:
            return self._models
        try:
            import torch
            from src.segmentation.build_dataset_geral import (
                keep_largest_component,
                predict_probability,
                prepare_tensor,
            )
            from src.segmentation.core.checkpoint_metadata import load_checkpoint_metadata
            from src.segmentation.core.model_loader import load_model_bundle
            from src.segmentation.experiments.train_inner_deeplab import CLASS_NAMES
            from src.segmentation.experiments.train_unet import UNet
        except ImportError as error:
            raise RuntimeError(
                "A inferência exige PyTorch e as dependências dos modelos instaladas neste ambiente."
            ) from error

        device = "cuda" if torch.cuda.is_available() else "cpu"
        kidney_checkpoint = self.project_root / "models" / "kidneyus_capsule_unet.pth"
        inner_checkpoint = self.project_root / "models" / "intrarenal_unet_multiclass_annotator1.pth"
        if not kidney_checkpoint.exists() or not inner_checkpoint.exists():
            raise RuntimeError("Os pesos de cápsula ou de segmentação intrarrenal não foram encontrados.")
        kidney = load_model_bundle("unet", device=device, checkpoint_path=kidney_checkpoint)
        metadata = load_checkpoint_metadata(inner_checkpoint)
        config = metadata.get("config", {})
        inner_size = int(config.get("img_size", 256))
        inner = UNet(
            in_channels=3,
            out_channels=len(CLASS_NAMES),
            base_channels=int(config.get("base_channels", 64)),
        ).to(device)
        inner.load_state_dict(torch.load(inner_checkpoint, map_location=device))
        self._models = {
            "torch": torch,
            "device": device,
            "kidney": kidney,
            "inner": inner.eval(),
            "inner_size": inner_size,
            "keep_largest_component": keep_largest_component,
            "predict_probability": predict_probability,
            "prepare_tensor": prepare_tensor,
        }
        return self._models

    @staticmethod
    def _bounds(mask, pad_ratio=0.12):
        ys, xs = np.where(mask > 0)
        if not xs.size:
            return None
        width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        padding = max(4, int(round(max(width, height) * pad_ratio)))
        return (
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(mask.shape[1], int(xs.max()) + padding + 1),
            min(mask.shape[0], int(ys.max()) + padding + 1),
        )

    @staticmethod
    def _decode_image(data_url):
        if not isinstance(data_url, str) or "," not in data_url:
            raise ValueError("Envie uma imagem PNG, JPG ou JPEG válida.")
        _, encoded = data_url.split(",", 1)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("O arquivo enviado não contém uma imagem válida.") from error
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError("A imagem excede o limite de 20 MB.")
        try:
            with Image.open(BytesIO(raw)) as source:
                return source.convert("L")
        except (OSError, ValueError) as error:
            raise ValueError("Não foi possível abrir a imagem enviada.") from error

    def infer(self, data_url, filename="imagem"):
        image = self._decode_image(data_url)
        models = self._load_models()
        image_id = f"infer_{uuid.uuid4().hex[:12]}"
        result_dir = self.output_dir / image_id
        masks_dir = result_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        image_path = result_dir / "original.png"
        image.save(image_path, format="PNG")
        pixels = np.asarray(image, dtype=np.uint8)

        tensor = models["prepare_tensor"](pixels, 256, models["device"], clahe=True)
        with models["torch"].no_grad():
            probability = models["predict_probability"](models["kidney"], tensor, 256)
        probability = cv2.resize(probability, (pixels.shape[1], pixels.shape[0]), interpolation=cv2.INTER_LINEAR)
        kidney = models["keep_largest_component"](probability >= float(models["kidney"]["threshold"]))
        if not kidney.any():
            raise RuntimeError("O modelo não identificou uma ROI renal nesta imagem.")

        bounds = self._bounds(kidney)
        x1, y1, x2, y2 = bounds
        roi_image, roi_kidney = pixels[y1:y2, x1:x2], kidney[y1:y2, x1:x2]
        size = models["inner_size"]
        resized = cv2.resize(roi_image, (size, size), interpolation=cv2.INTER_LINEAR)
        resized_kidney = cv2.resize(roi_kidney, (size, size), interpolation=cv2.INTER_NEAREST)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
        normalized = clahe.astype(np.float32) / 255.0
        kidney_channel = (resized_kidney > 0).astype(np.float32)
        channels = np.stack([normalized, normalized * kidney_channel, kidney_channel], axis=0)
        inner_tensor = models["torch"].tensor(channels, dtype=models["torch"].float32).unsqueeze(0).to(models["device"])
        with models["torch"].no_grad():
            roi_labels = models["inner"](inner_tensor).argmax(dim=1).cpu().numpy()[0].astype(np.uint8)
        roi_labels = cv2.resize(roi_labels, (roi_image.shape[1], roi_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        roi_labels[roi_kidney == 0] = 0
        labels = np.zeros_like(kidney, dtype=np.uint8)
        labels[y1:y2, x1:x2] = roi_labels

        mask_paths = {"rim": masks_dir / "rim.png"}
        Image.fromarray((kidney > 0).astype(np.uint8) * 255).save(mask_paths["rim"])
        class_ids = {"cortex": 1, "medulla": 2, "central_echo_complex": 3}
        for name, class_id in class_ids.items():
            path = masks_dir / f"{name}.png"
            Image.fromarray((labels == class_id).astype(np.uint8) * 255).save(path)
            mask_paths[name] = path
        meter = self.brightness_meter(image_path, mask_paths)
        self.results[image_id] = {
            "image_path": image_path,
            "mask_paths": mask_paths,
            "filename": Path(filename or "imagem").name,
            "meter": meter,
            "width": image.width,
            "height": image.height,
            "device": models["device"],
        }
        return {
            "image_id": image_id,
            "filename": Path(filename or "imagem").name,
            "image_url": f"/api/inferences/{image_id}/image",
            "layers": {name: f"/api/inferences/{image_id}/{name}" for name in mask_paths},
            "medidor_brilho": meter,
            "info": {"dimensao": f"{image.width} x {image.height}", "dispositivo": models["device"]},
            "aviso": "Máscaras preditas para revisão; esta tela não produz diagnóstico clínico.",
        }

    def media(self, image_id, kind):
        result = self.results.get(image_id)
        if not result:
            return None
        if kind == "image":
            return result["image_path"].read_bytes(), "image/png"
        mask_path = result["mask_paths"].get(kind)
        if not mask_path:
            return None
        with Image.open(result["image_path"]) as image, Image.open(mask_path) as mask:
            binary = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
            boundary = ImageChops.difference(binary, binary.filter(ImageFilter.MinFilter(3)))
            overlay = Image.new("RGBA", image.size, LAYER_COLORS[kind])
            overlay.putalpha(boundary.point(lambda value: 190 if value > 0 else 0))
            output = BytesIO()
            overlay.save(output, format="PNG")
        return output.getvalue(), "image/png"
