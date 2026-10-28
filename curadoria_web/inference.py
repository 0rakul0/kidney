"""Inferência local da cascata renal para imagens enviadas pela interface web."""

from __future__ import annotations

import base64
import json
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter

try:
    from image_validation import decode_image
except ImportError:  # package import in tests and tooling
    from curadoria_web.image_validation import decode_image


LAYER_COLORS = {
    "rim": (243, 79, 86, 235),
    "cortex": (65, 203, 208, 235),
    "medulla": (255, 219, 59, 245),
    "central_echo_complex": (255, 145, 0, 245),
}


class InferenceService:
    """Mantém os pesos carregados e salva resultados locais auditáveis."""

    REGION_NAMES = {
        "cortex": "Córtex",
        "medulla": "Medula",
        "central_echo_complex": "CEC",
    }
    CONFIDENCE_REVIEW_THRESHOLD = 0.70
    CONFIDENCE_HIGH_PRIORITY_THRESHOLD = 0.55
    MIN_REGION_AREA_PERCENT = 1.0

    def __init__(self, project_root: Path, output_dir: Path, brightness_meter):
        self.project_root = project_root
        self.output_dir = output_dir
        self.brightness_meter = brightness_meter
        self.results = {}
        self._models = None
        self._restore_results()

    def _restore_results(self):
        """Rebuild the inference index after a server restart."""
        if not self.output_dir.exists():
            return
        for metadata_path in self.output_dir.glob("infer_*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                image_id = str(metadata["image_id"])
                result_dir = metadata_path.parent
                image_path = result_dir / "original.png"
                mask_paths = {
                    name: result_dir / "masks" / f"{name}.png"
                    for name in ("rim", "cortex", "medulla", "central_echo_complex")
                }
                uncertainty_path = result_dir / "uncertainty.png"
                if not image_path.exists() or not all(path.exists() for path in mask_paths.values()):
                    continue
                active_mask_paths = {}
                for name, path in metadata.get("active_mask_paths", {}).items():
                    candidate = Path(path)
                    if candidate.exists() and name in mask_paths:
                        active_mask_paths[name] = candidate
                        mask_paths[name] = candidate
                self.results[image_id] = {
                    "image_path": image_path,
                    "mask_paths": mask_paths,
                    "filename": str(metadata.get("filename", image_id)),
                    "meter": metadata.get("meter", {}),
                    "width": int(metadata["width"]),
                    "height": int(metadata["height"]),
                    "device": str(metadata.get("device", "unknown")),
                    "capsule_bbox": metadata.get("capsule_bbox"),
                    "quality": metadata.get("quality", {}),
                    "region_review": metadata.get("region_review", []),
                    "uncertainty_path": uncertainty_path if uncertainty_path.exists() else None,
                    "active_mask_paths": active_mask_paths,
                }
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    @staticmethod
    def _write_metadata(result_dir: Path, result: dict):
        metadata_path = result_dir / "metadata.json"
        temporary_path = result_dir / "metadata.json.tmp"
        payload = {
            "schema_version": 1,
            "image_id": result["image_id"],
            "filename": result["filename"],
            "meter": result["meter"],
            "width": result["width"],
            "height": result["height"],
            "device": result["device"],
            "capsule_bbox": result["capsule_bbox"],
            "quality": result.get("quality", {}),
            "region_review": result.get("region_review", []),
            "active_mask_paths": {name: str(path) for name, path in result.get("active_mask_paths", {}).items()},
        }
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(metadata_path)

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

    @classmethod
    def _assess_regions(cls, labels, probabilities, kidney_mask):
        """Estimate operational review priority from segmentation uncertainty.

        These scores describe model uncertainty and mask geometry only. They are
        not calibrated probabilities of disease and must not be interpreted as
        clinical risk.
        """
        kidney_pixels = int(kidney_mask.sum())
        assessments = []
        for name, class_id in (("cortex", 1), ("medulla", 2), ("central_echo_complex", 3)):
            mask = labels == class_id
            pixels = int(mask.sum())
            label = cls.REGION_NAMES[name]
            reasons = []
            if not pixels:
                assessments.append({
                    "regiao": name,
                    "rotulo": label,
                    "pixels": 0,
                    "area_rim_pct": 0.0,
                    "confianca": 0.0,
                    "incerteza": 1.0,
                    "prioridade": "alta",
                    "pontuacao_revisao": 1.0,
                    "motivos": ["nenhuma região foi predita"],
                    "acao": "revisar a máscara e a qualidade da imagem",
                })
                continue

            class_probability = probabilities[class_id][mask]
            confidence = float(np.clip(class_probability.mean(), 0.0, 1.0))
            uncertainty = float(np.clip(1.0 - confidence, 0.0, 1.0))
            area_percent = float(pixels / max(kidney_pixels, 1) * 100.0)
            if confidence < cls.CONFIDENCE_HIGH_PRIORITY_THRESHOLD:
                reasons.append("baixa confiança da segmentação")
            elif confidence < cls.CONFIDENCE_REVIEW_THRESHOLD:
                reasons.append("confiança intermediária da segmentação")
            if area_percent < cls.MIN_REGION_AREA_PERCENT:
                reasons.append("região pequena em relação ao rim")

            score = uncertainty
            if area_percent < cls.MIN_REGION_AREA_PERCENT:
                score += 0.20
            score = float(np.clip(score, 0.0, 1.0))
            if score >= 0.45:
                priority = "alta"
            elif score >= 0.25:
                priority = "média"
            else:
                priority = "baixa"
            assessments.append({
                "regiao": name,
                "rotulo": label,
                "pixels": pixels,
                "area_rim_pct": round(area_percent, 2),
                "confianca": round(confidence, 4),
                "incerteza": round(uncertainty, 4),
                "prioridade": priority,
                "pontuacao_revisao": round(score, 4),
                "motivos": reasons or ["sem alerta automático de confiança ou área"],
                "acao": "inspecionar visualmente a máscara antes de usar as métricas",
            })
        return assessments

    @classmethod
    def _quality_assessment(cls, kidney_probability, kidney_mask, bbox, region_review):
        foreground = kidney_probability[kidney_mask]
        background = kidney_probability[~kidney_mask]
        confidence_parts = []
        if foreground.size:
            confidence_parts.append(float(foreground.mean()))
        if background.size:
            confidence_parts.append(float((1.0 - background).mean()))
        confidence = float(np.clip(np.mean(confidence_parts), 0.0, 1.0)) if confidence_parts else 0.0
        image_area = kidney_mask.shape[0] * kidney_mask.shape[1]
        kidney_area_percent = float(kidney_mask.sum() / max(image_area, 1) * 100.0)
        alerts = []
        if confidence < cls.CONFIDENCE_HIGH_PRIORITY_THRESHOLD:
            alerts.append("baixa confiança global da máscara renal")
        elif confidence < cls.CONFIDENCE_REVIEW_THRESHOLD:
            alerts.append("confiança intermediária da máscara renal")
        if kidney_area_percent < 1.0:
            alerts.append("área renal pequena em relação à imagem")
        if bbox:
            bbox_width = bbox["x2"] - bbox["x1"]
            bbox_height = bbox["y2"] - bbox["y1"]
            if min(bbox_width, bbox_height) < 32:
                alerts.append("recorte renal pequeno para inspeção")
        for region in region_review:
            if region["prioridade"] == "alta":
                alerts.append(f"{region['rotulo']}: {region['motivos'][0]}")
        return {
            "status": "revisar_com_atencao" if alerts else "sem_alerta_automatico",
            "confianca_rim": round(confidence, 4),
            "area_rim_pct_imagem": round(kidney_area_percent, 2),
            "alertas": alerts,
            "revisao_humana_obrigatoria": True,
            "interpretacao": "confiança operacional da segmentação; não é probabilidade diagnóstica",
        }

    @staticmethod
    def _uncertainty_overlay(probabilities, kidney_mask):
        max_probability = probabilities.max(axis=0)
        uncertainty = np.clip(1.0 - max_probability, 0.0, 1.0)
        values = (uncertainty * 255).astype(np.uint8)
        heatmap = cv2.cvtColor(cv2.applyColorMap(values, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        alpha = np.where(kidney_mask, 178, 0).astype(np.uint8)
        return Image.fromarray(np.dstack([heatmap, alpha]), mode="RGBA")

    @staticmethod
    def _largest_component(binary):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        if count <= 1:
            return binary.astype(bool)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    @classmethod
    def _refine_mask(cls, source, kidney_mask, operation):
        binary = np.asarray(source.convert("L")) > 0
        # Em imagens de ultrassom de 1024 x 768, um kernel fixo de 5 px
        # costuma alterar poucos pixels e não produz uma proposta visualmente
        # auditável. A escala acompanha a menor dimensão, com limite para não
        # deformar regiões anatômicas grandes.
        scale = max(5, int(round(min(binary.shape) * 0.01)))
        if scale % 2 == 0:
            scale += 1
        kernel_size = min(scale, 11)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if operation == "remover_ruido":
            kernel = np.ones((5, 5), dtype=np.uint8)
            refined = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
            refined = cls._largest_component(refined)
        elif operation == "preencher_vazios":
            closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            flood = (closed * 255).astype(np.uint8)
            flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
            cv2.floodFill(flood, flood_mask, (0, 0), 255)
            holes = cv2.bitwise_not(flood)
            refined = (closed | holes) > 0
        else:
            refined = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel) > 0
        if kidney_mask is not None and source.size == kidney_mask.shape[::-1]:
            refined &= kidney_mask
        return Image.fromarray((refined.astype(np.uint8) * 255), mode="L"), refined, kernel.shape[0]

    def propose_mask(self, image_id, layer, operation, reviewer):
        if not reviewer.strip():
            raise ValueError("Informe o revisor antes de gerar uma proposta de máscara.")
        if layer not in {"rim", "cortex", "medulla", "central_echo_complex"}:
            raise ValueError("Estrutura de máscara inválida.")
        if operation not in {"refinar_mascara", "preencher_vazios", "remover_ruido", "suavizar_bordas"}:
            raise ValueError("Operação de máscara inválida.")
        result = self.results.get(image_id.strip())
        if not result:
            raise ValueError("Resultado de inferência não encontrado.")
        source_path = result["mask_paths"].get(layer)
        if not source_path or not source_path.exists():
            raise ValueError("Máscara selecionada não está disponível.")
        kidney_path = result["mask_paths"].get("rim")
        kidney_mask = None
        if kidney_path and kidney_path.exists():
            with Image.open(kidney_path) as kidney_image:
                kidney_mask = np.asarray(kidney_image.convert("L")) > 0
        with Image.open(source_path) as source:
            proposal, binary, kernel_size = self._refine_mask(source, kidney_mask, operation)
            original = np.asarray(source.convert("L")) > 0
            proposal_id = f"{layer}_{operation}_{uuid.uuid4().hex[:8]}"
        proposal_dir = result["image_path"].parent / "proposals"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = proposal_dir / f"{proposal_id}.png"
        manifest_path = proposal_dir / f"{proposal_id}.json"
        proposal.save(proposal_path, format="PNG")
        metadata = {
            "proposal_id": proposal_id,
            "image_id": image_id,
            "layer": layer,
            "operation": operation,
            "reviewer": reviewer.strip(),
            "source_path": str(source_path),
            "proposal_path": str(proposal_path),
            "added_pixels": int(np.logical_and(binary, ~original).sum()),
            "removed_pixels": int(np.logical_and(original, ~binary).sum()),
            "kernel_size": kernel_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pendente",
        }
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            **metadata,
            "proposal_url": f"/api/inferences/{image_id}/proposals/{proposal_id}",
            "requires_confirmation": True,
            "message": "Proposta gerada sem substituir a máscara original. Confirme para aplicá-la à sessão.",
        }

    def proposal_media(self, image_id, proposal_id):
        if not re.fullmatch(r"[A-Za-z0-9_]+", str(proposal_id)):
            return None
        result = self.results.get(image_id)
        if not result:
            return None
        proposal_dir = result["image_path"].parent / "proposals"
        proposal_path = proposal_dir / f"{proposal_id}.png"
        manifest_path = proposal_dir / f"{proposal_id}.json"
        if not proposal_path.exists() or not manifest_path.exists():
            return None
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = Path(metadata["source_path"])
        with Image.open(source_path) as source_image, Image.open(proposal_path) as proposal_image:
            original = np.asarray(source_image.convert("L")) > 0
            proposed = np.asarray(proposal_image.convert("L")) > 0
        additions = np.logical_and(proposed, ~original)
        removals = np.logical_and(original, ~proposed)
        diff = np.zeros((*original.shape, 4), dtype=np.uint8)
        diff[additions] = (34, 197, 94, 235)  # verde: pixels adicionados
        diff[removals] = (239, 68, 68, 235)  # vermelho: pixels removidos
        output = BytesIO()
        Image.fromarray(diff, mode="RGBA").save(output, format="PNG")
        return output.getvalue(), "image/png"

    def apply_mask_proposal(self, image_id, proposal_id, reviewer):
        if not reviewer.strip():
            raise ValueError("Informe o revisor antes de aplicar a proposta.")
        result = self.results.get(image_id.strip())
        if not result:
            raise ValueError("Resultado de inferência não encontrado.")
        if not re.fullmatch(r"[A-Za-z0-9_]+", str(proposal_id)):
            raise ValueError("Identificador de proposta inválido.")
        proposal_dir = result["image_path"].parent / "proposals"
        manifest_path = proposal_dir / f"{proposal_id}.json"
        if not manifest_path.exists():
            raise ValueError("Proposta de máscara não encontrada.")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "aplicada":
            raise ValueError("Esta proposta já foi aplicada.")
        proposal_path = Path(metadata["proposal_path"])
        if not proposal_path.resolve().is_relative_to(proposal_dir.resolve()):
            raise ValueError("Arquivo da proposta fora do diretório permitido.")
        if not proposal_path.exists():
            raise ValueError("Arquivo da proposta não encontrado.")
        layer = metadata["layer"]
        approved_dir = result["image_path"].parent / "approved_masks"
        approved_dir.mkdir(parents=True, exist_ok=True)
        approved_path = approved_dir / f"{layer}.png"
        approved_path.write_bytes(proposal_path.read_bytes())
        result["mask_paths"][layer] = approved_path
        result.setdefault("active_mask_paths", {})[layer] = approved_path
        if callable(self.brightness_meter):
            result["meter"] = self.brightness_meter(result["image_path"], result["mask_paths"])
        metadata.update({
            "status": "aplicada",
            "approved_by": reviewer.strip(),
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approved_path": str(approved_path),
        })
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_metadata(result["image_path"].parent, result)
        return {
            "proposal_id": proposal_id,
            "image_id": image_id,
            "layer": layer,
            "status": "aplicada",
            "layers": {name: f"/api/inferences/{image_id}/{name}" for name in result["mask_paths"]},
            "message": "Proposta aplicada à sessão; a máscara original permanece preservada.",
        }

    @staticmethod
    def _decode_image(data_url):
        if not isinstance(data_url, str) or "," not in data_url:
            raise ValueError("Envie uma imagem PNG, JPG ou JPEG válida.")
        _, encoded = data_url.split(",", 1)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("O arquivo enviado não contém uma imagem válida.") from error
        try:
            return decode_image(raw)
        except ValueError as error:
            raise ValueError(str(error)) from error

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
            inner_logits = models["inner"](inner_tensor)
            inner_probabilities = models["torch"].softmax(inner_logits, dim=1).cpu().numpy()[0]
        roi_probabilities = np.stack(
            [cv2.resize(channel, (roi_image.shape[1], roi_image.shape[0]), interpolation=cv2.INTER_LINEAR) for channel in inner_probabilities]
        )
        roi_probabilities = np.clip(roi_probabilities, 0.0, 1.0)
        roi_probabilities[:, roi_kidney == 0] = 0.0
        roi_labels = np.argmax(roi_probabilities, axis=0).astype(np.uint8)
        roi_labels[roi_kidney == 0] = 0
        labels = np.zeros_like(kidney, dtype=np.uint8)
        labels[y1:y2, x1:x2] = roi_labels
        full_probabilities = np.zeros((roi_probabilities.shape[0], pixels.shape[0], pixels.shape[1]), dtype=np.float32)
        full_probabilities[:, y1:y2, x1:x2] = roi_probabilities

        mask_paths = {"rim": masks_dir / "rim.png"}
        Image.fromarray((kidney > 0).astype(np.uint8) * 255).save(mask_paths["rim"])
        class_ids = {"cortex": 1, "medulla": 2, "central_echo_complex": 3}
        for name, class_id in class_ids.items():
            path = masks_dir / f"{name}.png"
            Image.fromarray((labels == class_id).astype(np.uint8) * 255).save(path)
            mask_paths[name] = path
        meter = self.brightness_meter(image_path, mask_paths)
        region_review = self._assess_regions(labels, full_probabilities, kidney)
        quality = self._quality_assessment(probability, kidney, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, region_review)
        uncertainty_path = result_dir / "uncertainty.png"
        self._uncertainty_overlay(full_probabilities, kidney).save(uncertainty_path, format="PNG")
        persisted_result = {
            "image_id": image_id,
            "image_path": image_path,
            "mask_paths": mask_paths,
            "filename": Path(filename or "imagem").name,
            "meter": meter,
            "width": image.width,
            "height": image.height,
            "device": models["device"],
            "capsule_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "quality": quality,
            "region_review": region_review,
            "uncertainty_path": uncertainty_path,
        }
        self.results[image_id] = persisted_result
        self._write_metadata(result_dir, persisted_result)
        return {
            "image_id": image_id,
            "filename": Path(filename or "imagem").name,
            "image_url": f"/api/inferences/{image_id}/image",
            "layers": {
                **{name: f"/api/inferences/{image_id}/{name}" for name in mask_paths},
                "uncertainty": f"/api/inferences/{image_id}/uncertainty",
            },
            "medidor_brilho": meter,
            "capsule_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "qualidade": quality,
            "priorizacao_regioes": region_review,
            "info": {"dimensao": f"{image.width} x {image.height}", "dispositivo": models["device"]},
            "aviso": "Máscaras preditas para revisão; esta tela não produz diagnóstico clínico.",
        }

    def media(self, image_id, kind):
        result = self.results.get(image_id)
        if not result:
            return None
        if kind == "image":
            return result["image_path"].read_bytes(), "image/png"
        if kind == "uncertainty":
            uncertainty_path = result.get("uncertainty_path")
            if uncertainty_path and uncertainty_path.exists():
                return uncertainty_path.read_bytes(), "image/png"
            return None
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
