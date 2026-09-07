"""Servidor local para curadoria visual de mascaras renais.

O aplicativo evita transportar milhares de imagens dentro de uma planilha:
serve os arquivos do manifesto validado, desenha contornos na mesma resolucao
da imagem e persiste uma avaliacao por revisor em SQLite.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from inference import InferenceService


PROJECT_ROOT = Path(os.environ.get("CURATORIA_DATA_ROOT", Path(__file__).resolve().parents[1]))
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "dataset_aumentado" / "curadoria" / "miniaturas_completas" / "manifest.csv"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset_aumentado" / "curadoria" / "respostas"
DATASET_MANIFEST = PROJECT_ROOT / "dataset_aumentado" / "dataset_geral" / "manifest.csv"
MEDULLA_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "medulla_predictions_consensus_v1_dataset_geral"
    / "manifest.csv"
)
MEDULLA_MODEL_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "segmentation_experiments"
    / "medulla_deeplab_resnet50_consensus_v1_summary.json"
)
KIDNEY_MODEL_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "stability_evaluation_consensus_v1"
    / "deeplab"
    / "summary.json"
)
CONSENSUS_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "medulla_consensus_review_v2"
    / "selected_for_review.csv"
)
CORTEX_MODEL_SUMMARY = (
    PROJECT_ROOT / "results" / "intrarenal_model3" / "cortex_roi_unet_annotator1" / "summary.json"
)
INTRARENAL_MULTICLASS_SUMMARY = (
    PROJECT_ROOT / "results" / "intrarenal_model3" / "intrarenal_deeplab_resnet50_multiclass_annotator1" / "summary.json"
)
INTRARENAL_MODEL_MANIFESTS = {
    "deeplab": PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "intrarenal_multiclass_predictions_dataset_geral_unet085"
    / "manifest.csv",
    "unet": PROJECT_ROOT
    / "results"
    / "intrarenal_model3"
    / "intrarenal_unet_multiclass_predictions_dataset_geral"
    / "manifest.csv",
}
INTRARENAL_MODEL_SUMMARIES = {
    "deeplab": INTRARENAL_MULTICLASS_SUMMARY,
    "unet": PROJECT_ROOT / "results" / "intrarenal_model3" / "intrarenal_unet_multiclass_annotator1" / "summary.json",
}
DEFAULT_INTRARENAL_MODEL = "unet"
STATUS_VALUES = {"pendente", "aceita", "corrigir", "rejeitada", "indisponivel"}
LAYER_FIELDS = {
    "rim": ("mascara_rim_visual", (255, 62, 73, 235)),
    "cortex": ("mascara_cortex_visual", (0, 211, 224, 235)),
    "medulla": ("mascara_medulla_visual", (255, 213, 61, 245)),
    "central_echo_complex": ("mascara_central_echo_complex_visual", (255, 145, 0, 245)),
    "anomalia": (None, (184, 97, 255, 245)),
}
REVIEW_STATUS_FIELDS = {
    "rim": "status_rim",
    "cortex": "status_cortex",
    "medulla": "status_medulla",
    "central_echo_complex": "status_central_echo_complex",
}
CORRECTION_APPROVAL_VALUES = {"pendente", "aprovada", "reprovada"}


@dataclass(frozen=True)
class MaskCorrection:
    image_id: str
    reviewer: str
    layer: str
    mask_path: str
    operation: str
    approval_status: str
    created_at: str
    updated_at: str
    approved_at: str = ""
    approved_by: str = ""


class MaskCorrectionService:
    def __init__(self, connect, corrections_dir: Path, image_size_lookup, effective_mask_lookup, safe_name):
        self.connect = connect
        self.corrections_dir = corrections_dir
        self.image_size_lookup = image_size_lookup
        self.effective_mask_lookup = effective_mask_lookup
        self.safe_name = safe_name

    def create_from_polygon(self, payload):
        image_id = str(payload.get("image_id", "")).strip()
        reviewer = str(payload.get("reviewer", "")).strip()
        layer = str(payload.get("layer", "")).strip()
        operation = str(payload.get("operation", "substituir")).strip()
        tool = str(payload.get("tool", "polygon")).strip()
        points = payload.get("points", [])
        if layer not in LAYER_FIELDS:
            raise ValueError("Classe de mascara invalida.")
        if not reviewer:
            raise ValueError("Informe o revisor antes de corrigir uma mascara.")
        if operation not in {"substituir", "adicionar", "apagar"}:
            raise ValueError("Operacao de desenho invalida.")
        if tool not in {"polygon", "brush"}:
            raise ValueError("Ferramenta de desenho invalida.")
        minimum_points = 3 if tool == "polygon" else 1
        if not isinstance(points, list) or len(points) < minimum_points:
            raise ValueError("Desenho insuficiente para gerar a mascara.")
        size = self.image_size_lookup(image_id)
        if size is None:
            raise ValueError("Imagem desconhecida.")
        radius = int(payload.get("radius", 12) or 12)
        radius = min(max(radius, 2), 80)
        coordinates = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Coordenadas de desenho invalidas.")
            x, y = float(point[0]), float(point[1])
            coordinates.append((min(max(x, 0), size[0] - 1), min(max(y, 0), size[1] - 1)))
        previous = self.effective_mask_lookup(image_id, layer, reviewer)
        if operation == "substituir" or previous is None:
            mask = Image.new("L", size, 0)
        else:
            with Image.open(previous) as current:
                if current.size != size:
                    raise ValueError("Mascara existente nao corresponde a imagem exibida.")
                mask = current.convert("L")
        drawer = ImageDraw.Draw(mask)
        fill = 0 if operation == "apagar" else 255
        if tool == "polygon":
            drawer.polygon(coordinates, fill=fill)
        else:
            if len(coordinates) == 1:
                x, y = coordinates[0]
                drawer.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
            else:
                drawer.line(coordinates, fill=fill, width=radius * 2, joint="curve")
                for x, y in coordinates:
                    drawer.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        now = datetime.now(timezone.utc).isoformat()
        destination = (
            self.corrections_dir
            / self.safe_name(reviewer)
            / layer
            / f"{self.safe_name(image_id)}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        mask.save(destination, format="PNG")
        history_destination = (
            self.corrections_dir / self.safe_name(reviewer) / "historico" / layer
            / self.safe_name(image_id) / f"{now.replace(':', '-').replace('+', '_')}.png"
        )
        history_destination.parent.mkdir(parents=True, exist_ok=True)
        mask.save(history_destination, format="PNG")
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM mask_edits WHERE image_id = ? AND reviewer = ? AND layer = ?",
                (image_id, reviewer, layer),
            ).fetchone()
            created_at = existing["created_at"] if existing and existing["created_at"] else now
            connection.execute(
                """
                INSERT OR REPLACE INTO mask_edits (
                    image_id, reviewer, layer, mask_path, operation, approval_status,
                    created_at, updated_at, approved_at, approved_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    reviewer,
                    layer,
                    str(destination),
                    operation,
                    "pendente",
                    created_at,
                    now,
                    "",
                    "",
                ),
            )
            connection.execute(
                """
                INSERT INTO mask_edit_history (image_id, reviewer, layer, mask_path, operation, saved_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (image_id, reviewer, layer, str(history_destination), operation, now),
            )
        return self.read(image_id=image_id, reviewer=reviewer, layer=layer)[0]

    def read(self, image_id="", reviewer="", layer=""):
        where = []
        params = []
        if image_id:
            where.append("image_id = ?")
            params.append(image_id)
        if reviewer:
            where.append("reviewer = ?")
            params.append(reviewer)
        if layer:
            where.append("layer = ?")
            params.append(layer)
        sql = "SELECT * FROM mask_edits"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC, image_id, layer"
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def update_approval(self, image_id, reviewer, layer, approval_status, approved_by=""):
        if layer not in LAYER_FIELDS:
            raise ValueError("Classe de mascara invalida.")
        if approval_status not in CORRECTION_APPROVAL_VALUES:
            raise ValueError("Status de aprovacao invalido.")
        now = datetime.now(timezone.utc).isoformat()
        approved_at = now if approval_status == "aprovada" else ""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT image_id FROM mask_edits WHERE image_id = ? AND reviewer = ? AND layer = ?",
                (image_id, reviewer, layer),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE mask_edits
                SET approval_status = ?, approved_at = ?, approved_by = ?, updated_at = ?
                WHERE image_id = ? AND reviewer = ? AND layer = ?
                """,
                (approval_status, approved_at, approved_by, now, image_id, reviewer, layer),
            )
        return self.read(image_id=image_id, reviewer=reviewer, layer=layer)[0]

    def delete(self, image_id, reviewer, layer):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT mask_path FROM mask_edits WHERE image_id = ? AND reviewer = ? AND layer = ?",
                (image_id, reviewer, layer),
            ).fetchone()
            connection.execute(
                "DELETE FROM mask_edits WHERE image_id = ? AND reviewer = ? AND layer = ?",
                (image_id, reviewer, layer),
            )
        path = Path(row["mask_path"]) if row else None
        if path and path.exists():
            path.unlink()
        return row is not None


class CurationStore:
    def __init__(self, manifest_path: Path, database_path: Path, cache_dir: Path):
        self.manifest_path = manifest_path
        self.database_path = database_path
        self.cache_dir = cache_dir
        self.corrections_dir = database_path.parent / "mascaras_corrigidas"
        self.items = self._load_manifest()
        self.dataset_metadata = self._read_csv_index(DATASET_MANIFEST)
        self.medulla_predictions = self._read_csv_index(MEDULLA_PREDICTIONS)
        self.consensus_selected = self._read_csv_index(CONSENSUS_SELECTED)
        self.medulla_summary = self._read_json(MEDULLA_MODEL_SUMMARY)
        self.kidney_summary = self._read_json(KIDNEY_MODEL_SUMMARY)
        self.cortex_summary = self._read_json(CORTEX_MODEL_SUMMARY)
        self.intrarenal_multiclass_summary = self._read_json(INTRARENAL_MULTICLASS_SUMMARY)
        self.intrarenal_model_predictions = {
            name: self._read_csv_index(path)
            for name, path in INTRARENAL_MODEL_MANIFESTS.items()
        }
        self.intrarenal_model_summaries = {
            name: self._read_json(path)
            for name, path in INTRARENAL_MODEL_SUMMARIES.items()
        }
        self.by_id = {item["image_id"]: item for item in self.items}
        self._prepare_database()
        self.corrections = MaskCorrectionService(
            self.connect,
            self.corrections_dir,
            self.image_size,
            self.effective_mask,
            self._safe_name,
        )

    @staticmethod
    def _read_json(path):
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_csv_index(path):
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {row["image_id"]: row for row in csv.DictReader(handle)}

    @staticmethod
    def _resolve_path(value):
        if not value:
            return None
        # Manifestos criados no Windows podem conter, por exemplo,
        # D:\\kidney\\dataset_aumentado\\... . No contêiner, o conteúdo do
        # projeto é montado em PROJECT_ROOT (/data), então remove-se apenas a
        # unidade e a pasta-raiz original antes de resolver o caminho.
        windows_path = re.match(r"^[A-Za-z]:[\\/][^\\/]+[\\/](.+)$", str(value))
        if windows_path and os.name != "nt":
            path = PROJECT_ROOT / windows_path.group(1).replace("\\", "/")
            return path if path.exists() else None
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path if path.exists() else None

    def _load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifesto visual nao encontrado: {self.manifest_path}. "
                "Execute engenharia_dataset/build_curation_thumbnails.py primeiro."
            )
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"image_id", "origem_visual", "imagem_visual"} | {
            field for field, _ in LAYER_FIELDS.values() if field
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError("O manifesto visual nao contem as colunas esperadas.")
        return rows

    def connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _prepare_database(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    image_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    reviewer_type TEXT NOT NULL,
                    status_rim TEXT NOT NULL,
                    status_cortex TEXT NOT NULL,
                    status_medulla TEXT NOT NULL,
                    status_central_echo_complex TEXT NOT NULL DEFAULT 'pendente',
                    fibrose TEXT NOT NULL,
                    fonte_fibrose TEXT NOT NULL,
                    observacao TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (image_id, reviewer)
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(reviews)").fetchall()
            }
            if "status_central_echo_complex" not in columns:
                connection.execute(
                    "ALTER TABLE reviews ADD COLUMN status_central_echo_complex TEXT NOT NULL DEFAULT 'pendente'"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mask_edits (
                    image_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    mask_path TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    approval_status TEXT NOT NULL DEFAULT 'pendente',
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    approved_at TEXT NOT NULL DEFAULT '',
                    approved_by TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (image_id, reviewer, layer)
                )
                """
            )
            edit_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(mask_edits)").fetchall()
            }
            for column, definition in {
                "approval_status": "TEXT NOT NULL DEFAULT 'pendente'",
                "created_at": "TEXT NOT NULL DEFAULT ''",
                "approved_at": "TEXT NOT NULL DEFAULT ''",
                "approved_by": "TEXT NOT NULL DEFAULT ''",
            }.items():
                if column not in edit_columns:
                    connection.execute(f"ALTER TABLE mask_edits ADD COLUMN {column} {definition}")
            # O nome da camada mudou, mas as marcações já feitas continuam
            # disponíveis como marcadores de anomalia para o mesmo revisor.
            legacy_anomaly_edits = connection.execute(
                "SELECT image_id, reviewer FROM mask_edits WHERE layer = 'fibrose'"
            ).fetchall()
            for edit in legacy_anomaly_edits:
                has_new_layer = connection.execute(
                    "SELECT 1 FROM mask_edits WHERE image_id = ? AND reviewer = ? AND layer = 'anomalia'",
                    (edit["image_id"], edit["reviewer"]),
                ).fetchone()
                if not has_new_layer:
                    connection.execute(
                        "UPDATE mask_edits SET layer = 'anomalia' WHERE image_id = ? AND reviewer = ? AND layer = 'fibrose'",
                        (edit["image_id"], edit["reviewer"]),
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mask_edit_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    mask_path TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS curation_database_export (
                    image_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    reviewer_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    exported_at TEXT NOT NULL,
                    PRIMARY KEY (image_id, reviewer)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS brightness_roi_selections (
                    image_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    pairs_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (image_id, reviewer, model)
                )
                """
            )

    def summary(self, reviewer=""):
        with self.connect() as connection:
            if reviewer:
                reviewed = connection.execute(
                    "SELECT COUNT(*) FROM reviews WHERE reviewer = ?", (reviewer,)
                ).fetchone()[0]
            else:
                reviewed = connection.execute(
                    "SELECT COUNT(DISTINCT image_id) FROM reviews"
                ).fetchone()[0]
        source_counts = {}
        for item in self.items:
            source = item["origem_visual"]
            source_counts[source] = source_counts.get(source, 0) + 1
        return {
            "total": len(self.items),
            "revisados": reviewed,
            "pendentes": len(self.items) - reviewed,
            "origens": source_counts,
            "manifesto": str(self.manifest_path),
        }

    def is_pseudo(self, item):
        dataset_info = self.dataset_metadata.get(item["image_id"], {})
        kidney_generated = dataset_info.get("mask_status", "").startswith("generated")
        medulla_generated = (
            item["origem_visual"] == "dataset_geral_prediction_space"
            and bool(item["mascara_medulla_visual"])
        )
        cortex_generated = (
            item["origem_visual"] == "dataset_geral_prediction_space"
            and bool(item["mascara_cortex_visual"])
        )
        central_generated = (
            item["origem_visual"] == "dataset_geral_prediction_space"
            and bool(item.get("mascara_central_echo_complex_visual", ""))
        )
        return kidney_generated or medulla_generated or cortex_generated or central_generated

    def has_kidney_mask(self, item, reviewer=""):
        if reviewer and self.edited_mask(item["image_id"], "rim", reviewer) is not None:
            return True
        dataset_info = self.dataset_metadata.get(item["image_id"], {})
        if dataset_info:
            return (
                dataset_info.get("has_mask", "").lower() == "true"
                and bool(dataset_info.get("dataset_mask_path", ""))
            )
        return bool(item["mascara_rim_visual"])

    def list_items(self, reviewer="", state="todos", source="", annotation="", search="", limit=150):
        reviews = self.reviews_by_image(reviewer) if reviewer else {}
        normalized_search = search.casefold().strip()
        listed = []
        for item in self.items:
            if self.source_file(item["image_id"], "image") is None:
                continue
            has_review = item["image_id"] in reviews
            if state == "pendentes" and has_review:
                continue
            if state == "revisados" and not has_review:
                continue
            if source and item["origem_visual"] != source:
                continue
            is_pseudo = self.is_pseudo(item)
            has_kidney_mask = self.has_kidney_mask(item, reviewer)
            if annotation == "pseudo" and not is_pseudo:
                continue
            if annotation == "manual" and is_pseudo:
                continue
            if annotation == "sem_mascara" and has_kidney_mask:
                continue
            if normalized_search and normalized_search not in item["image_id"].casefold():
                continue
            listed.append(
                {
                    "image_id": item["image_id"],
                    "origem_visual": item["origem_visual"],
                    "pseudo_mascara": is_pseudo,
                    "sem_mascara": not has_kidney_mask,
                    "revisado": has_review,
                    "tem_cortex": bool(item["mascara_cortex_visual"]) and has_kidney_mask,
                    "thumb_url": f"/api/media/{item['image_id']}/image",
                }
            )
            if len(listed) >= limit:
                break
        return listed

    def reviews_by_image(self, reviewer):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE reviewer = ?", (reviewer,)
            ).fetchall()
        return {row["image_id"]: dict(row) for row in rows}

    def get_item(self, image_id, reviewer="", model=DEFAULT_INTRARENAL_MODEL):
        item = self.by_id.get(image_id)
        if item is None:
            return None
        if model not in self.intrarenal_model_predictions:
            model = DEFAULT_INTRARENAL_MODEL
        review = None
        if reviewer:
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM reviews WHERE image_id = ? AND reviewer = ?",
                    (image_id, reviewer),
                ).fetchone()
            review = dict(row) if row else None
        dataset_info = self.dataset_metadata.get(image_id, {})
        prediction = self.medulla_predictions.get(image_id, {})
        agreement = self.consensus_selected.get(image_id, {})
        is_pseudo = self.is_pseudo(item)
        medulla_generated = (
            item["origem_visual"] == "dataset_geral_prediction_space"
            and bool(item["mascara_medulla_visual"])
        )
        cortex_generated = (
            item["origem_visual"] == "dataset_geral_prediction_space"
            and bool(item["mascara_cortex_visual"])
        )
        central_generated = (
            item["origem_visual"] == "dataset_geral_prediction_space"
            and bool(item.get("mascara_central_echo_complex_visual", ""))
        )
        effective_layers = {
            name: self.effective_mask(image_id, name, reviewer, model)
            for name in LAYER_FIELDS
        }
        image_path = self.source_file(image_id, "image")
        if image_path is None:
            return None
        with Image.open(image_path) as source_image:
            dimensions = f"{source_image.width} x {source_image.height}"
            width, height = source_image.width, source_image.height
        roi_selection = self.brightness_roi_selection(image_id, reviewer, model)
        brightness_meter = self.brightness_meter(
            image_path,
            effective_layers,
            roi_selection["pares"] if roi_selection else None,
        )
        if roi_selection:
            brightness_meter["rois_atualizadas_em"] = roi_selection["updated_at"]
        medulla_metrics = {}
        if medulla_generated:
            selected_summary = self.intrarenal_model_summaries.get(model, {})
            multiclass_test = selected_summary.get("test", {})
            dice = multiclass_test.get("medulla_dice") or self.medulla_summary.get("test_dice")
            medulla_metrics = {
                "modelo": Path(
                    selected_summary.get("checkpoint", "")
                    or self.medulla_summary.get("checkpoint_path", "")
                ).name,
                "dice": dice,
                "iou": multiclass_test.get("medulla_iou") or self.medulla_summary.get("test_iou"),
                "f1": dice,
                "escopo": "Teste do modelo multiclasse intrarrenal selecionado; F1 equivale ao Dice por classe.",
            }
        kidney_generated = dataset_info.get("mask_status", "").startswith("generated")
        cortex_metrics = {}
        if cortex_generated:
            selected_summary = self.intrarenal_model_summaries.get(model, {})
            multiclass_test = selected_summary.get("test", {})
            dice = multiclass_test.get("cortex_dice") or self.cortex_summary.get("test_dice")
            cortex_metrics = {
                "modelo": Path(
                    selected_summary.get("checkpoint", "")
                    or self.cortex_summary.get("checkpoint", "")
                ).name,
                "dice": dice,
                "iou": multiclass_test.get("cortex_iou") or self.cortex_summary.get("test_iou"),
                "f1": dice,
                "escopo": "Teste do DeepLab multiclasse intrarrenal; F1 equivale ao Dice por classe.",
            }
        central_metrics = {}
        if central_generated:
            selected_summary = self.intrarenal_model_summaries.get(model, {})
            multiclass_test = selected_summary.get("test", {})
            dice = multiclass_test.get("central_echo_complex_dice")
            central_metrics = {
                "modelo": Path(selected_summary.get("checkpoint", "")).name,
                "dice": dice,
                "iou": multiclass_test.get("central_echo_complex_iou"),
                "f1": dice,
                "escopo": "Teste do DeepLab multiclasse intrarrenal; F1 equivale ao Dice por classe.",
            }
        kidney_metrics = {}
        if kidney_generated:
            kidney_result = self.kidney_summary.get("model2_kidney_against_manual_capsule", {})
            kidney_metrics = {
                "modelo": Path(self.kidney_summary.get("kidney_checkpoint", "")).name,
                "dice": kidney_result.get("global_dice"),
                "iou": kidney_result.get("global_iou"),
                "f1": kidney_result.get("global_dice"),
                "escopo": "Holdout de capsula renal; F1 equivale ao Dice binario.",
            }
        return {
            "image_id": image_id,
            "origem_visual": item["origem_visual"],
            "pseudo_mascara": is_pseudo,
            "tem_cortex": bool(effective_layers["cortex"]),
            "camadas_disponiveis": {
                name: bool(effective_layers[name]) for name in LAYER_FIELDS
            },
            "modelo_segmentacao_interna": model,
            "modelos_segmentacao_interna": sorted(self.intrarenal_model_predictions),
            "info": {
                "origem": dataset_info.get("source_name", image_id.split("__", 1)[0]),
                "dimensao": dimensions,
                "largura": width,
                "altura": height,
                "mascara_rim": "pseudo" if kidney_generated else "existente/manual",
                "mascara_cortex": "pseudo" if cortex_generated else ("manual" if item["mascara_cortex_visual"] else "indisponivel"),
                "mascara_medulla": "pseudo" if medulla_generated else ("manual" if item["mascara_medulla_visual"] else "indisponivel"),
                "mascara_central_echo_complex": "pseudo" if central_generated else ("manual" if item.get("mascara_central_echo_complex_visual", "") else "indisponivel"),
            },
            "metricas": {
                "rim": kidney_metrics,
                "cortex": cortex_metrics,
                "medulla": medulla_metrics,
                "central_echo_complex": central_metrics,
                "concordancia_modelos": {
                    "dice": agreement.get("model_dice", ""),
                    "iou": agreement.get("model_iou", ""),
                    "escopo": "Concordancia DeepLab x ROI-UNet para priorizacao de revisao.",
                }
                if agreement
                else {},
            },
            "medidor_brilho": brightness_meter,
            "image_url": f"/api/media/{image_id}/image",
            "layers": {
                name: (
                    f"/api/media/{image_id}/{name}?reviewer={reviewer}&model={model}"
                    if effective_layers[name]
                    else ""
                )
                for name in LAYER_FIELDS
            },
            "camadas_editadas": {
                name: self.edited_mask(image_id, name, reviewer) is not None
                for name in LAYER_FIELDS
            },
            "review": review,
        }

    def save_review(self, payload):
        image_id = str(payload.get("image_id", "")).strip()
        reviewer = str(payload.get("reviewer", "")).strip()
        reviewer_type = str(payload.get("reviewer_type", "")).strip()
        if image_id not in self.by_id:
            raise ValueError("Imagem desconhecida.")
        if not reviewer:
            raise ValueError("Informe o identificador do revisor.")
        if reviewer_type not in {"especialista", "nao_especialista"}:
            raise ValueError("Escolha o tipo de revisor.")
        values = {
            field: str(payload.get(field, "pendente")).strip()
            for field in REVIEW_STATUS_FIELDS.values()
        }
        if any(value not in STATUS_VALUES for value in values.values()):
            raise ValueError("Estado de mascara invalido.")
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT created_at, fibrose, fonte_fibrose FROM reviews WHERE image_id = ? AND reviewer = ?",
                (image_id, reviewer),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            connection.execute(
                """
                INSERT OR REPLACE INTO reviews (
                    image_id, reviewer, reviewer_type, status_rim, status_cortex,
                    status_medulla, status_central_echo_complex, fibrose, fonte_fibrose, observacao,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    reviewer,
                    reviewer_type,
                    values["status_rim"],
                    values["status_cortex"],
                    values["status_medulla"],
                    values["status_central_echo_complex"],
                    existing["fibrose"] if existing else "",
                    existing["fonte_fibrose"] if existing else "",
                    str(payload.get("observacao", "")).strip(),
                    created_at,
                    now,
                ),
            )
        self.sync_correction_approvals(image_id, reviewer, reviewer_type, values)
        return self.get_item(image_id, reviewer)

    def export_rows(self):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews ORDER BY updated_at, image_id, reviewer"
            ).fetchall()
            edits = connection.execute(
                """
                SELECT image_id, reviewer, layer, mask_path, operation, approval_status,
                    created_at, updated_at, approved_at, approved_by
                FROM mask_edits
                """
            ).fetchall()
            roi_selections = connection.execute(
                "SELECT image_id, reviewer, model, pairs_json, updated_at FROM brightness_roi_selections"
            ).fetchall()
        edits_by_key = {
            (row["image_id"], row["reviewer"], row["layer"]): dict(row)
            for row in edits
        }
        roi_selections_by_key = {}
        for selection in roi_selections:
            key = (selection["image_id"], selection["reviewer"])
            roi_selections_by_key.setdefault(key, []).append(dict(selection))
        exported = []
        reviewed_keys = set()
        for row in rows:
            reviewed_keys.add((row["image_id"], row["reviewer"]))
            result = dict(row)
            result["rois_brilho_ajustadas"] = json.dumps(
                roi_selections_by_key.get((row["image_id"], row["reviewer"]), []), ensure_ascii=False
            )
            for layer in LAYER_FIELDS:
                edit = edits_by_key.get((row["image_id"], row["reviewer"], layer), {})
                result[f"mascara_corrigida_{layer}"] = edit.get("mask_path", "")
                result[f"operacao_corrigida_{layer}"] = edit.get("operation", "")
                result[f"aprovacao_corrigida_{layer}"] = edit.get("approval_status", "")
                result[f"aprovada_em_{layer}"] = edit.get("approved_at", "")
                result[f"aprovada_por_{layer}"] = edit.get("approved_by", "")
            exported.append(result)
        correction_keys = {
            (row["image_id"], row["reviewer"])
            for row in edits
            if (row["image_id"], row["reviewer"]) not in reviewed_keys
            and row["image_id"] in self.by_id
            and self.source_file(row["image_id"], "image") is not None
            and self.get_item(row["image_id"], row["reviewer"]) is not None
        }
        correction_keys.update(
            key for key in roi_selections_by_key
            if key not in reviewed_keys
            and key[0] in self.by_id
            and self.source_file(key[0], "image") is not None
            and self.get_item(key[0], key[1]) is not None
        )
        for image_id, reviewer in sorted(correction_keys):
            related_edits = [
                dict(row)
                for row in edits
                if row["image_id"] == image_id and row["reviewer"] == reviewer
            ]
            latest = max((row.get("updated_at", "") for row in related_edits), default="")
            result = {
                "image_id": image_id,
                "reviewer": reviewer,
                "reviewer_type": "",
                "status_rim": "pendente",
                "status_cortex": "pendente",
                "status_medulla": "pendente",
                "status_central_echo_complex": "pendente",
                "fibrose": "",
                "fonte_fibrose": "",
                "observacao": "",
                "created_at": "",
                "updated_at": latest,
                "rois_brilho_ajustadas": json.dumps(
                    roi_selections_by_key.get((image_id, reviewer), []), ensure_ascii=False
                ),
            }
            for layer in LAYER_FIELDS:
                edit = edits_by_key.get((image_id, reviewer, layer), {})
                result[f"mascara_corrigida_{layer}"] = edit.get("mask_path", "")
                result[f"operacao_corrigida_{layer}"] = edit.get("operation", "")
                result[f"aprovacao_corrigida_{layer}"] = edit.get("approval_status", "")
                result[f"aprovada_em_{layer}"] = edit.get("approved_at", "")
                result[f"aprovada_por_{layer}"] = edit.get("approved_by", "")
            exported.append(result)
        return exported

    def export_to_database(self, reviewer=""):
        rows = self.export_rows()
        if reviewer:
            rows = [row for row in rows if row.get("reviewer") == reviewer]
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            for row in rows:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO curation_database_export (
                        image_id, reviewer, reviewer_type, payload_json, exported_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("image_id", ""),
                        row.get("reviewer", ""),
                        row.get("reviewer_type", ""),
                        json.dumps(row, ensure_ascii=False),
                        now,
                    ),
                )
        return {
            "exported": len(rows),
            "database": str(self.database_path),
            "table": "curation_database_export",
            "exported_at": now,
        }

    def model_mask_file(self, image_id, kind, model):
        if kind == "rim" or model not in self.intrarenal_model_predictions:
            return None
        row = self.intrarenal_model_predictions[model].get(image_id, {})
        field = {
            "cortex": "predicted_cortex_mask_path",
            "medulla": "predicted_medulla_mask_path",
            "central_echo_complex": "predicted_central_echo_complex_mask_path",
        }.get(kind)
        return self._resolve_path(row.get(field, ""))

    def source_file(self, image_id, kind, model=DEFAULT_INTRARENAL_MODEL):
        item = self.by_id.get(image_id)
        if item is None:
            return None
        if kind == "image":
            image_path = self._resolve_path(item["imagem_visual"])
            if image_path is not None:
                return image_path
            # As imagens que ainda não têm máscara renal não foram copiadas
            # para dataset_geral. Elas permanecem nas fontes em resolução de
            # trabalho (em geral 1024 x 768), com o prefixo de origem no ID.
            # Essa cópia deve ser preferida à miniatura na tela de revisão.
            source_roots = {
                "dataset_loader__": PROJECT_ROOT / "dataset_aumentado" / "fontes" / "dataset_loader",
                "monai_curated_png__": (
                    PROJECT_ROOT
                    / "dataset_aumentado"
                    / "fontes"
                    / "external_data"
                    / "processed"
                    / "monai_renal_png"
                    / "images"
                ),
            }
            for prefix, source_root in source_roots.items():
                if image_id.startswith(prefix):
                    original_path = source_root / f"{image_id[len(prefix):]}.png"
                    if original_path.exists():
                        return original_path
            # O manifesto preserva o caminho da imagem de origem. Em algumas
            # importações esse PNG não acompanha o projeto, mas a miniatura
            # JPEG usada pela curadoria está ao lado do próprio manifesto.
            # Usá-la como reserva mantém o caso revisável, inclusive quando
            # ele ainda não possui máscara renal.
            thumbnail_dir = self.manifest_path.parent / "imagem"
            for extension in (".jpg", ".jpeg", ".png"):
                thumbnail_path = thumbnail_dir / f"{image_id}{extension}"
                if thumbnail_path.exists():
                    return thumbnail_path
            return None
        if kind not in LAYER_FIELDS:
            return None
        if kind == "anomalia":
            return None
        model_mask = self.model_mask_file(image_id, kind, model)
        if item["origem_visual"] == "dataset_geral_prediction_space" and kind != "rim":
            return model_mask
        field, _ = LAYER_FIELDS[kind]
        return self._resolve_path(item[field])

    def edited_mask(self, image_id, kind, reviewer):
        if not reviewer:
            return None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT mask_path FROM mask_edits WHERE image_id = ? AND reviewer = ? AND layer = ?",
                (image_id, reviewer, kind),
            ).fetchone()
        path = Path(row["mask_path"]) if row else None
        return path if path and path.exists() else None

    def effective_mask(self, image_id, kind, reviewer="", model=DEFAULT_INTRARENAL_MODEL):
        return self.edited_mask(image_id, kind, reviewer) or self.source_file(image_id, kind, model)

    def brightness_roi_selection(self, image_id, reviewer, model):
        if not reviewer:
            return None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT pairs_json, updated_at FROM brightness_roi_selections WHERE image_id = ? AND reviewer = ? AND model = ?",
                (image_id, reviewer, model),
            ).fetchone()
        if not row:
            return None
        try:
            return {"pares": json.loads(row["pairs_json"]), "updated_at": row["updated_at"]}
        except json.JSONDecodeError:
            return None

    def save_brightness_rois(self, payload):
        image_id = str(payload.get("image_id", "")).strip()
        reviewer = str(payload.get("reviewer", "")).strip()
        model = str(payload.get("model", DEFAULT_INTRARENAL_MODEL)).strip() or DEFAULT_INTRARENAL_MODEL
        if image_id not in self.by_id:
            raise ValueError("Imagem nao encontrada.")
        if not reviewer:
            raise ValueError("Informe o revisor antes de ajustar as ROIs.")
        if model not in self.intrarenal_model_predictions:
            model = DEFAULT_INTRARENAL_MODEL
        if payload.get("restore"):
            with self.connect() as connection:
                connection.execute(
                    "DELETE FROM brightness_roi_selections WHERE image_id = ? AND reviewer = ? AND model = ?",
                    (image_id, reviewer, model),
                )
            return {"restored": True}
        item = self.by_id[image_id]
        image_path = self.source_file(image_id, "image")
        if image_path is None:
            raise ValueError("Imagem nao encontrada.")
        layers = {name: self.effective_mask(image_id, name, reviewer, model) for name in LAYER_FIELDS}
        meter = self.brightness_meter(image_path, layers, payload.get("pares"))
        if payload.get("preview"):
            return {"preview": True, "medidor_brilho": meter}
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO brightness_roi_selections (image_id, reviewer, model, pairs_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (image_id, reviewer, model, json.dumps(meter.get("pares", [])), now),
            )
        return {"saved": True, "medidor_brilho": meter, "updated_at": now}

    @staticmethod
    def brightness_meter(image_path, layers, roi_pairs=None):
        """Compara a luminância média original (sem CLAHE) do córtex e do CEC."""
        cortex_path = layers.get("cortex")
        cec_path = layers.get("central_echo_complex")
        medulla_path = layers.get("medulla")
        if not cortex_path or not cec_path:
            return {"disponivel": False}
        # A visualização pode aplicar CLAHE, mas o marcador sempre lê o arquivo
        # original para manter a medida independente de qualquer realce.
        with Image.open(image_path) as image, Image.open(cortex_path) as cortex_mask, Image.open(cec_path) as cec_mask:
            if image.size != cortex_mask.size or image.size != cec_mask.size:
                return {"disponivel": False}
            pixels = np.asarray(image.convert("L"), dtype=np.float32)
            cortex = np.asarray(cortex_mask.convert("L")) > 0
            cec = np.asarray(cec_mask.convert("L")) > 0
        if not cortex.any() or not cec.any():
            return {"disponivel": False}

        roi_region_masks = {"cortex": cortex, "central_echo_complex": cec}
        for region_name in ("medulla", "rim"):
            mask_path = layers.get(region_name)
            if not mask_path:
                continue
            try:
                with Image.open(mask_path) as region_mask:
                    if region_mask.size == pixels.shape[::-1]:
                        roi_region_masks[region_name] = np.asarray(region_mask.convert("L")) > 0
            except OSError:
                continue

        def region_stats(mask):
            values = pixels[mask]
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            return {
                "media": float(values.mean()),
                "mediana": float(median),
                "iqr": float(q75 - q25),
                "pixels": int(values.size),
                "saturacao_pct": float(((values <= 1) | (values >= 254)).mean() * 100),
            }

        def paired_rois(first, second, radius=12, requested_pairs=None):
            first_inner = cv2.erode(first.astype(np.uint8), np.ones((5, 5), np.uint8))
            second_inner = cv2.erode(second.astype(np.uint8), np.ones((5, 5), np.uint8))
            first_distance = cv2.distanceTransform(first_inner, cv2.DIST_L2, 5)
            second_distance = cv2.distanceTransform(second_inner, cv2.DIST_L2, 5)
            valid_first = first_distance >= radius
            valid_second = second_distance >= radius

            def identify_region(disc):
                labels = {
                    "cortex": "Córtex",
                    "central_echo_complex": "CEC",
                    "medulla": "Medula",
                    "rim": "Rim",
                }
                coverage = {
                    name: float(mask[disc].mean() * 100)
                    for name, mask in roi_region_masks.items()
                }
                best_name, best_coverage = max(coverage.items(), key=lambda item: item[1])
                if best_coverage < 50:
                    return "Área não segmentada", round(best_coverage, 1)
                return labels[best_name], round(best_coverage, 1)

            def measure_pair(label, px, py, cx, cy, pair_radius):
                if not 2 <= pair_radius <= 24:
                    raise ValueError("O raio das ROIs deve estar entre 2 e 24 pixels.")
                if not (0 <= px < pixels.shape[1] and 0 <= py < pixels.shape[0] and 0 <= cx < pixels.shape[1] and 0 <= cy < pixels.shape[0]):
                    raise ValueError("Centro de ROI fora dos limites da imagem.")
                depth_delta = abs(py - cy)
                yy, xx = np.indices(pixels.shape)
                first_disc = (yy - py) ** 2 + (xx - px) ** 2 <= pair_radius ** 2
                second_disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= pair_radius ** 2
                free_pair = bool(requested_pairs is not None and current_pair.get("modo") == "livre")
                if requested_pairs is not None and (min(px, py, cx, cy) < pair_radius or max(px, cx) + pair_radius >= pixels.shape[1] or max(py, cy) + pair_radius >= pixels.shape[0] or (not free_pair and (not first_inner[first_disc].all() or not second_inner[second_disc].all()))):
                    raise ValueError("Cada círculo deve caber dentro da imagem; para CEC/córtex, também deve caber na máscara erodida. Reduza o raio se necessário.")
                first_values = pixels[first_disc]
                second_values = pixels[second_disc]
                c_mean, e_mean = float(first_values.mean()), float(second_values.mean())
                if e_mean <= 0:
                    raise ValueError("O ponto de referência tem brilho zero; escolha outra área.")
                if free_pair:
                    # `first` é o ponto B (destino) e `second` é o ponto A
                    # (origem). No comparador livre a leitura solicitada é A/B.
                    a_mean, b_mean = e_mean, c_mean
                    a_q25, a_median, a_q75 = np.percentile(second_values, [25, 50, 75])
                    b_q25, b_median, b_q75 = np.percentile(first_values, [25, 50, 75])
                    region_a, coverage_a = identify_region(second_disc)
                    region_b, coverage_b = identify_region(first_disc)
                    return {"faixa": label, "a": round(a_mean, 1), "b": round(b_mean, 1), "a_mediana": round(float(a_median), 1), "a_iqr": round(float(a_q75 - a_q25), 1), "b_mediana": round(float(b_median), 1), "b_iqr": round(float(b_q75 - b_q25), 1), "a_pixels": int(second_values.size), "b_pixels": int(first_values.size), "a_saturacao": round(float(((second_values <= 1) | (second_values >= 254)).mean() * 100), 1), "b_saturacao": round(float(((first_values <= 1) | (first_values >= 254)).mean() * 100), 1), "percentual": round(a_mean / b_mean * 100, 1), "diferenca": round(b_mean - a_mean, 1), "contraste": round((b_mean - a_mean) / (b_mean + a_mean) * 100, 1), "ponto_a": [int(cx), int(cy)], "ponto_b": [int(px), int(py)], "regiao_a": region_a, "regiao_b": region_b, "cobertura_a_pct": coverage_a, "cobertura_b_pct": coverage_b, "cortex_centro": [int(px), int(py)], "cec_centro": [int(cx), int(cy)], "raio": pair_radius, "modo": "livre", "diferenca_profundidade_px": int(depth_delta)}
                return {"faixa": label, "cortex": round(c_mean, 1), "cec": round(e_mean, 1), "percentual": round(c_mean / e_mean * 100, 1), "diferenca": round(e_mean - c_mean, 1), "contraste": round((e_mean - c_mean) / (e_mean + c_mean) * 100, 1), "cortex_centro": [int(px), int(py)], "cec_centro": [int(cx), int(cy)], "raio": pair_radius, "modo": "livre" if free_pair else "anatomica", "diferenca_profundidade_px": int(depth_delta)}

            if requested_pairs is not None:
                if not isinstance(requested_pairs, list) or not requested_pairs:
                    raise ValueError("Informe pelo menos um par de ROIs.")
                if len(requested_pairs) > 3:
                    raise ValueError("São permitidos no máximo três pares de ROIs.")
                measured = []
                for pair in requested_pairs:
                    try:
                        px, py = (int(value) for value in pair["cortex_centro"])
                        cx, cy = (int(value) for value in pair["cec_centro"])
                        pair_radius = int(pair["raio"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError("Formato de ROI inválido.") from error
                    current_pair = pair
                    measured.append(measure_pair(str(pair.get("faixa", "ajustada")), px, py, cx, cy, pair_radius))
                return measured
            ys = np.where(first | second)[0]
            if not ys.size:
                return []
            low, high = int(ys.min()), int(ys.max()) + 1
            pairs = []
            for index, label in enumerate(("superior", "medio", "inferior")):
                start = low + (high - low) * index / 3
                end = low + (high - low) * (index + 1) / 3
                target = (start + end) / 2
                candidates = np.argwhere(valid_second & (np.indices(valid_second.shape)[0] >= start) & (np.indices(valid_second.shape)[0] < end))
                if not candidates.size:
                    continue
                cy, cx = min(candidates, key=lambda point: abs(point[0] - target) - second_distance[tuple(point)] * .15)
                cortex_candidates = np.argwhere(valid_first)
                if not cortex_candidates.size:
                    continue
                py, px = min(cortex_candidates, key=lambda point: (abs(point[0] - cy), abs(point[1] - cx)))
                pairs.append(measure_pair(label, int(px), int(py), int(cx), int(cy), radius))
            return pairs

        cortex_stats = region_stats(cortex)
        cec_stats = region_stats(cec)
        cortex_mean = cortex_stats["media"]
        cec_mean = cec_stats["media"]
        signed_difference = cec_mean - cortex_mean
        difference = abs(signed_difference)
        # O CEC é a referência de 100%; o cursor não ultrapassa o fim da escala.
        brightness_percent = (cortex_mean / cec_mean) * 100 if cec_mean else 0.0
        position = max(0.0, min(100.0, brightness_percent))
        result = {
            "disponivel": True,
            "percentual": round(brightness_percent, 1),
            "posicao": round(position, 1),
            "diferenca": round(difference, 1),
            "diferenca_cec_cortex": round(signed_difference, 1),
            "contraste_normalizado": round((signed_difference / (cec_mean + cortex_mean)) * 100, 1),
            "cortex": round(cortex_mean, 1),
            "cec": round(cec_mean, 1),
            "cortex_mediana": round(cortex_stats["mediana"], 1),
            "cortex_iqr": round(cortex_stats["iqr"], 1),
            "cec_mediana": round(cec_stats["mediana"], 1),
            "cec_iqr": round(cec_stats["iqr"], 1),
            "cortex_pixels": cortex_stats["pixels"],
            "cec_pixels": cec_stats["pixels"],
            "cortex_saturacao": round(cortex_stats["saturacao_pct"], 1),
            "cec_saturacao": round(cec_stats["saturacao_pct"], 1),
            "imagem_base": "original_sem_clahe",
        }
        scale = max(6, min(24, int(round(min(pixels.shape) * .025))))
        pairs = paired_rois(cortex, cec, scale, roi_pairs)
        if pairs:
            values = np.asarray([pair["percentual"] for pair in pairs])
            result.update({"pares": pairs, "percentual_pares_mediana": round(float(np.median(values)), 1), "percentual_pares_iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 1), "origem_pares": "manual" if roi_pairs is not None else "automatica"})
            if any(pair.get("modo") == "livre" for pair in pairs):
                differences = np.asarray([pair["diferenca"] for pair in pairs])
                contrasts = np.asarray([pair["contraste"] for pair in pairs])
                result.update({"modo_metrica": "pontos_livres", "percentual": round(float(np.median(values)), 1), "posicao": round(float(np.clip(np.median(values), 0, 100)), 1), "diferenca_cec_cortex": round(float(np.median(differences)), 1), "contraste_normalizado": round(float(np.median(contrasts)), 1)})
                first_free = next(pair for pair in pairs if pair.get("modo") == "livre")
                result.update({"cortex_medulla_percentual": round(float(np.median(values)), 1), "diferenca_cortex_medulla": round(float(-np.median(differences)), 1), "cortex_mediana": first_free["a_mediana"], "cortex_iqr": first_free["a_iqr"], "cec_mediana": first_free["b_mediana"], "cec_iqr": first_free["b_iqr"], "cortex_pixels": sum(pair.get("a_pixels", 0) for pair in pairs), "cec_pixels": sum(pair.get("b_pixels", 0) for pair in pairs), "cortex_saturacao": first_free["a_saturacao"], "cec_saturacao": first_free["b_saturacao"]})
        if medulla_path and result.get("modo_metrica") != "pontos_livres":
            with Image.open(medulla_path) as medulla_mask:
                if medulla_mask.size == pixels.shape[::-1]:
                    medulla = np.asarray(medulla_mask.convert("L")) > 0
                    if medulla.any():
                        medulla_stats = region_stats(medulla)
                        if medulla_stats["media"]:
                            result.update({
                                "cortex_medulla_percentual": round((cortex_mean / medulla_stats["media"]) * 100, 1),
                                "diferenca_cortex_medulla": round(cortex_mean - medulla_stats["media"], 1),
                                "medulla": round(medulla_stats["media"], 1),
                            })
        return result

    def image_size(self, image_id):
        item = self.by_id.get(image_id)
        if item is None:
            return None
        image_path = self.source_file(image_id, "image")
        if image_path is None:
            return None
        with Image.open(image_path) as source:
            return source.size

    def sync_correction_approvals(self, image_id, reviewer, approved_by, values):
        if not hasattr(self, "corrections"):
            return
        for layer, field in REVIEW_STATUS_FIELDS.items():
            if values.get(field) == "aceita":
                self.corrections.update_approval(image_id, reviewer, layer, "aprovada", approved_by)
            elif values.get(field) in {"corrigir", "rejeitada"}:
                self.corrections.update_approval(image_id, reviewer, layer, "reprovada", approved_by)

    @staticmethod
    def _safe_name(value):
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in value
        )

    def save_polygon(self, payload):
        image_id = str(payload.get("image_id", "")).strip()
        reviewer = str(payload.get("reviewer", "")).strip()
        self.corrections.create_from_polygon(payload)
        return self.get_item(image_id, reviewer)

    def contour_bytes(self, image_id, kind, reviewer="", model=DEFAULT_INTRARENAL_MODEL):
        item = self.by_id.get(image_id)
        if item is None or kind not in LAYER_FIELDS:
            return None
        _, color = LAYER_FIELDS[kind]
        mask_path = self.effective_mask(image_id, kind, reviewer, model)
        if not mask_path:
            return None
        image_path = self.source_file(image_id, "image")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        reviewer_suffix = f"__{self._safe_name(reviewer)}" if reviewer else ""
        model_suffix = f"__{self._safe_name(model)}" if kind != "rim" else ""
        target = self.cache_dir / f"{image_id}__{kind}{reviewer_suffix}{model_suffix}.png"
        if target.exists() and target.stat().st_mtime >= mask_path.stat().st_mtime:
            return target.read_bytes()
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"Dimensoes incompatíveis: {image_id} {kind} "
                    f"imagem={image.size} mascara={mask.size}"
                )
            binary = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
            eroded = binary.filter(ImageFilter.MinFilter(3))
            boundary = ImageChops.difference(binary, eroded)
            overlay = Image.new("RGBA", image.size, color)
            alpha_value = min(color[3], 190)
            alpha = boundary.point(lambda value: alpha_value if value > 0 else 0)
            overlay.putalpha(alpha)
            output = BytesIO()
            overlay.save(output, format="PNG")
        target.write_bytes(output.getvalue())
        return output.getvalue()

    def transformed_image_bytes(self, image_id, view_mode="original"):
        image_path = self.source_file(image_id, "image")
        if image_path is None:
            return None
        if view_mode == "original":
            return image_path.read_bytes()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        if view_mode in {"superres", "superres_clahe"}:
            height, width = image.shape[:2]
            image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        if view_mode in {"clahe", "superres_clahe"}:
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        success, encoded = cv2.imencode(".png", image)
        return encoded.tobytes() if success else None


class CurationHandler(BaseHTTPRequestHandler):
    store: CurationStore
    inference: InferenceService

    def log_message(self, format_string, *args):
        print(f"[web] {self.address_string()} - {format_string % args}")

    def send_json(self, value, status=HTTPStatus.OK):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, content_type=None):
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        reviewer = query.get("reviewer", [""])[0].strip()
        model = query.get("model", [DEFAULT_INTRARENAL_MODEL])[0].strip() or DEFAULT_INTRARENAL_MODEL
        try:
            if path == "/api/meta":
                self.send_json(self.store.summary(reviewer))
                return
            if path == "/api/items":
                limit = min(max(int(query.get("limit", ["150"])[0]), 1), 500)
                self.send_json(
                    self.store.list_items(
                        reviewer=reviewer,
                        state=query.get("state", ["todos"])[0],
                        source=query.get("source", [""])[0],
                        annotation=query.get("annotation", [""])[0],
                        search=query.get("search", [""])[0],
                        limit=limit,
                    )
                )
                return
            if path.startswith("/api/item/"):
                image_id = unquote(path[len("/api/item/") :])
                item = self.store.get_item(image_id, reviewer, model)
                if item is None:
                    self.send_json({"error": "Imagem nao encontrada."}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json(item)
                return
            if path.startswith("/api/inferences/"):
                parts = path.split("/")
                if len(parts) != 5:
                    raise ValueError("Caminho de inferência inválido.")
                image_id, kind = unquote(parts[3]), parts[4]
                media = self.inference.media(image_id, kind)
                if media is None:
                    self.send_json({"error": "Resultado de inferência não encontrado."}, HTTPStatus.NOT_FOUND)
                    return
                body, content_type = media
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/corrections":
                limit = min(max(int(query.get("limit", ["80"])[0]), 1), 300)
                corrections = [
                    correction
                    for correction in self.store.corrections.read(reviewer=reviewer)
                    if correction["image_id"] in self.store.by_id
                    and self.store.source_file(correction["image_id"], "image") is not None
                    and self.store.get_item(correction["image_id"], reviewer, model) is not None
                ][:limit]
                self.send_json({"corrections": corrections, "total_listed": len(corrections)})
                return
            if path.startswith("/api/corrections/"):
                image_id = unquote(path[len("/api/corrections/") :])
                if image_id not in self.store.by_id:
                    self.send_json({"error": "Imagem nao encontrada."}, HTTPStatus.NOT_FOUND)
                    return
                layer = query.get("layer", [""])[0].strip()
                corrections = self.store.corrections.read(
                    image_id=image_id,
                    reviewer=reviewer,
                    layer=layer,
                )
                self.send_json({"image_id": image_id, "corrections": corrections})
                return
            if path.startswith("/api/media/"):
                parts = path.split("/")
                if len(parts) != 5:
                    raise ValueError("Caminho de mídia inválido.")
                image_id, kind = unquote(parts[3]), parts[4]
                if kind == "image":
                    view_mode = query.get("view", ["original"])[0].strip() or "original"
                    if view_mode not in {"original", "clahe", "superres", "superres_clahe"}:
                        raise ValueError("Modo de visualizacao invalido.")
                    body = self.store.transformed_image_bytes(image_id, view_mode)
                    if body is None:
                        self.send_json({"error": "Imagem nao encontrada."}, HTTPStatus.NOT_FOUND)
                    else:
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(body)
                    return
                body = self.store.contour_bytes(image_id, kind, reviewer, model)
                if body is None:
                    self.send_json({"error": "Mascara nao encontrada."}, HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/export.json":
                self.send_json(self.store.export_rows())
                return
            if path == "/api/export.csv":
                rows = self.store.export_rows()
                fields = [
                    "image_id",
                    "reviewer",
                    "reviewer_type",
                    "status_rim",
                    "status_cortex",
                    "status_medulla",
                    "status_central_echo_complex",
                    "observacao",
                    "rois_brilho_ajustadas",
                    "mascara_corrigida_rim",
                    "operacao_corrigida_rim",
                    "aprovacao_corrigida_rim",
                    "aprovada_em_rim",
                    "aprovada_por_rim",
                    "mascara_corrigida_cortex",
                    "operacao_corrigida_cortex",
                    "aprovacao_corrigida_cortex",
                    "aprovada_em_cortex",
                    "aprovada_por_cortex",
                    "mascara_corrigida_medulla",
                    "operacao_corrigida_medulla",
                    "aprovacao_corrigida_medulla",
                    "aprovada_em_medulla",
                    "aprovada_por_medulla",
                    "mascara_corrigida_central_echo_complex",
                    "operacao_corrigida_central_echo_complex",
                    "aprovacao_corrigida_central_echo_complex",
                    "aprovada_em_central_echo_complex",
                    "aprovada_por_central_echo_complex",
                    "mascara_corrigida_anomalia",
                    "operacao_corrigida_anomalia",
                    "aprovacao_corrigida_anomalia",
                    "aprovada_em_anomalia",
                    "aprovada_por_anomalia",
                    "created_at",
                    "updated_at",
                ]
                output = StringIO()
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                body = output.getvalue().encode("utf-8-sig")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=curadoria_respostas.csv")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            target = STATIC_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
            if target.exists() and target.resolve().is_relative_to(STATIC_ROOT.resolve()):
                self.send_file(target)
                return
            self.send_json({"error": "Recurso nao encontrado."}, HTTPStatus.NOT_FOUND)
        except (OSError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/reviews", "/api/corrections", "/api/database-export", "/api/inferences", "/api/brightness-rois"}:
            self.send_json({"error": "Recurso nao encontrado."}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/database-export":
                reviewer = str(payload.get("reviewer", "")).strip()
                result = self.store.export_to_database(reviewer)
                self.send_json({"saved": True, "export": result})
                return
            if parsed.path == "/api/inferences":
                result = self.inference.infer(
                    payload.get("image_data", ""),
                    payload.get("filename", "imagem"),
                )
                self.send_json({"saved": True, "inference": result})
                return
            if parsed.path == "/api/reviews":
                item = self.store.save_review(payload)
                self.send_json({"saved": True, "item": item})
                return
            if parsed.path == "/api/corrections":
                item = self.store.save_polygon(payload)
                self.send_json({"saved": True, "item": item})
                return
            if parsed.path == "/api/brightness-rois":
                result = self.store.save_brightness_rois(payload)
                self.send_json(result)
                return
        except (json.JSONDecodeError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_PUT(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/corrections/"):
            self.send_json({"error": "Recurso nao encontrado."}, HTTPStatus.NOT_FOUND)
            return
        try:
            image_id = unquote(parsed.path[len("/api/corrections/") :])
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            reviewer = str(payload.get("reviewer", "")).strip()
            layer = str(payload.get("layer", "")).strip()
            approval_status = str(payload.get("approval_status", "")).strip()
            approved_by = str(payload.get("approved_by", reviewer)).strip()
            if not reviewer:
                raise ValueError("Informe o revisor.")
            correction = self.store.corrections.update_approval(
                image_id,
                reviewer,
                layer,
                approval_status,
                approved_by,
            )
            if correction is None:
                self.send_json({"error": "Correcao nao encontrada."}, HTTPStatus.NOT_FOUND)
                return
            self.send_json({"saved": True, "correction": correction})
        except (json.JSONDecodeError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def parse_args():
    parser = argparse.ArgumentParser(description="Servidor web local para curadoria renal.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATA_DIR / "curadoria.sqlite3")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_DATA_DIR / "overlay_cache")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    store = CurationStore(args.manifest, args.database, args.cache_dir)
    CurationHandler.store = store
    CurationHandler.inference = InferenceService(
        PROJECT_ROOT,
        args.database.parent / "inferencias",
        CurationStore.brightness_meter,
    )
    server = ThreadingHTTPServer((args.host, args.port), CurationHandler)
    print(f"Curadoria Renal: http://{args.host}:{args.port}")
    print(f"Casos: {len(store.items)} | Banco: {args.database.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
