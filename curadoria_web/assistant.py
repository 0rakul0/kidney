"""Automatic bridge between Curadoria Web artifacts and the visual assistant."""

from __future__ import annotations

import json
import mimetypes
import re
import sys
import unicodedata
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from image_validation import decode_image
except ImportError:  # package import in tests and tooling
    from curadoria_web.image_validation import decode_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AssistantAuthorizationError(PermissionError):
    """Raised when a local reviewer or external-processing consent is absent."""


class AssistantService:
    """Builds assistant requests from server-side artifacts only.

    The browser sends an ``image_id`` and a reviewer identifier. It never
    supplies paths, masks or evidence JSON. This prevents a client from
    substituting unrelated masks for the image being reviewed.
    """

    LEGEND = {"rim": "rim", "cortex": "cortex", "medulla": "medula", "central_echo_complex": "CEC"}
    COLORS = {
        "rim": (243, 79, 86),
        "cortex": (65, 203, 208),
        "medulla": (255, 219, 59),
        "central_echo_complex": (255, 145, 0),
    }
    REGION_TERMS = (
        ("cortex", ("córtex", "cortex")),
        ("medulla", ("medula",)),
        ("central_echo_complex", ("cec", "complexo ecogênico central")),
        ("rim", ("rim", "renal")),
    )

    def __init__(self, store, inference):
        self.store = store
        self.inference = inference

    @staticmethod
    def parse_mask_command(question):
        """Translate a natural-language mask request into a review proposal.

        The command only describes a candidate operation. It never edits a
        mask by itself; the interface must generate and explicitly approve the
        proposal first.
        """
        original = str(question or "").strip()
        normalized = unicodedata.normalize("NFKD", original.casefold()).encode("ascii", "ignore").decode("ascii")
        if not any(term in normalized for term in ("melhor", "corrig", "refin", "ajust", "limp", "preench", "remov")):
            return None
        layer_terms = (
            ("central_echo_complex", ("cec", "complexo ecogenico central", "complexo ecogenico")),
            ("cortex", ("cortex",)),
            ("medulla", ("medula",)),
            ("rim", ("rim", "renal", "capsula")),
        )
        layer = next((name for name, terms in layer_terms if any(term in normalized for term in terms)), None)
        if layer is None:
            return {
                "intent": "refinar_mascara",
                "layer": None,
                "operation": None,
                "status": "estrutura_nao_identificada",
                "requires_confirmation": True,
                "instruction": "Informe a estrutura: rim, córtex, medula ou CEC.",
            }
        if "preench" in normalized or "burac" in normalized:
            operation = "preencher_vazios"
        elif "ruido" in normalized or "fragment" in normalized or "remov" in normalized:
            operation = "remover_ruido"
        elif "suav" in normalized:
            operation = "suavizar_bordas"
        else:
            operation = "refinar_mascara"
        labels = {"rim": "rim", "cortex": "córtex", "medulla": "medula", "central_echo_complex": "CEC"}
        descriptions = {
            "refinar_mascara": "suavizar pequenas irregularidades e preservar a região dominante",
            "preencher_vazios": "preencher pequenos vazios internos da máscara",
            "remover_ruido": "remover componentes pequenos e ruído isolado",
            "suavizar_bordas": "suavizar a borda sem substituir a máscara original",
        }
        return {
            "intent": "refinar_mascara",
            "layer": layer,
            "rotulo": labels[layer],
            "operation": operation,
            "status": "proposta_disponivel",
            "requires_confirmation": True,
            "instruction": f"Preparar uma proposta para {descriptions[operation]} na máscara de {labels[layer]}.",
        }

    @staticmethod
    def _mask_info(path: Path | None, image_size: tuple[int, int]):
        if path is None or not path.exists():
            return {"disponivel": False, "pixels": 0}
        try:
            with Image.open(path) as source:
                source.load()
                size = source.size
                mask = np.asarray(source.convert("L")) > 0
            if size != image_size:
                return {
                    "disponivel": False,
                    "pixels": 0,
                    "dimensao": list(size),
                    "erro": "dimensao_incompativel",
                }
            return {"disponivel": True, "pixels": int(mask.sum()), "dimensao": list(size)}
        except (OSError, ValueError):
            return {"disponivel": False, "pixels": 0, "erro": "mascara_invalida"}

    @staticmethod
    def _bbox(path: Path | None):
        if path is None or not path.exists():
            return None
        try:
            with Image.open(path) as source:
                source.load()
                mask = np.asarray(source.convert("L")) > 0
            ys, xs = np.where(mask)
            if not xs.size:
                return None
            return {"x1": int(xs.min()), "y1": int(ys.min()), "x2": int(xs.max() + 1), "y2": int(ys.max() + 1)}
        except (OSError, ValueError):
            return None

    @staticmethod
    def _validated_png(path: Path):
        if path is None or not path.exists():
            raise ValueError("Imagem da inferência não encontrada.")
        raw = path.read_bytes()
        image = decode_image(raw)
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), "image/png", image.size

    def _mask_overlay(self, raw: bytes, mask_paths, image_size: tuple[int, int]):
        """Create one visual input containing the original plus all masks."""
        with Image.open(BytesIO(raw)) as source:
            canvas = source.convert("RGB")
        for name, path in mask_paths.items():
            if not path or not path.exists() or name not in self.COLORS:
                continue
            try:
                with Image.open(path) as source:
                    mask = source.convert("L")
                    if mask.size != image_size:
                        continue
                    layer = Image.new("RGB", image_size, self.COLORS[name])
                    alpha = mask.point(lambda value: 86 if value > 0 else 0)
                    canvas = Image.composite(layer, canvas, alpha)
            except (OSError, ValueError):
                continue
        output = BytesIO()
        canvas.save(output, format="PNG")
        return output.getvalue()

    def _artifacts(self, image_id: str, reviewer: str, model: str):
        if image_id in self.inference.results:
            result = self.inference.results[image_id]
            image_path = result["image_path"]
            mask_paths = result["mask_paths"]
            with Image.open(image_path) as image:
                image_size = image.size
            bbox = result.get("capsule_bbox") or self._bbox(mask_paths.get("rim"))
            return {
                "image_path": image_path,
                "mask_paths": mask_paths,
                "image_size": image_size,
                "bbox": bbox,
                "meter": result.get("meter", {}),
                "metrics": {},
                "quality": result.get("quality", {}),
                "region_review": result.get("region_review", []),
                "origin": "inferencia_web",
                "filename": result.get("filename", image_id),
                "device": result.get("device", "unknown"),
            }
        if image_id not in self.store.by_id:
            raise ValueError("Imagem desconhecida ou resultado de inferência expirado.")
        item = self.store.get_item(image_id, reviewer, model)
        if item is None:
            raise ValueError("Imagem não encontrada na Curadoria Web.")
        image_path = self.store.source_file(image_id, "image")
        mask_paths = {
            name: self.store.effective_mask(image_id, name, reviewer, model)
            for name in self.LEGEND
        }
        return {
            "image_path": image_path,
            "mask_paths": mask_paths,
            "image_size": (int(item["info"]["largura"]), int(item["info"]["altura"])),
            "bbox": self._bbox(mask_paths.get("rim")),
            "meter": item.get("medidor_brilho", {}),
            "metrics": item.get("metricas", {}),
            "quality": item.get("qualidade_inferencia", {}),
            "region_review": item.get("priorizacao_regioes", []),
            "origin": item.get("origem_visual", "curadoria"),
            "filename": image_id,
            "device": "curadoria",
        }

    def build_request(self, image_id: str, reviewer: str, model: str = "unet", visual_mode: str = "capsule"):
        if not reviewer.strip():
            raise AssistantAuthorizationError("Informe o identificador do revisor.")
        if visual_mode not in {"capsule", "full"}:
            raise ValueError("Modo visual inválido.")
        artifacts = self._artifacts(image_id.strip(), reviewer.strip(), model)
        raw, mime, image_size = self._validated_png(artifacts["image_path"])
        masks = {
            name: self._mask_info(path, image_size)
            for name, path in artifacts["mask_paths"].items()
        }
        available = [name for name, info in masks.items() if info["disponivel"]]
        bbox = artifacts["bbox"]
        evidence = {
            "image_id": image_id,
            "origem": artifacts["origin"],
            "arquivo": artifacts["filename"],
            "dimensao_imagem": list(image_size),
            "modo_visual": visual_mode,
            "mascaras_utilizadas": {
                name: {"disponivel": info["disponivel"], "pixels": info["pixels"], "dimensao": info.get("dimensao")}
                for name, info in masks.items()
            },
            "legenda_mascaras": self.LEGEND,
            "medidor_brilho": artifacts["meter"],
            "metricas_segmentacao": artifacts["metrics"],
            "segmentacao": {"capsule_bbox": bbox, "dispositivo": artifacts["device"]},
            "qualidade_inferencia": artifacts.get("quality", {}),
            "priorizacao_regioes": artifacts.get("region_review", []),
            "sobreposicao_mascaras_enviada": True,
            "limitacoes": [
                "Máscaras e métricas são evidências de pesquisa e precisam de revisão humana.",
                "A resposta não confirma diagnóstico nem caracteriza isoladamente um foco ecogênico.",
            ],
        }
        if visual_mode == "capsule" and bbox:
            evidence["capsule_bbox"] = bbox
            evidence["capsule_padding_ratio"] = 0.12
            evidence["contexto_visual"] = "recorte_da_capsula_renal_com_margem"
        else:
            evidence["contexto_visual"] = "imagem_completa_para_comparacao_anatomica"
        raw = self._mask_overlay(raw, artifacts["mask_paths"], image_size)
        context = {
            "image_id": image_id,
            "visual_mode": visual_mode,
            "image_size": list(image_size),
            "capsule_bbox": bbox,
            "mascaras_disponiveis": available,
            "evidence_refs": [
                "imagem_original",
                *[f"mascara_{name}" for name in available],
                "medidor_brilho",
                "metricas_segmentacao",
            ],
        }
        if artifacts.get("quality"):
            context["qualidade_inferencia"] = artifacts["quality"]
        if artifacts.get("region_review"):
            context["priorizacao_regioes"] = artifacts["region_review"]
        evidence["evidence_refs"] = context["evidence_refs"]
        return raw, mime, evidence, context

    @staticmethod
    def _prompt(question: str):
        return (
            f"{question.strip()}\n\n"
            "Use as evidências anexadas, incluindo as máscaras e o medidor de brilho. "
            "Diferencie observação direta, métrica computacional e hipótese. "
            "Procure áreas ecogênicas dentro da cápsula quando o modo for recorte, "
            "mas não trate o CEC fisiológico como lesão sem evidência independente. "
            "Não liste rim, córtex, medula ou CEC apenas porque a máscara existe: "
            "aponte uma região somente quando houver evidência visual ou métrica "
            "que justifique a revisão. Se não houver foco localizado, declare que "
            "não é possível localizar um foco ecogênico com segurança. "
            "Não conclua presença ou ausência de edema, inflamação ou infiltração, "
            "pois essas condições não são avaliadas por este protocolo. "
            "Não invente medidas, lateralidade, inflamação, edema, gravidade ou diagnóstico. "
            "Use a priorização automática apenas como indicação operacional de revisão. "
            "Se a pergunta solicitar melhorar, corrigir ou refinar uma máscara, não diga que a máscara foi editada: descreva apenas uma proposta que depende de confirmação do revisor. "
            "Quando organizar a resposta, separe observação, evidência, região mencionada e limitação. "
            "Responda em português brasileiro, de forma concisa e auditável."
        )

    @classmethod
    def _regions_mentioned(cls, answer):
        text = str(answer or "").casefold()
        mentioned = []
        for name, terms in cls.REGION_TERMS:
            if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) for term in terms):
                mentioned.append(name)
        return mentioned

    @classmethod
    def _structured_response(cls, answer, evidence, context):
        return {
            "observacao": answer,
            "evidencias_utilizadas": context.get("evidence_refs", []),
            "regioes_mencionadas": cls._regions_mentioned(answer),
            "regioes_priorizadas": [
                {
                    "regiao": region.get("regiao"),
                    "prioridade": region.get("prioridade"),
                    "motivos": region.get("motivos", []),
                }
                for region in context.get("priorizacao_regioes", [])
            ],
            "limitacoes": evidence.get("limitacoes", []),
            "status": "revisao_humana_obrigatoria",
        }

    def analyze(self, payload):
        image_id = str(payload.get("image_id", "")).strip()
        reviewer = str(payload.get("reviewer", "")).strip()
        model = str(payload.get("model", "unet")).strip() or "unet"
        mode = str(payload.get("visual_mode", "capsule")).strip() or "capsule"
        provider = str(payload.get("provider", "local")).strip() or "local"
        question = str(payload.get("question", "")).strip() or "Quais áreas ecogênicas merecem revisão?"
        consent = bool(payload.get("consent_external_processing", False))
        if provider not in {"local", "cloud"}:
            raise ValueError("Provedor inválido.")
        if provider == "cloud" and not consent:
            raise AssistantAuthorizationError("Consentimento explícito é obrigatório para processamento externo.")
        raw, mime, evidence, context = self.build_request(image_id, reviewer, model, mode)
        mask_command = self.parse_mask_command(question)
        if mask_command and provider == "local":
            if mask_command.get("status") == "estrutura_nao_identificada":
                command_answer = (
                    "Identifiquei um pedido de melhoria de máscara, mas a estrutura não foi determinada. "
                    "Informe se deseja revisar rim, córtex, medula ou CEC. Nenhuma máscara foi alterada."
                )
            else:
                command_answer = (
                    f"Identifiquei um pedido para {mask_command['rotulo']}. "
                    f"A operação sugerida é {mask_command['operation']}. "
                    "Gere a proposta, revise a sobreposição e confirme sua aplicação. "
                    "Nenhuma máscara foi alterada automaticamente."
                )
            response = {"answer": command_answer, "model": "mask-command-router", "usage": {}}
        elif provider == "local":
            from src.kidney_assistant import local_vllm
            response = local_vllm.ask_about_session(self._prompt(question), evidence, raw, mime)
        else:
            from src.kidney_assistant import local_vllm, luna
            cloud_evidence = dict(evidence)
            cloud_raw = raw
            if mode == "capsule" and evidence.get("capsule_bbox"):
                cloud_raw, crop_info = local_vllm._crop_capsule(raw, mime, evidence)
                cloud_evidence.pop("capsule_bbox", None)
                cloud_evidence.pop("capsule_padding_ratio", None)
                cloud_evidence["contexto_visual"] = {"imagem_enviada": "recorte_da_capsula_renal", **(crop_info or {})}
            response = luna.ask_about_session(self._prompt(question), cloud_evidence, cloud_raw, mime)
        result = {
            **response,
            "run_id": uuid.uuid4().hex,
            "image_id": image_id,
            "provider": provider,
            "visual_mode": mode,
            "evidence_refs": context["evidence_refs"],
            "evidence_context": context,
            "limitations": evidence["limitacoes"],
            "mask_command": mask_command,
        }
        result["structured"] = self._structured_response(result.get("answer", ""), evidence, context)
        self.store.save_assistant_run(result, reviewer, question)
        return result

    def compare(self, payload):
        if not bool(payload.get("consent_external_processing", False)):
            raise AssistantAuthorizationError("Consentimento explícito é obrigatório para comparar com a OpenAI.")
        local_payload = dict(payload, provider="local")
        cloud_payload = dict(payload, provider="cloud")
        local_result = self.analyze(local_payload)
        cloud_result = self.analyze(cloud_payload)
        return {
            "image_id": local_result["image_id"],
            "visual_mode": local_result["visual_mode"],
            "evidence_context": local_result["evidence_context"],
            "limitations": local_result["limitations"],
            "results": [local_result, cloud_result],
        }
