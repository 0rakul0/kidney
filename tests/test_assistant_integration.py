from io import BytesIO
from pathlib import Path

from PIL import Image

from curadoria_web.assistant import AssistantService


class FakeStore:
    by_id = {"case-1": {}}

    def __init__(self, image, masks):
        self.image = image
        self.masks = masks
        self.saved = []

    def get_item(self, image_id, reviewer, model):
        return {
            "image_id": image_id,
            "origem_visual": "manual_full_resolution",
            "info": {"largura": 64, "altura": 48},
            "medidor_brilho": {"disponivel": True, "percentual": 54.9},
            "metricas": {"cortex": {"dice": 0.8}},
        }

    def source_file(self, image_id, kind):
        return self.image if kind == "image" else None

    def effective_mask(self, image_id, kind, reviewer, model):
        return self.masks.get(kind)

    def save_assistant_run(self, result, reviewer, question):
        self.saved.append((result, reviewer, question))


class FakeInference:
    results = {}


def png(path, size, color=255):
    Image.new("L", size, color).save(path, format="PNG")


def test_build_request_recovers_artifacts_and_builds_overlay(tmp_path):
    image_path = tmp_path / "image.png"
    png(image_path, (64, 48), 80)
    masks = {}
    for name in ("rim", "cortex", "medulla", "central_echo_complex"):
        path = tmp_path / f"{name}.png"
        png(path, (64, 48), 0)
        Image.new("L", (64, 48), 255).save(path, format="PNG")
        masks[name] = path
    service = AssistantService(FakeStore(image_path, masks), FakeInference())
    raw, mime, evidence, context = service.build_request("case-1", "Revisor 01", visual_mode="capsule")
    assert mime == "image/png"
    assert Image.open(BytesIO(raw)).size == (64, 48)
    assert evidence["sobreposicao_mascaras_enviada"] is True
    assert evidence["capsule_bbox"] == {"x1": 0, "y1": 0, "x2": 64, "y2": 48}
    assert context["evidence_refs"]
