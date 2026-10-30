from pathlib import Path

import numpy as np
from PIL import Image

from curadoria_web.assistant import AssistantService
from curadoria_web.inference import InferenceService


def test_region_assessment_marks_low_confidence_for_review():
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[1:4, 1:4] = 1
    labels[4:7, 1:4] = 2
    labels[7:9, 1:4] = 3
    kidney = labels > 0
    probabilities = np.zeros((4, 10, 10), dtype=np.float32)
    probabilities[0] = 0.1
    probabilities[1, labels == 1] = 0.90
    probabilities[2, labels == 2] = 0.60
    probabilities[3, labels == 3] = 0.85

    result = InferenceService._assess_regions(labels, probabilities, kidney)

    by_name = {item["regiao"]: item for item in result}
    assert by_name["cortex"]["prioridade"] == "baixa"
    assert by_name["medulla"]["prioridade"] == "média"
    assert "confiança intermediária" in by_name["medulla"]["motivos"][0]
    assert by_name["central_echo_complex"]["confianca"] == 0.85


def test_quality_assessment_is_explicitly_non_diagnostic():
    kidney = np.ones((20, 20), dtype=bool)
    probability = np.full((20, 20), 0.8, dtype=np.float32)
    quality = InferenceService._quality_assessment(
        probability,
        kidney,
        {"x1": 0, "y1": 0, "x2": 64, "y2": 64},
        [],
    )

    assert quality["status"] == "sem_alerta_automatico"
    assert quality["revisao_humana_obrigatoria"] is True
    assert "não é probabilidade diagnóstica" in quality["interpretacao"]


def test_distilled_checkpoint_is_described_for_traceability():
    descriptor = InferenceService._model_descriptor(
        Path("models/kidneyus_capsule_student_unet_deeplab.pth"),
        "localização da cápsula renal",
    )

    assert descriptor["destilado"] is True
    assert "U-Net estudante destilada" in descriptor["nome"]
    assert descriptor["checkpoint"] == "kidneyus_capsule_student_unet_deeplab.pth"


def test_structured_assistant_response_reports_regions_and_evidence():
    result = AssistantService._structured_response(
        "**Observação:** revisar o córtex e o CEC.",
        {"limitacoes": ["Revisão humana obrigatória."]},
        {
            "evidence_refs": ["imagem_original", "mascara_cortex"],
            "priorizacao_regioes": [{"regiao": "cortex", "prioridade": "alta", "motivos": ["incerteza"]}],
        },
    )

    assert result["regioes_mencionadas"] == ["cortex", "central_echo_complex"]
    assert result["regioes_priorizadas"][0]["prioridade"] == "alta"
    assert result["status"] == "revisao_humana_obrigatoria"


def test_structured_response_links_chat_regions_to_segmentation_evidence():
    result = AssistantService._structured_response(
        "Revisar o córtex e o CEC.",
        {"limitacoes": ["Revisão humana obrigatória."]},
        {
            "evidence_refs": ["imagem_original", "mascara_cortex", "mascara_central_echo_complex"],
            "mascaras_disponiveis": ["cortex", "central_echo_complex"],
            "modelos_segmentacao": {"capsula": {"nome": "U-Net estudante destilada"}},
            "qualidade_inferencia": {"status": "revisar_com_atencao"},
            "medidor_brilho": {"diferenca_cec_cortex": 12.4},
        },
    )

    assert result["regioes_com_evidencia"] == ["cortex", "central_echo_complex"]
    assert result["regioes_sem_evidencia"] == []
    assert result["modelos_segmentacao"]["capsula"]["nome"].startswith("U-Net")
    assert result["metricas_ecogenicidade"]["diferenca_cec_cortex"] == 12.4


def test_mask_command_is_a_confirmation_dependent_proposal():
    command = AssistantService.parse_mask_command("pode melhorar a mascara do cortex?")

    assert command["layer"] == "cortex"
    assert command["operation"] == "refinar_mascara"
    assert command["requires_confirmation"] is True


def test_mask_proposal_preserves_original_until_apply(tmp_path):
    result_dir = tmp_path / "infer_case"
    masks_dir = result_dir / "masks"
    masks_dir.mkdir(parents=True)
    image_path = result_dir / "original.png"
    Image.new("RGB", (32, 32), (80, 80, 80)).save(image_path)

    kidney = np.zeros((32, 32), dtype=np.uint8)
    kidney[4:28, 4:28] = 255
    cortex = np.zeros((32, 32), dtype=np.uint8)
    cortex[10:20, 10:20] = 255
    cortex[6, 6] = 255  # ruído isolado que a proposta deve remover
    mask_paths = {}
    for name, data in {
        "rim": kidney,
        "cortex": cortex,
        "medulla": np.zeros((32, 32), dtype=np.uint8),
        "central_echo_complex": np.zeros((32, 32), dtype=np.uint8),
    }.items():
        path = masks_dir / f"{name}.png"
        Image.fromarray(data).save(path)
        mask_paths[name] = path

    service = InferenceService(tmp_path, tmp_path / "outputs", None)
    service.results["infer_case"] = {
        "image_id": "infer_case",
        "image_path": image_path,
        "mask_paths": mask_paths.copy(),
        "active_mask_paths": {},
        "filename": "case.png",
        "meter": {},
        "width": 32,
        "height": 32,
        "device": "cpu",
        "capsule_bbox": None,
        "quality": {},
        "region_review": [],
        "uncertainty_path": None,
    }

    proposal = service.propose_mask("infer_case", "cortex", "remover_ruido", "jefferson")
    assert proposal["requires_confirmation"] is True
    assert proposal["status"] == "pendente"
    assert mask_paths["cortex"].exists()
    assert "approved_masks" not in str(mask_paths["cortex"])

    applied = service.apply_mask_proposal("infer_case", proposal["proposal_id"], "jefferson")
    assert applied["status"] == "aplicada"
    assert service.results["infer_case"]["mask_paths"]["cortex"].parent.name == "approved_masks"
