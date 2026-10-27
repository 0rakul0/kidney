import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from src.kidney_assistant.cloud_app import app
from src.kidney_assistant.schemas import ClinicalFinding, LLMNoteProposal, NormalizedNoteResponse, ChatResponse


def test_rejects_unexpected_llm_fields():
    with pytest.raises(ValidationError):
        LLMNoteProposal.model_validate({'suggested': 'Texto', 'warnings': [], 'diagnosis': 'fibrose'})


def test_response_contracts():
    assert ClinicalFinding(anatomy='rim', attribute='ecogenicidade', value='nao_avaliada').laterality is None
    assert ChatResponse(answer='Sem evidencias suficientes', model='test', image_sent=False).research_only


def test_normalize_endpoint_schema():
    response = TestClient(app).post('/api/notes/normalize', json={
        'image_id': 'A', 'observation': 'Rim direito com ecogenicidade cortical aumentada.'})
    assert response.status_code == 200
    obj = NormalizedNoteResponse.model_validate(response.json())
    assert obj.original == 'Rim direito com ecogenicidade cortical aumentada.'
    assert obj.approved is False
    assert obj.findings[0].laterality == 'direita'


def test_openapi_documents_response_models():
    paths = app.openapi()['paths']
    for path in ['/api/notes/normalize', '/api/notes/luna-suggest', '/api/chat/luna', '/api/chat/luna-image']:
        assert '200' in paths[path]['post']['responses']
        assert '$ref' in paths[path]['post']['responses']['200']['content']['application/json']['schema']
