from unittest.mock import patch

from fastapi.testclient import TestClient

from src.kidney_assistant.local_app import app


client = TestClient(app)


def test_health_is_local_only():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['provider'] == 'local-vllm'


def test_whitespace_note_is_client_error():
    response = client.post('/api/notes/local-suggest', json={
        'observation': '   ',
    })
    assert response.status_code == 422


def test_local_suggest_keeps_pending_approval():
    proposal = {
        'suggested': 'Sem alteracoes relatadas.',
        'warnings': [],
        'model': 'test-model',
        'approved': False,
    }
    with patch('src.kidney_assistant.local_app.local_vllm.suggest_note', return_value=proposal):
        response = client.post('/api/notes/local-suggest', json={
            'image_id': 'US_001',
            'observation': 'Sem alteracoes relatadas.',
        })
    assert response.status_code == 200
    assert response.json()['llm_proposal']['approved'] is False
