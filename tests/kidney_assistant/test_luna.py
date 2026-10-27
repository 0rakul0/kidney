import json
import pytest
from src.kidney_assistant import luna
from src.kidney_assistant.cloud_app import require_consent
from fastapi import HTTPException

class FakeResponse:
    output_text = json.dumps({'suggested':'Sem alteracoes relatadas.', 'warnings':[]})
class FakeResponses:
    def __init__(self): self.kwargs = None
    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()
class FakeClient:
    def __init__(self): self.responses = FakeResponses()

def test_consent_required():
    with pytest.raises(HTTPException) as err:
        require_consent(False)
    assert err.value.status_code == 403

def test_suggest_does_not_approve(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(luna, '_client', lambda: client)
    result = luna.suggest_note('Sem alteracoes relatadas.', [])
    assert result['approved'] is False
    assert client.responses.kwargs['store'] is False
    assert client.responses.kwargs['model'] == 'gpt-6-luna'

def test_image_is_optional(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(luna, '_client', lambda: client)
    result = luna.ask_about_session('Quais achados?', {})
    assert result['image_sent'] is False
    assert len(client.responses.kwargs['input'][0]['content']) == 1

def test_invalid_image_type_rejected():
    with pytest.raises(ValueError):
        luna.ask_about_session('Teste', {}, b'abc', 'image/gif')
