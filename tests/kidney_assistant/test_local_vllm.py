import base64
import json
from io import BytesIO
from unittest.mock import patch
from PIL import Image
from src.kidney_assistant import local_vllm
from src.kidney_assistant.schemas import LLMNoteProposal

class Obj:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


def test_structured_note():
    result = Obj(choices=[Obj(message=Obj(content='{"suggested":"Ecogenicidade cortical aparentemente aumentada.","warnings":[]}'))])
    with patch('src.kidney_assistant.local_vllm._client') as client:
        client.return_value.chat.completions.create.return_value = result
        out = local_vllm.suggest_note('Cortex mais claro', [])
        assert out['approved'] is False
        assert out['model'] == local_vllm.MODEL
        kwargs = client.return_value.chat.completions.create.call_args.kwargs
        assert kwargs['extra_body']['structured_outputs']['json'] == LLMNoteProposal.model_json_schema()


def test_image_request():
    result = Obj(choices=[Obj(message=Obj(content='Nao ha evidencias suficientes.'))])
    with patch('src.kidney_assistant.local_vllm._client') as client:
        client.return_value.chat.completions.create.return_value = result
        out = local_vllm.ask_about_session('O que observar?', {}, b'fakepng', 'image/png')
        assert out['image_sent'] is True
        sent = client.return_value.chat.completions.create.call_args.kwargs['messages'][1]['content']
        assert sent[1]['type'] == 'image_url'


def test_content_parts_are_supported():
    message = Obj(content=[Obj(text='parte 1'), {'text': ' parte 2'}])
    assert local_vllm._message_text(message) == 'parte 1 parte 2'


def test_capsule_bbox_crops_before_encoding():
    image = Image.new('RGB', (100, 80), 'white')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    cropped, info = local_vllm._crop_capsule(
        buffer.getvalue(),
        'image/png',
        {'capsule_bbox': {'x1': 20, 'y1': 10, 'x2': 80, 'y2': 60},
         'capsule_padding_ratio': 0},
    )
    with Image.open(BytesIO(cropped)) as result:
        assert result.size == (60, 50)
    assert info['crop_box'] == [20, 10, 80, 60]


def test_image_request_sends_capsule_crop_and_context():
    image = Image.new('RGB', (100, 80), 'white')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    result = Obj(choices=[Obj(message=Obj(content='Recorte recebido.'))])
    with patch('src.kidney_assistant.local_vllm._client') as client:
        client.return_value.chat.completions.create.return_value = result
        local_vllm.ask_about_session(
            'O que observar?',
            {'capsule_bbox': {'x1': 20, 'y1': 10, 'x2': 80, 'y2': 60},
             'capsule_padding_ratio': 0},
            buffer.getvalue(),
            'image/png',
        )
        sent = client.return_value.chat.completions.create.call_args.kwargs['messages'][1]['content']
    sent_image = sent[1]['image_url']['url'].split(',', 1)[1]
    with Image.open(BytesIO(base64.b64decode(sent_image))) as cropped:
        assert cropped.size == (60, 50)
    context = json.loads(sent[0]['text'])['evidencias_da_sessao']['contexto_visual']
    assert context['imagem_enviada'] == 'recorte_da_capsula_renal'
