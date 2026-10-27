"""Local-only API: same Pydantic contracts as cloud version."""
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from .schemas import NoteIn, ChatIn, NormalizedNoteResponse, ChatResponse
from .clinical_notes import normalize_note
from . import local_vllm

app = FastAPI(title='Kidney Assistant - vLLM local', version='0.3.0')


def _fail(exc):
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=502, detail='Falha no servidor vLLM local') from exc


def _normalize(data: NoteIn):
    try:
        return normalize_note(data.observation, data.image_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/health')
def health():
    """Report that the HTTP wrapper is running; it does not call the GPU."""
    return {'status': 'ok', 'provider': 'local-vllm', 'model': local_vllm.MODEL}


@app.post('/api/notes/normalize', response_model=NormalizedNoteResponse)
def normalize(data: NoteIn):
    return _normalize(data)


@app.post('/api/notes/local-suggest', response_model=NormalizedNoteResponse)
def local_suggest(data: NoteIn):
    try:
        result = _normalize(data)
        result['llm_proposal'] = local_vllm.suggest_note(data.observation, result['findings'])
        return result
    except Exception as exc:
        _fail(exc)


@app.post('/api/chat/local', response_model=ChatResponse)
def local_chat(data: ChatIn):
    try:
        return local_vllm.ask_about_session(data.question, data.evidence)
    except Exception as exc:
        _fail(exc)


@app.post('/api/chat/local-image', response_model=ChatResponse)
async def local_image(question: str = Form(...), evidence_json: str = Form('{}'),
                      image: UploadFile = File(...)):
    if image.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise HTTPException(status_code=415, detail='Use PNG, JPEG ou WebP')
    raw = await image.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Imagem acima de 5 MB')
    try:
        evidence = json.loads(evidence_json)
        if not isinstance(evidence, dict):
            raise ValueError('evidence_json deve ser objeto JSON')
        return local_vllm.ask_about_session(question, evidence, raw, image.content_type)
    except Exception as exc:
        _fail(exc)
