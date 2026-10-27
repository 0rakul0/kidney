"""Kidney research API; explicit consent is required before external LLM calls."""
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from .schemas import NoteIn, LunaNoteIn, ChatIn, NormalizedNoteResponse, ChatResponse
from .clinical_notes import normalize_note
from .luna import suggest_note, ask_about_session

app = FastAPI(title='Kidney Assistant - research prototype', version='0.2.0')

def require_consent(consent: bool):
    if not consent:
        raise HTTPException(status_code=403, detail='Consentimento explicito para envio a API externa obrigatorio')


def external_error(exc: Exception):
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError) and 'OPENAI_API_KEY' in str(exc):
        raise HTTPException(status_code=503, detail='Chave de API nao configurada') from exc
    raise HTTPException(status_code=502, detail='Falha ao consultar o provedor de IA') from exc

@app.post('/api/notes/normalize', response_model=NormalizedNoteResponse)
def normalize(data: NoteIn):
    try:
        return normalize_note(data.observation, data.image_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post('/api/notes/luna-suggest', response_model=NormalizedNoteResponse)
def luna_suggest(data: LunaNoteIn):
    require_consent(data.consent_external_processing)
    baseline = normalize_note(data.observation, data.image_id)
    try:
        baseline['llm_proposal'] = suggest_note(data.observation, baseline['findings'])
        return baseline
    except Exception as exc:
        external_error(exc)

@app.post('/api/chat/luna', response_model=ChatResponse)
def luna_chat(data: ChatIn):
    require_consent(data.consent_external_processing)
    try:
        return ask_about_session(data.question, data.evidence)
    except Exception as exc:
        external_error(exc)

@app.post('/api/chat/luna-image', response_model=ChatResponse)
async def luna_image(question: str = Form(...), evidence_json: str = Form('{}'),
                     consent_external_processing: bool = Form(False),
                     image: UploadFile = File(...)):
    require_consent(consent_external_processing)
    if image.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise HTTPException(status_code=415, detail='Use PNG, JPEG ou WebP')
    raw = await image.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Imagem acima de 5 MB')
    try:
        evidence = json.loads(evidence_json)
        if not isinstance(evidence, dict):
            raise ValueError('evidence_json deve ser objeto JSON')
        return ask_about_session(question, evidence, raw, image.content_type)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        external_error(exc)
