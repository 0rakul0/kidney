"""Pydantic v2 contracts for the renal ultrasound research assistant."""
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ClinicalFinding(StrictSchema):
    anatomy: str
    attribute: str
    value: str
    laterality: str | None = None
    certainty: str = 'nao_especificada'
    evidence: str = ''


class NoteIn(StrictSchema):
    image_id: str = Field(default='', max_length=100)
    observation: str = Field(min_length=1, max_length=10000)


class LunaNoteIn(NoteIn):
    consent_external_processing: bool = False


class ChatIn(StrictSchema):
    question: str = Field(min_length=1, max_length=4000)
    evidence: dict[str, Any] = Field(default_factory=dict)
    consent_external_processing: bool = False


class LLMNoteProposal(StrictSchema):
    suggested: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class LunaNoteProposal(LLMNoteProposal):
    model: str
    approved: bool = False


class NormalizedNoteResponse(StrictSchema):
    image_id: str
    original: str
    suggested: str | None = None
    findings: list[ClinicalFinding] = Field(default_factory=list)
    unmapped_text_requires_review: bool = True
    approved: bool = False
    disclaimer: str
    llm_proposal: LunaNoteProposal | None = None


class ChatResponse(StrictSchema):
    answer: str
    model: str
    image_sent: bool
    research_only: bool = True
