"""Conservative structured normalization of renal ultrasound specialist notes.

No diagnosis is inferred. Original observations are retained verbatim.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Optional


def fold(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))


@dataclass
class Finding:
    anatomy: str
    attribute: str
    value: str
    laterality: Optional[str]
    certainty: str
    evidence: str


def normalize_note(note: str, image_id: str = '') -> dict:
    if not note.strip():
        raise ValueError('A observacao nao pode estar vazia')
    text = fold(note)
    laterality = 'direita' if re.search(r'\b(direit[oa]|rd)\b', text) else 'esquerda' if re.search(r'\b(esquerd[oa]|re)\b', text) else None
    if re.search(r'\b(direit[oa]|rd)\b', text) and re.search(r'\b(esquerd[oa]|re)\b', text):
        laterality = None  # bilateral or ambiguous; never guess
    uncertain = bool(re.search(r'\b(parece|aparent|possiv|sugest|suspeit|talvez|provav)', text))
    findings = []
    patterns = [
        (r'ecogenicidade\s+(?:cortical\s+)?aumentad[ao]|cortex\s+(?:mais\s+)?(?:claro|hiperecogenico)|hiperecogenic', 'cortex_renal', 'ecogenicidade', 'aumento_relatado'),
        (r'piramid\w*\s+(?:pouco\s+visiveis|mal\s+definid\w*|pouco\s+definid\w*)|pouca\s+definicao\s+d\w*\s+piramid', 'piramides_renais', 'definicao_visual', 'reduzida_relatada'),
    ]
    for pattern, anatomy, attribute, value in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        prefix = text[max(0, match.start()-35):match.start()]
        negated = bool(re.search(r'\b(sem|nao|ausencia de|nega)\s+(?:\w+\s+){0,3}$', prefix))
        findings.append(Finding(anatomy, attribute, 'negado' if negated else value, laterality, 'incerta' if uncertain else 'nao_especificada', note[match.start():match.end()]))
    return {
        'image_id': image_id,
        'original': note,
        'suggested': None,  # never silently rewrite clinical notes
        'findings': [asdict(f) for f in findings],
        'unmapped_text_requires_review': True,
        'approved': False,
        'disclaimer': 'Extracao lexical preliminar; requer revisao do especialista. Nao e diagnostico.'
    }
