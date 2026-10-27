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


def _sentence_context(text: str, start: int, end: int) -> str:
    """Return the clause containing one finding, avoiding global laterality."""
    left = max(text.rfind(mark, 0, start) for mark in '.;!?\n')
    right_candidates = [text.find(mark, end) for mark in '.;!?\n']
    right_candidates = [value for value in right_candidates if value >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right]


def _local_laterality(context: str, text: str, start: int) -> Optional[str]:
    if re.search(r'\b(bilateral|bilaterais|ambos|ambas)\b', context):
        return 'bilateral'
    if re.search(r'\b(direit[oa]|rim\s*D|rd)\b', context):
        return 'direita'
    if re.search(r'\b(esquerd[oa]|rim\s*E|re)\b', context):
        return 'esquerda'
    # A short look-behind handles clauses such as "rim direito ... e ...".
    prefix = text[max(0, start - 55):start]
    right = prefix.lower().rfind('direit')
    left = prefix.lower().rfind('esquerd')
    if right >= 0 and right > left:
        return 'direita'
    if left >= 0:
        return 'esquerda'
    return None


def _is_uncertain(context: str) -> bool:
    return bool(re.search(r'\b(parece|aparent|possiv|sugest|suspeit|talvez|provav|pode ser|poderia)\w*', context))


def _is_negated(context: str, match_start: int) -> bool:
    prefix = context[:match_start]
    return bool(re.search(r'\b(sem|nao|ausencia\s+de|nega(?:-se)?)\s+(?:\w+\s+){0,4}$', prefix))


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
    findings = []
    patterns = [
        (r'ecogenicidade\s+(?:cortical\s+)?(?:aumentad[ao]|elevad[ao]|preservad[ao]|reduzid[ao])|'
         r'cortex\s+(?:mais\s+)?(?:claro|hiperecogen\w*|ecogen\w*)|'
         r'hiperecogenic(?:idade|o|a)?\s+(?:cortical|do\s+cortex)',
         'cortex_renal', 'ecogenicidade', 'relato_qualitativo'),
        (r'piramid\w*\s+(?:pouco\s+visiveis|mal\s+definid\w*|pouco\s+definid\w*|proeminentes)\s*|'
         r'pouca\s+definicao\s+d\w*\s+piramid',
         'piramides_renais', 'definicao_visual', 'relato_qualitativo'),
        (r'calcificad\w*|foco(?:s)?\s+(?:hiperecogen\w*|ecogen\w*)|'
         r'artefato(?:s)?\s+(?:posterior(?:es)?\s+)?(?:reverberac\w*|sombra)',
         'estrutura_renal', 'foco_ecogenico_ou_artefato', 'relato_qualitativo'),
    ]
    for pattern, anatomy, attribute, value in patterns:
        for match in re.finditer(pattern, text):
            context = _sentence_context(text, match.start(), match.end())
            local_start = context.find(match.group(0))
            local_start = 0 if local_start < 0 else local_start
            negated = _is_negated(context, local_start)
            findings.append(Finding(
                anatomy,
                attribute,
                'negado' if negated else value,
                _local_laterality(context, text, match.start()),
                'incerta' if _is_uncertain(context) else 'nao_especificada',
                note[match.start():match.end()],
            ))
    return {
        'image_id': image_id,
        'original': note,
        'suggested': None,  # never silently rewrite clinical notes
        'findings': [asdict(f) for f in findings],
        'unmapped_text_requires_review': True,
        'approved': False,
        'disclaimer': 'Extracao lexical preliminar; requer revisao do especialista. Nao e diagnostico.'
    }
