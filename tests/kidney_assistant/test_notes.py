from src.kidney_assistant.clinical_notes import normalize_note

def test_original_and_approval():
    note='Rim direito com ecogenicidade cortical aumentada.'
    result=normalize_note(note, 'A')
    assert result['original']==note
    assert result['approved'] is False
    assert result['findings'][0]['laterality']=='direita'

def test_negation():
    result=normalize_note('Rim esquerdo sem ecogenicidade cortical aumentada.')
    assert result['findings'][0]['value']=='negado'

def test_uncertainty():
    result=normalize_note('Parece ter piramides pouco visiveis.')
    assert result['findings'][0]['certainty']=='incerta'


def test_laterality_is_local_to_each_finding():
    result = normalize_note(
        'Rim direito com ecogenicidade cortical aumentada. '
        'Rim esquerdo sem ecogenicidade cortical aumentada.'
    )
    assert [finding['laterality'] for finding in result['findings']] == ['direita', 'esquerda']
    assert result['findings'][1]['value'] == 'negado'


def test_ecogenic_focus_is_uncertain_and_localized():
    result = normalize_note('Achado possivel no rim esquerdo com foco hiperecogenico.')
    finding = result['findings'][0]
    assert finding['laterality'] == 'esquerda'
    assert finding['certainty'] == 'incerta'
    assert finding['attribute'] == 'foco_ecogenico_ou_artefato'
