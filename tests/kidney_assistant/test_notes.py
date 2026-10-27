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
