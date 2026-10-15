# Capturas da Curadoria Web para o artigo-base

Este diretório armazena as **capturas reais** da interface de Curadoria Web desenvolvida na pesquisa. O arquivo `../main.tex` contém quatro blocos LaTeX condicionais (`\IfFileExists`) que somente entram no PDF quando as respectivas imagens PNG estão presentes. Isso evita figuras fictícias, referências quebradas e erros de compilação enquanto as capturas não forem produzidas.

## Como obter as capturas

Na raiz do repositório, no ambiente que possui os dados de curadoria:

```powershell
.\.venv\Scripts\python.exe .\curadoria_web\app.py
```

Abra `http://127.0.0.1:8765`. Selecione um caso que possua imagens e máscaras reais, verificadas na aplicação. Capture as telas utilizando a ferramenta de captura do Windows ou equivalente, preferencialmente em PNG com resolução suficiente para leitura no PDF.

| Nome do arquivo | Captura necessária | Posição no artigo |
| --- | --- | --- |
| `curadoria_visao_geral.png` | Tela com fila de casos, filtros e resumo do progresso; evitar tela vazia. | Materiais e Métodos → Curadoria human-in-the-loop |
| `curadoria_revisao_caso.png` | Caso de ultrassom com sobreposição das camadas e controles de revisão das máscaras. | Materiais e Métodos → Curadoria human-in-the-loop |
| `curadoria_correcao_mascara.png` | Edição de uma máscara por polígono/pincel/borracha; mostrar os controles da operação. | Materiais e Métodos → Curadoria human-in-the-loop |
| `curadoria_rois_ecogenicidade.png` | ROIs córtex--CEC visíveis e painel de ecogenicidade com referência ao CEC. | Materiais e Métodos → Mensuração quantitativa de ecogenicidade |

Salve os quatro arquivos **com os nomes exatos** na pasta `artigo/artigo_base/figures/`. O artigo passa a incorporar automaticamente cada figura existente na próxima compilação, sem precisar alterar `main.tex`.

## Critérios editoriais e de integridade científica

1. Use capturas feitas do **sistema em funcionamento com dados de pesquisa apropriados**, nunca montagens que simulem uma avaliação inexistente.
2. Exclua ou anonimize nomes, IDs e quaisquer informações que possam identificar pacientes. Identificadores técnicos de caso devem ser tratados segundo a autorização dos dados.
3. Recorte para manter texto e interface legíveis, sem ocultar informações necessárias à interpretação do exemplo.
4. No print de correção, confirme que a ferramenta e a estrutura alvo estão visíveis. O print prova a existência da interface; o histórico de auditoria é comprovado pela implementação e requer avaliação própria.
5. Não apresente o painel de ecogenicidade como diagnóstico de fibrose, doença renal crônica ou outro desfecho clínico.
6. Se a captura não documentar exatamente o que a legenda diz, ajuste a legenda antes da publicação.
7. Registre em caderno experimental a data, versão do sistema e origem dos casos usados nas capturas (informações sensíveis não devem ficar no repositório público).

> Nota: As capturas ainda não foram inseridas nesta atualização. O arquivo LaTeX permanece compilável sem elas, porque as figuras são condicionais.
