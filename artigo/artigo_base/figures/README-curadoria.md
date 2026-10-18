# Capturas da Curadoria Web para o artigo-base

Este diretório contém **quatro capturas reais** da Curadoria Web desenvolvida na pesquisa. As imagens foram fornecidas pelo autor e adicionadas à branch `main` do repositório. O `main.tex` as inclui explicitamente durante a compilação: a falta de qualquer PNG deve ser corrigida, pois as figuras não são mais opcionais. O arquivo `../main.tex` inclui explicitamente as quatro capturas PNG. A publicação do PDF depende da presença dos arquivos reais na pasta.

## Como obter as capturas

Na raiz do repositório, no ambiente que possui os dados de curadoria:

```powershell
.\.venv\Scripts\python.exe .\curadoria_web\app.py
```

Abra `http://127.0.0.1:8765`. Selecione um caso que possua imagens e máscaras reais, verificadas na aplicação. Capture as telas utilizando a ferramenta de captura do Windows ou equivalente, preferencialmente em PNG com resolução suficiente para leitura no PDF.

| Nome do arquivo | Captura necessária | Posição no artigo |
| --- | --- | --- |
| `curadoria_visao_geral.png` | Tela principal, com fila de casos, formulário de revisão e metadados. O painel de progresso/filtros está recolhido. | Materiais e Métodos → Curadoria human-in-the-loop |
| `curadoria_revisao_caso.png` | Tela de visualização ampliada de um caso. As camadas estão desmarcadas no exemplo; não afirmar que há máscaras sobrepostas. | Materiais e Métodos → Curadoria human-in-the-loop |
| `curadoria_correcao_mascara.png` | Ferramenta de pincel configurada para córtex e substituição da máscara; a captura não demonstra edição concluída. | Materiais e Métodos → Curadoria human-in-the-loop |
| `curadoria_rois_ecogenicidade.png` | Comparação manual A/B com CEC (A) e córtex (B), máscaras visíveis e painel de brilho. A razão CEC/córtex mostrada é inversa à razão córtex/CEC do protocolo global. | Materiais e Métodos → Mensuração quantitativa de ecogenicidade |

Salve os quatro arquivos **com os nomes exatos** na pasta `artigo/artigo_base/figures/`. O artigo passa a incorporar automaticamente cada figura existente na próxima compilação, sem precisar alterar `main.tex`.

## Critérios editoriais e de integridade científica

1. Use capturas feitas do **sistema em funcionamento com dados de pesquisa apropriados**, nunca montagens que simulem uma avaliação inexistente.
2. Exclua ou anonimize nomes, IDs e quaisquer informações que possam identificar pacientes. Identificadores técnicos de caso devem ser tratados segundo a autorização dos dados.
3. Recorte para manter texto e interface legíveis, sem ocultar informações necessárias à interpretação do exemplo.
4. No print de correção, confirme que a ferramenta e a estrutura alvo estão visíveis. O print prova a existência da interface; o histórico de auditoria é comprovado pela implementação e requer avaliação própria.
5. Não apresente o painel de ecogenicidade como diagnóstico de fibrose, doença renal crônica ou outro desfecho clínico.
6. Se a captura não documentar exatamente o que a legenda diz, ajuste a legenda antes da publicação.
7. Registre em caderno experimental a data, versão do sistema e origem dos casos usados nas capturas (informações sensíveis não devem ficar no repositório público).

> Nota: As quatro capturas estão no repositório. O LaTeX não as oculta em caso de ausência: um arquivo faltante deve ser corrigido antes da publicação.

## Integração dos quatro arquivos recebidos

Os nomes listados acima correspondem às capturas fornecidas pelo autor nesta revisão. Os quatro PNGs já estão na pasta indicada. Compile novamente `main.tex` para renderizá-los no PDF atualizado. Não altere valores percentuais exibidos nos prints e mantenha a distinção entre a ferramenta de comparação manual A/B e o protocolo automático de ROIs pareadas.
