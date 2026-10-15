# Capturas da Curadoria Web para o artigo-base

Este diretório receberá as **capturas reais** da interface de Curadoria Web desenvolvida na pesquisa. As quatro capturas foram fornecidas pelo autor para inclusão no artigo e devem ser adicionadas ao repositório como arquivos de imagem. O arquivo `../main.tex` contém quatro blocos LaTeX condicionais (`\IfFileExists`) que somente entram no PDF quando as respectivas imagens PNG estão presentes. Isso evita figuras fictícias, referências quebradas e erros de compilação enquanto as capturas não forem produzidas.

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

> Nota: As capturas ainda não foram inseridas nesta atualização. O arquivo LaTeX permanece compilável sem elas, porque as figuras são condicionais.

## Integração dos quatro arquivos recebidos

Os nomes listados acima correspondem às capturas fornecidas pelo autor nesta revisão. Após salvar os quatro PNGs na pasta indicada, compile novamente `main.tex` para ativar as figuras. Não altere valores percentuais exibidos nos prints e mantenha a distinção entre a ferramenta de comparação manual A/B e o protocolo automático de ROIs pareadas.
