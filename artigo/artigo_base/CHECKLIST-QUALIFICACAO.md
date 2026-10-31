# Checklist para a banca de qualificação — ultrassonografia renal

**Status:** revisão do artigo-base em 31/10/2026. Esta lista diferencia o que está implementado no texto, a adequação formal dependente do formato aceito pelo PPCIC e os experimentos ainda necessários. **Não é uma lista de exigências obrigatórias para todas as qualificações.**

## Já presente no artigo-base

- [x] Pergunta de pesquisa, hipótese, objetivos e delimitação do escopo
- [x] Fundamentação e trabalhos relacionados
- [x] Metodologia e distinção das tarefas/anotações/testes
- [x] Resultados, discussão, limitações e plano de continuidade
- [x] Resumo em português e abstract em inglês
- [x] Sumário navegável com seções e subseções
- [x] Referências e exemplo multimodal como apêndice da qualificação

## Elementos pré-textuais acrescentados

- [x] Palavras-chave e *keywords* (apenas no modo `\versaoqualificacaotrue`)
- [x] Lista alfabética de abreviaturas e siglas (17 entradas)
- [x] Lista de símbolos (16 entradas)
- [x] Lista de figuras (16 ilustrações, com títulos curtos)
- [x] Lista de tabelas (16 tabelas, com títulos curtos)
- [ ] Confirmar a compilação mais recente e conferir visualmente todas as novas listas

As listas de figuras, tabelas, abreviaturas/siglas e símbolos são **opcionais** pela ABNT NBR 14724:2024 e pelo manual institucional do CEFET/RJ (edição disponível de 2016); sua inclusão aqui é uma escolha editorial adequada ao volume do trabalho.

## Pendências formais, condicionadas ao tipo de entrega

- [ ] Confirmar com a orientação se o texto da **Defesa de Proposta de Dissertação** pode seguir o artigo-base SBC ou se deve adotar integralmente o template institucional de dissertação
- [x] Estrutura da proposta: capa e folha de rosto com identificação do discente, PPCIC, natureza do trabalho, cidade e ano; orientação preenchida com Felipe da R. Henriques
- [x] Registrar Felipe da R. Henriques como orientador(a) na folha de rosto
- [ ] Conferir a grafia completa do nome do discente e do orientador para a entrega oficial
- [x] Resumo em português antes do Abstract, em páginas próprias; margens de 3 cm, Times 12 aproximado e espaçamento 1,5 no modo de qualificação
- [x] Pré-textuais sem numeração visível; numeração no alto à direita a partir da Introdução
- [ ] Confirmar aderência final do PDF ao padrão específico aceito pelo PPCIC
- [x] Separar citações autor-data via `abntex2cite` na qualificação e `sbc.bst` no artigo SBC
- [ ] Conferir campos e apresentação final da bibliografia em relação às normas vigentes e às orientações da coordenação
- [ ] Conferir numeração, fonte e localização de todas as figuras e tabelas; registrar a autoria/fonte mesmo quando a imagem é de elaboração própria
- [ ] Conferir identificação e expansão das siglas no primeiro uso, além da lista pré-textual
- [ ] Conferir nome do programa, título oficial, resumo em português/inglês, orientação e comunicação à coordenação, conforme o procedimento da banca
- [ ] Para a **versão final da dissertação**, providenciar folha de aprovação/assinaturas, ficha catalográfica da biblioteca e demais exigências institucionais no momento apropriado; não antecipar assinatura ou ficha fictícia

## Pendências científicas prioritárias

- [ ] Congelar manifestos, checkpoints, protocolos e limiares antes de avaliação final
- [ ] Selecionar combinações de professores **somente pela validação**; executar teste independente não consultado na escolha
- [ ] Reportar variação por exame/paciente, intervalos de confiança e, quando viável, múltiplas sementes
- [ ] Construir um subconjunto externo MONAI com máscaras de referência revisadas para quantificar qualidade anatômica; não confundir máscara não vazia com máscara correta
- [ ] Medir benefício real da Curadoria Web: aceitação, correção, rejeição, tempo e concordância entre revisores
- [ ] Comparar treino manual × manual+pseudo × manual+pseudo revisado, preservando as mesmas condições de avaliação
- [ ] Medir repetibilidade e sensibilidade das razões de ecogenicidade (máscaras, tamanho/posição de ROI, profundidade, variação intra/interobservador)
- [ ] Definir, com orientador, escopo e condições de eventual estudo clínico pareado (exames, eGFR, histopatologia, governança de dados e ética)
- [x] Incluir cronograma preliminar de novembro/2026 a abril/2027 com prazos e entregáveis verificáveis
- [ ] Validar as datas indicativas com o orientador e com o calendário institucional
- [ ] Manter o piloto multimodal como demonstração exploratória até existir amostra, gabarito e documentação técnica do modelo externo

## Documentos de referência

- [PPCIC — Documentos e regras para Proposta de Dissertação](https://eic.cefet-rj.br/ppcic/documentos/)
- [PPCIC — Divulgação de qualificação](https://eic.cefet-rj.br/ppcic/divulgacao-no-site/)
- [Manual CEFET/RJ de elaboração de teses, dissertações e monografias (2016)](https://eic.cefet-rj.br/ppcic/wp-content/uploads/2018/07/Manual-de-Elaboracao-de-Teses-Dissertacoes-e-Monografias.pdf)
- [ABNT NBR 14724:2024 — orientações de estrutura](https://sddarquivos.webhostusp.sti.usp.br/arquivos/Guia_Teses_Dissertacoes_convencional_FOB-USP.html)

**Importante:** o manual CEFET/RJ publicado no site tem edição de 2016; a ABNT NBR 14724 foi atualizada em 2024. Caso haja conflito entre modelo institucional disponível, norma atual e orientação local da banca, confirmar com a secretaria e o orientador antes de converter o artigo-base em dissertação formal.
