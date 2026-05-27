# Pipeline anatomico renal e marcadores ultrassonograficos

## Formulacao demonstrada com os dados atuais

O pipeline de estudo passa a ser:

```text
Imagem de ultrassom
-> modelo_rim
-> mascara do rim
-> ROI renal isolada
-> modelo_medula
-> mascara da classe Medulla anotada
-> extracao exploratoria de medidas de ecogenicidade/textura
```

A segmentacao da classe `Medulla` e uma tarefa anatomica demonstravel. Ela nao
deve ser descrita como segmentacao validada de piramides isoladas nem como
evidencia de fibrose renal.

## Trilha futura para fibrose

Na literatura, a fibrose renal e geralmente operacionalizada como
`interstitial fibrosis and tubular atrophy` (`IFTA`), com referencia derivada
de biopsia e predominio de medidas corticais/parenquimatosas. Por isso, uma
trilha clinicamente alinhada deve ser:

```text
imagem de ultrassom
-> mascara renal
-> ROI cortical/parenquimatosa e diferenciacao cortico-medular
-> ecogenicidade relativa e radiomica
-> rotulo clinico/histologico de IFTA
-> estratificacao de fibrose/alteracao cronica
```

A medula permanece util como estrutura auxiliar para medir contraste
cortex-medula e para estudar a anatomia interna, mas nao e o desfecho primario
de fibrose sem validacao clinica.

## Dados disponiveis

O `dataset_geral` possui `4.853` imagens com caminho de mascara renal:

| Origem da mascara renal | Imagens |
| --- | ---: |
| Mascara existente | 1.001 |
| Mascara gerada e aceita pelo modelo do rim | 3.852 |
| Total | 4.853 |

Entre essas entradas, `46` mascaras renais estao vazias e nao produzem uma
ROI utilizavel para o modelo da medula.

O `kidneyUS` fornece mascaras manuais de `Medulla`:

| Referencia | Imagens com `Medulla` |
| --- | ---: |
| Anotador 1 | 336 |
| Anotador 2 | 327 |

O primeiro treinamento usa o anotador 1 sem duplicar imagens entre splits:

| Split | Imagens |
| --- | ---: |
| Treino | 236 |
| Validacao | 50 |
| Teste | 50 |

## Modelos de medula

| Arquitetura | Entrada | Dice teste | IoU teste |
| --- | --- | ---: | ---: |
| Heuristica anterior | ROI renal + intensidade escura | 0.3291 | 0.2057 |
| DeepLabV3-ResNet50 | ROI renal recortada | 0.7528 | 0.6035 |
| MedullaROIUNet | ROI, ROI mascarada, mascara renal | 0.7326 | 0.5780 |

O `DeepLabV3-ResNet50` e atualmente o melhor resultado quantitativo. A
`MedullaROIUNet` representa explicitamente o pipeline em cascata, pois recebe
a mascara do rim como entrada e impede predicoes fora da ROI renal.

## Pseudo-mascaras candidatas

O modelo DeepLab de medula produziu:

| Status | Quantidade |
| --- | ---: |
| Candidatas sobre mascara renal existente | 942 |
| Candidatas sobre mascara renal gerada, exigindo revisao da ROI | 3.610 |
| Sem ROI, predicao vazia ou area insuficiente | 301 |

A `MedullaROIUNet`, em saida independente, produziu:

| Status | Quantidade |
| --- | ---: |
| Candidatas sobre mascara renal existente | 954 |
| Candidatas sobre mascara renal gerada, exigindo revisao da ROI | 3.731 |
| Sem ROI, predicao vazia ou area insuficiente | 168 |

Nenhuma pseudo-mascara deve entrar automaticamente no treino. A expansao mais
conservadora comeca pela revisao das candidatas baseadas em mascaras renais
existentes.

## Estabilidade da cascata

Em um holdout de `50` imagens do anotador 1, o modelo do rim atingiu Dice
global `0.9433` contra a capsula manual. Usando o DeepLab de medula, a
substituicao da ROI renal manual pela ROI produzida pelo modelo do rim mudou o
Dice global da medula de `0.7626` para `0.7610`.

| Modelo de medula | Dice com ROI manual | Dice com ROI do modelo do rim | Correlacao da opacidade media em cascata |
| --- | ---: | ---: | ---: |
| DeepLabV3-ResNet50 | 0.7626 | 0.7610 | 0.9508 |
| MedullaROIUNet | 0.7418 | 0.7300 | 0.9444 |

O DeepLab permanece o segmentador principal. A ROIUNet continua util como
controle de estabilidade da opacidade e como segundo voto para selecionar
pseudo-mascaras.

## Lote de revisao por consenso

Considerando somente mascaras renais existentes, `941` pseudo-mascaras foram
produzidas por ambos os modelos. O filtro `Dice entre modelos >= 0.75`
selecionou `457` imagens para a primeira rodada de revisao:

```text
results/intrarenal_model3/medulla_consensus_review/selected_for_review.csv
```

O resumo detalhado da estabilidade e as ressalvas metodologicas estao em
`docs/resultados_estabilidade_medula.md`.

## Expansao controlada de Medulla

Das `457` candidatas iniciais, `342` passaram em um filtro estrito de consenso
e geometria e foram incorporadas apenas ao treino experimental; validacao e
teste permaneceram manuais. O DeepLab retreinado manteve Dice de teste
`0.7523`, comparavel ao baseline `0.7528`.

Aplicado novamente nas imagens com mascara renal existente, o modelo expandido
produziu `951` candidatas. O novo cruzamento com a ROIUNet gerou `599`
candidatas para revisao, incluindo `162` novas imagens. Destas, `474` passam
no filtro estrito para uma possivel expansao v2, ainda dependente de auditoria
humana antes de novo treinamento.

Resultados detalhados:
`docs/resultados_expansao_pseudomascaras_medulla.md`.

## Estudo exploratorio de ecogenicidade

Com mascaras anatomicas revisadas, a primeira analise exploratoria pode medir:

- intensidade media e mediana dentro de `Medulla` e `Cortex`;
- percentis de intensidade e dispersao local;
- razao de intensidade `cortex/medulla` e contraste cortico-medular;
- razao `medula/complexo_ecogenico_central`, quando disponivel;
- textura por compartimento e no parenquima renal.

Essas medidas devem ser descritas inicialmente como caracterizacao
ultrassonografica anatomica, nao como diagnostico de fibrose. A revisao
medico-metodologica completa esta em
`docs/revisao_medico_metodologica_fibrose.md`.
