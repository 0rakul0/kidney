# Expansao controlada de pseudo-mascaras de Medulla

## Objetivo

Esta etapa amplia o treino do segmentador de `Medulla` usando apenas
pseudo-mascaras produzidas sobre mascaras renais existentes e selecionadas por
concordancia entre dois modelos:

```text
DeepLab de Medulla + MedullaROIUNet
-> consenso dentro da mascara renal existente
-> auditoria visual
-> expansao experimental do treino
```

As novas mascaras nao sao consideradas anotacoes manuais. Todas permanecem
marcadas como `pending_clinical_audit`.

## Rodada inicial de consenso

O filtro inicial encontrou `457` imagens com Dice entre modelos de pelo menos
`0.75`. Nenhuma delas e duplicata por conteudo das `336` imagens manuais
separadas em treino, validacao e teste.

Para uso automatico no treino, foi aplicado um filtro mais estrito:

| Criterio | Regra |
| --- | ---: |
| Dice entre modelos | `>= 0.78` |
| Area de Medulla em relacao ao rim | `>= 0.10` |
| Componentes conectados | `<= 3` |
| Fracao do maior componente | `>= 0.80` |

| Resultado v1 | Imagens |
| --- | ---: |
| Candidatas para auditoria | 457 |
| Pseudo-mascaras adicionadas ao treino | 342 |
| Retidas para auditoria prioritaria | 115 |
| Treino manual original | 236 |
| Treino expandido v1 | 578 |
| Validacao manual preservada | 50 |
| Teste manual preservado | 50 |

O alvo pseudo-rotulado usado no treino foi a predicao do DeepLab, pois ele teve
melhor Dice supervisionado; a ROIUNet foi usada como modelo de controle para
aceitar somente casos concordantes.

Arquivos:

```text
dataset_aumentado/dataset_intrarrenal/pseudo_expandido/medulla_expanded_consensus_v1/
results/intrarenal_model3/medulla_consensus_review/audit_packet_v1/
```

## Treinamento com expansao v1

Foi treinado um novo DeepLabV3-ResNet50 usando as `342` pseudo-mascaras
estritas no treino e mantendo validacao/teste exclusivamente manuais:

| Modelo | Treino | Val Dice | Teste Dice | Teste IoU |
| --- | ---: | ---: | ---: | ---: |
| Baseline manual | 236 | 0.7235 | 0.7528 | 0.6035 |
| Expansao consenso v1 | 578 | 0.7424 | 0.7523 | 0.6030 |

O resultado indica que a expansao nao prejudicou o teste manual, embora tambem
nao tenha aumentado o Dice de teste. O ganho mais claro foi na validacao e na
cobertura de novas candidatas.

Na avaliacao em cascata usando a mascara prevista do rim:

| Modelo | Dice global vs. anotador 1 | Dice global vs. anotador 2 |
| --- | ---: | ---: |
| Baseline manual | 0.7610 | 0.6996 |
| Expansao consenso v1 | 0.7618 | 0.7045 |

A medida de intensidade media em cascata teve correlacao menor no modelo
expandido (`0.9227` contra `0.9508` no baseline). Por isso, o modelo expandido
pode ser usado para ampliar candidatas de segmentacao, mas nao substitui o
baseline na definicao da futura medida de ecogenicidade.

## Expansao para imagens com mascara renal existente

O modelo v1 foi aplicado nas `4.853` imagens que possuem alguma mascara renal.
As predicoes sobre mascara renal gerada continuam fora de uso automatico, pois
necessitam validacao anatomica da ROI.

| Saida do modelo v1 | Imagens |
| --- | ---: |
| Candidatas sobre mascara renal existente | 951 |
| Candidatas sobre mascara renal gerada, exigindo revisao da ROI | 3.696 |
| Predicao vazia, pequena ou ROI vazia | 206 |

Ao cruzar as `951` candidatas confiaveis quanto a ROI com a ROIUNet:

| Consenso v2 | Imagens |
| --- | ---: |
| Produzidas por ambos os modelos | 950 |
| Selecionadas para revisao (`Dice >= 0.75`) | 599 |
| Retidas da fila inicial | 437 |
| Novas candidatas selecionadas | 162 |

Aplicando o filtro automatico estrito na fila v2:

| Dataset expandido v2 preparado | Imagens |
| --- | ---: |
| Pseudo-mascaras elegiveis para treino experimental | 474 |
| Retidas para auditoria prioritaria | 125 |
| Treino total caso v2 seja aprovado | 710 |

Arquivos:

```text
results/intrarenal_model3/medulla_predictions_consensus_v1_dataset_geral/
results/intrarenal_model3/medulla_consensus_review_v2/
results/intrarenal_model3/medulla_consensus_review_v2/audit_packet_v2/
dataset_aumentado/dataset_intrarrenal/pseudo_expandido/medulla_expanded_consensus_v2/
```

## Decisao da etapa

- A expansao v1 foi treinada e validada como experimento de segmentacao.
- A fila v2 amplia a revisao para `599` imagens e encontra `162` candidatas
  novas.
- O dataset v2 esta materializado, mas nao deve iniciar novo retreinamento
  ate que as novas candidatas sejam revisadas, especialmente as `125`
  sinalizadas.
- A etapa seguinte do projeto pode iniciar o estudo de `Cortex` em paralelo,
  mantendo a auditoria de `Medulla` como tarefa de controle de qualidade.

## Aplicacao sobre mascaras renais geradas pelo Modelo 2

O modelo de Medulla ja havia produzido mascaras em `3.696` imagens cuja
mascara renal foi gerada pelo Modelo 2. Este grupo foi processado
separadamente porque um erro na mascara renal contamina o recorte e a
pseudo-mascara intrarrenal.

Ao comparar DeepLab e MedullaROIUNet neste grupo:

| Grupo com ROI renal gerada | Imagens |
| --- | ---: |
| Predicoes de Medulla geradas pelo DeepLab | 3.696 |
| Predicoes candidatas presentes nos dois modelos | 3.630 |
| Consenso para revisao (`Dice >= 0.75`) | 1.494 |
| Passam tambem pelo filtro morfologico estrito | 1.111 |
| Retidas para auditoria prioritaria | 383 |

As `1.111` mascaras filtradas foram materializadas como candidatas de
expansao, com proveniencia
`generated_kidney_mask_requires_review`. Elas nao foram usadas para novo
retreinamento nesta rodada, pois a mascara renal tambem precisa ser auditada.

Arquivos:

```text
results/intrarenal_model3/medulla_consensus_generated_kidney_review_v1/
results/intrarenal_model3/medulla_consensus_generated_kidney_review_v1/audit_packet_v1/
dataset_aumentado/dataset_intrarrenal/pseudo_expandido/medulla_generated_kidney_consensus_v1/
```
