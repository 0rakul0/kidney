# Baseline supervisionado do modelo 3: medula/piramides

## Objetivo

Este experimento estabelece a primeira referencia numerica para a segmentacao
intrarrenal do projeto. O alvo supervisionado e a anotacao `Medulla` do
`kidneyUS`, usada como aproximacao anatomica inicial das piramides renais
visiveis em ultrassom.

## Base preparada

O script `engenharia_dataset/build_intrarenal_kidneyus_dataset.py` converteu
os poligonos dos dois revisores em mascaras binarias e ROIs:

| Medida | Valor |
| --- | ---: |
| Imagens PNG de origem | 534 |
| Imagens com regioes anatomicas suportadas | 487 |
| Imagens sem regiao anatomica suportada | 47 |
| Medula elegivel, anotador 1 | 336 |
| Medula elegivel, anotador 2 | 327 |

As ROIs deste primeiro experimento usam a mascara manual `Capsule`, para medir
a dificuldade da segmentacao interna sem propagar erro do modelo 2.

## Concordancia entre anotadores

| Estrutura | Imagens marcadas por ambos | Dice medio quando ambos marcaram |
| --- | ---: | ---: |
| Capsule | 486 | 0.9461 |
| Cortex | 323 | 0.5961 |
| Medulla | 325 | 0.6936 |
| Central Echo Complex | 468 | 0.8688 |

A concordancia menor em `Medulla` indica que essa tarefa possui variabilidade
de anotacao relevante. Resultados de novos modelos devem ser interpretados em
relacao a esse limite observacional.

## Heuristica atual contra Medulla

O script `src/segmentation/tools/evaluate_pyramid_heuristic.py` comparou a
mascara heuristica existente de piramides candidatas com as mascaras reais de
`Medulla`.

| Referencia | Imagens | Dice medio | IoU medio | Precisao media | Recall medio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anotador 1 | 336 | 0.3291 | 0.2057 | 0.3774 | 0.3365 |
| Anotador 2 | 327 | 0.3621 | 0.2333 | 0.4110 | 0.3774 |

## Interpretacao

A heuristica atual e insuficiente como segmentador das piramides/medula. Ela
pode permanecer como baseline minimo e ferramenta de triagem visual, mas nao
deve fornecer o alvo estrutural principal do modelo 3.

## Segmentador DeepLab inicial

Foi treinado um DeepLabV3-ResNet50 binario sobre ROIs de `Medulla` do
anotador 1, usando `236` imagens para treino, `50` para validacao e `50` para
teste. A funcao de perda utilizada foi `focal_tversky`, com CLAHE e aumento de
dados durante o treinamento.

| Metrica | Validacao | Teste |
| --- | ---: | ---: |
| Dice | 0.7235 | 0.7528 |
| IoU | 0.5668 | 0.6035 |

O checkpoint selecionado foi obtido na epoca `22`, com limiar de decisao
`0.35`. O resultado supera amplamente a heuristica inicial, mas ainda mede o
cenario com ROI manual da capsula.

Tambem foi criada uma arquitetura dedicada, `MedullaROIUNet`, que recebe tres
canais (`ROI em cinza`, `ROI mascarada pelo rim` e `mascara renal`) e restringe
a saida ao interior do rim:

| Arquitetura | Dice validacao | IoU validacao | Dice teste | IoU teste |
| --- | ---: | ---: | ---: | ---: |
| DeepLabV3-ResNet50 | 0.7235 | 0.5668 | 0.7528 | 0.6035 |
| MedullaROIUNet | 0.7295 | 0.5742 | 0.7326 | 0.5780 |

O DeepLab permanece como melhor modelo por Dice no teste. A `MedullaROIUNet`
fica registrada como arquitetura condicionada pela mascara renal para
comparacao e evolucao do pipeline em cascata.

## Geracao de candidatos sobre rins segmentados

O checkpoint de medula foi aplicado nas `4.853` imagens de `dataset_geral`
que possuem caminho de mascara renal. A mascara prevista de medula foi sempre
intersectada com a mascara do rim.

| Status | Imagens |
| --- | ---: |
| Candidata sobre mascara renal existente | 942 |
| Candidata que requer revisao da ROI renal gerada | 3.610 |
| Mascara renal vazia, sem ROI utilizavel | 46 |
| Predicao vazia de medula | 159 |
| Regiao prevista pequena demais | 96 |

Distribuicao das candidatas principais:

| Origem | Candidatas |
| --- | ---: |
| `dataset_train`, `dataset_val`, `dataset_test` | 467 |
| `identificada` | 465 |
| `dataset_loader` | 474 |
| `monai_curated_png` | 3.136 |
| `dataset_augmented_*` | 10 |

As mascaras geradas permanecem como pseudo-rotulos separados das anotacoes
manuais. Uma inspecao do lote MONAI identificou ao menos um caso em que a
pseudo-mascara renal anterior delimitava bexiga, nao rim. Portanto, predições
de medula produzidas sobre mascaras renais geradas nao podem ser aceitas
automaticamente: primeiro e necessario revisar a validade anatomica da ROI
renal. O lote MONAI tambem representa mudanca de dominio em relacao ao
kidneyUS usado no treino de `Medulla`.

A inferencia independente da `MedullaROIUNet` gerou `954` candidatas sobre
mascaras renais existentes e `3.731` candidatas que dependem da revisao da
ROI renal gerada.

## Proximos benchmarks

O proximo benchmark deve comparar:

- uma rede convolucional pequena treinada sobre a ROI;
- os pesos externos `nnUNet Task002_KidneyRegions`;
- uma arquitetura sem pesos, como WiSARD, treinada em pixels ou patches
  binarizados da ROI.

Em uma segunda avaliacao, a ROI manual deve ser substituida pela mascara renal
gerada pelo modelo 2 para mensurar o desempenho real do pipeline em cascata.
