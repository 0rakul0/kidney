# Estabilidade da segmentacao de medula

## Objetivo

Esta etapa verifica se o modelo de medula continua estavel quando a ROI renal
manual e substituida pela mascara produzida pelo modelo do rim:

```text
imagem -> modelo_rim -> ROI renal prevista -> modelo_medula -> opacidade
```

O holdout tem `50` imagens de `Medulla` do anotador 1. Em `49` delas tambem
ha mascara de `Medulla` do anotador 2.

## Estabilidade da ROI renal

Na comparacao contra a capsula manual, a mascara produzida pelo modelo do rim
atingiu:

| Metrica | Valor |
| --- | ---: |
| Dice global | 0.9433 |
| IoU global | 0.8927 |
| Dice medio por imagem | 0.9341 |

Assim, neste holdout, a ROI automatica do rim e suficientemente proxima da ROI
manual para avaliar a cascata do modelo de medula.

## Segmentacao em cascata

| Modelo de medula | ROI renal | Dice global vs. anotador 1 | IoU global vs. anotador 1 |
| --- | --- | ---: | ---: |
| DeepLabV3-ResNet50 | Manual | 0.7626 | 0.6163 |
| DeepLabV3-ResNet50 | Modelo do rim | 0.7610 | 0.6142 |
| MedullaROIUNet | Manual | 0.7418 | 0.5896 |
| MedullaROIUNet | Modelo do rim | 0.7300 | 0.5748 |

Para o DeepLab, a troca da ROI manual pela ROI do modelo reduziu o Dice global
em apenas `0.0016`. Ele permanece como modelo principal para segmentacao.

Contra o anotador 2, o DeepLab em cascata atingiu Dice global `0.6996` em `49`
imagens. Este valor deve ser lido junto com a variabilidade entre anotadores,
e nao como falha isolada do modelo.

## Estabilidade da opacidade

As intensidades foram normalizadas para `0-1` e medidas dentro da mascara de
medula. A tabela mostra a concordancia entre a medicao manual do anotador 1 e
a medicao obtida com a predicao em cascata.

| Modelo de medula | Medida | MAE | Diferenca media | Correlacao |
| --- | --- | ---: | ---: | ---: |
| DeepLabV3-ResNet50 | Media na medula | 0.0149 | +0.0087 | 0.9508 |
| DeepLabV3-ResNet50 | Razao medula/rim | 0.0645 | +0.0369 | 0.8271 |
| MedullaROIUNet | Media na medula | 0.0122 | -0.0017 | 0.9444 |
| MedullaROIUNet | Razao medula/rim | 0.0529 | -0.0056 | 0.8161 |

O DeepLab e superior para segmentacao e mantem correlacao alta na intensidade
media. A ROIUNet apresenta menor erro e menor vies na opacidade media, portanto
e util como modelo de controle durante a validacao da medida.

## Consenso para pseudo-mascaras

Para reduzir ruido antes de revisar pseudo-mascaras, foram cruzadas apenas
predicoes obtidas sobre mascaras renais existentes. Uma candidata entra na fila
prioritaria quando ambos os modelos geram uma predicao valida e o Dice entre
eles e pelo menos `0.75`.

| Etapa | Imagens |
| --- | ---: |
| Candidatas presentes nos dois modelos | 941 |
| Selecionadas para revisao (`Dice >= 0.75`) | 457 |
| Consenso forte (`Dice >= 0.80`) | 287 |
| Consenso muito forte (`Dice >= 0.85`) | 130 |

O Dice medio entre modelos nessas `941` candidatas foi `0.7085`, com mediana
`0.7468`.

Arquivos gerados:

```text
results/intrarenal_model3/medulla_consensus_review/summary.json
results/intrarenal_model3/medulla_consensus_review/selected_for_review.csv
results/intrarenal_model3/medulla_consensus_review/previews/selected/
```

## Decisao atual

- Usar o `DeepLabV3-ResNet50` como segmentador principal da medula.
- Manter a `MedullaROIUNet` como controle para estabilidade das medidas de
  opacidade e para selecao por consenso.
- Revisar primeiro as `457` pseudo-mascaras com alto consenso e ROI renal
  existente.
- Nao usar automaticamente mascaras produzidas sobre ROIs renais geradas,
  pois ja foram observadas ROIs anatomicamente incorretas em imagens externas.

Esta avaliacao mede robustez interna da cascata. Ela nao e validacao externa
completa, pois o modelo do rim foi treinado em uma base mais ampla que pode
conter imagens provenientes da mesma fonte local.

Por fim, a classe manual `Medulla` deve ser interpretada conforme a anotacao
disponivel. Afirmar que ela identifica especificamente piramides renais exige
revisao clinica das mascaras selecionadas.
