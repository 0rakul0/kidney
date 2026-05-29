# Protocolo revisado: kidneyUS como fonte unica e WiSARD na etapa 2

## Decisao metodologica

O novo ciclo de modelos deve usar o `kidneyUS_images_25_june_2025/` como fonte
canonica das imagens. O `flood_1` sai do fluxo de treinamento porque suas 513
imagens correspondem a um subconjunto preprocessado das mesmas imagens do
kidneyUS, em 256x256, enquanto o kidneyUS preserva resolucoes originais e traz
anotacoes poligonais multiclasse.

O `flood_1` fica apenas como historico do primeiro protocolo e nao deve ser
contado como base independente.

## Papel das anotacoes

O kidneyUS possui dois arquivos de anotacao:

| Arquivo | Uso proposto |
| --- | --- |
| `reviewed_labels_1.csv` | Anotador principal para treino e validacao interna inicial. |
| `reviewed_labels_2.csv` | Anotador secundario para estabilidade, concordancia e consenso. |

As duas anotacoes descrevem as mesmas classes anatomicas:

| Classe | Papel no novo pipeline |
| --- | --- |
| `Capsule` | Mascara principal do rim; substitui o contorno renal vindo do `flood_1`. |
| `Cortex` | Classe intrarrenal para segmentacao/análise cortical. |
| `Medulla` | Classe intrarrenal associada a regiao medular/piramides. |
| `Central Echo Complex` | Classe intrarrenal do complexo ecogenico central. |

## Regra de treinamento

1. Gerar mascaras a partir do `reviewed_labels_1.csv`.
2. Usar `Capsule` como mascara renal externa.
3. Recortar a ROI renal pela `Capsule`.
4. Treinar a segmentacao intrarrenal com `Cortex`, `Medulla` e
   `Central Echo Complex`.
5. Usar o `reviewed_labels_2.csv` para medir estabilidade por classe.
6. Criar consenso quando houver boa sobreposicao entre anotadores.

Regra inicial de consenso:

- `Capsule`: pode usar consenso ou anotador 1, pois a concordancia e alta.
- `Central Echo Complex`: bom candidato a consenso.
- `Medulla`: usar consenso apenas quando a sobreposicao for suficiente.
- `Cortex`: tratar com cautela, pois a variabilidade entre anotadores e maior.

## Papel do WiSARD/WNN

O WiSARD entra na segunda etapa, depois que o rim ja foi isolado pela mascara
renal. Ele nao deve competir como segmentador de imagem inteira.

Entrada proposta:

```text
Imagem original
-> mascara Capsule
-> ROI renal recortada/mascarada
-> codificacao binaria por pixels, patches ou descritores
-> WiSARD/WNN
```

Tarefas comparativas possiveis:

1. Classificacao de pixels ou patches dentro da ROI em `Cortex`, `Medulla`,
   `Central Echo Complex` e fundo intrarrenal.
2. Classificacao de patches como pertencentes ou nao a uma estrutura interna.
3. Classificacao exploratoria da ROI renal por qualidade/anatomia avaliavel.

Para uma comparacao justa, o WiSARD deve receber apenas informacao intrarrenal,
assim como os modelos da etapa 2. As metricas devem ser calculadas contra as
mascaras do anotador 1 e, separadamente, contra o anotador 2/consenso.

## Comparativos esperados

Modelos a comparar no novo ciclo:

- DeepLabV3 multiclasse na ROI renal.
- U-Net/UNet++ multiclasse na ROI renal, se mantidos no benchmark.
- SegFormer na ROI renal, se houver custo computacional aceitavel.
- WiSARD/WNN como baseline sem pesos para pixels, patches ou descritores.
- Heuristicas simples apenas como baseline minimo.

## Saidas esperadas

O novo dataset supervisionado deve gerar:

```text
dataset_aumentado/dataset_intrarrenal/
  intermediario/kidneyus_regions/
  supervisionado/capsule_annotator_1/
  supervisionado/regions_multiclass_annotator_1/
  validacao/regions_multiclass_annotator_2/
  consenso/regions_multiclass_consensus/
```

O nome exato pode mudar, mas a separacao conceitual deve permanecer:

- treino principal;
- validacao por segundo anotador;
- consenso quando aplicavel.

