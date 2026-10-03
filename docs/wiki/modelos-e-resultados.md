# Modelos e resultados

[← Wiki](README.md)

## Objetivo desta página

Esta página consolida os principais experimentos já realizados e, principalmente, explica **qual resultado pertence a qual protocolo**.

Isso evita comparar números obtidos em datasets diferentes como se fossem parte do mesmo benchmark.

## Tarefa 1 — segmentação da cápsula renal

O primeiro problema é segmentar o contorno externo do rim.

### Arquiteturas comparadas

- U-Net;
- UNet++;
- DeepLabV3 com ResNet50;
- SegFormer-B0.

### Resultado no teste deduplicado do kidneyUS

| Modelo | Dice | IoU | Precisão | Recall | FPS médio |
|---|---:|---:|---:|---:|---:|
| U-Net | **0,9290** | **0,8722** | **0,9236** | **0,9440** | 18,11 |
| SegFormer-B0 | 0,9223 | 0,8598 | 0,9117 | 0,9426 | **31,34** |
| DeepLabV3 R50 | 0,9203 | 0,8606 | 0,9219 | 0,9338 | 22,26 |
| UNet++ | 0,9099 | 0,8419 | 0,8910 | 0,9419 | 16,57 |

### Interpretação

A U-Net apresentou o melhor resultado agregado e foi escolhida como modelo principal para delimitar a ROI renal.

O SegFormer foi mais rápido, mas a escolha da cascata priorizou qualidade de segmentação.

## Tarefa 2 — consenso para priorização

A U-Net principal foi combinada com uma DeepLabV3 independente.

O objetivo não é fazer ensemble final, mas estimar estabilidade.

### Calibração out-of-fold

O consenso foi calibrado em cinco folds, garantindo que cada imagem manual fosse avaliada por modelos que não a utilizaram no treino daquele fold.

Resultados:

- consenso médio: `0,9416`;
- mediana: `0,9627`;
- correlação com o menor Dice frente à referência: `0,840`.

### Uso operacional

```text
consenso alto → revisão mais simples/provável estabilidade
consenso baixo → prioridade para inspeção
```

Esse indicador não substitui a referência manual.

## Tarefa 3 — segmentação intrarrenal

Depois da cápsula, a tarefa passa a ser multiclasse:

- córtex;
- medula;
- CEC.

### Modelos comparados

- U-Net multiclasse;
- DeepLabV3-ResNet50 multiclasse.

### Resultado em 50 imagens de teste

| Modelo | Cortex | Medulla | CEC | Dice médio | IoU médio |
|---|---:|---:|---:|---:|---:|
| U-Net | **0,7075** | **0,7237** | 0,8469 | **0,7594** | **0,6163** |
| DeepLabV3 R50 | 0,6822 | 0,7156 | **0,8497** | 0,7492 | 0,6045 |

### Interpretação por estrutura

**Córtex:** é uma das classes mais difíceis, por apresentar limites menos marcados.

**Medula:** exige localizar estruturas internas pequenas e heterogêneas.

**CEC:** apresenta maior contraste e, por isso, alcançou os maiores valores de Dice.

A U-Net foi mantida como melhor modelo agregado, apesar de a DeepLab apresentar pequena vantagem no CEC.

## Experimento de expansão de Medulla

O projeto testou pseudo-labeling de medula com consenso entre modelos.

### Treino

| Configuração | Imagens de treino | Dice teste |
|---|---:|---:|
| Manual | 236 | 0,7528 |
| Expandido v1 | 578 | 0,7523 |

### Leitura correta

A expansão:

- aumentou o volume de treino;
- ampliou cobertura de casos;
- não degradou significativamente o teste;
- não mostrou melhora clara no Dice de teste.

Por isso, o valor científico principal desse experimento está na engenharia de dados e na análise de pseudo-rótulos, não em declarar superioridade do modelo expandido.

## Resultado histórico no dataset_geral

O documento:

```text
docs/resultados_deeplab_dataset_geral_cv.md
```

registra Dice médio de teste de `0,9366`.

Esse número pertence a uma base expandida que misturava referências de origens diferentes e pseudo-máscaras selecionadas por critérios operacionais.

### Portanto

Não comparar diretamente:

```text
0,9290 — kidneyUS manual deduplicado
vs.
0,9366 — dataset_geral expandido
```

São protocolos diferentes.

## Checkpoints

A pasta `models/` contém metadados de várias famílias:

- U-Net;
- UNet++;
- DeepLabV3;
- SegFormer;
- folds OOF;
- modelos de córtex;
- modelos de medula;
- modelos intrarrenais multiclasse.

Os arquivos `.meta.json` ajudam a registrar parâmetros do experimento.

## Como citar um resultado

Sempre informar:

- tarefa;
- dataset;
- número de imagens;
- estratégia de split;
- arquitetura;
- checkpoint;
- pré-processamento;
- limiar;
- métrica;
- status de curadoria dos rótulos.

## Modelo “campeão” depende da tarefa

Não existe necessariamente um único melhor modelo para todo o projeto.

| Tarefa | Referência atual |
|---|---|
| Cápsula | U-Net |
| Intrarrenal agregado | U-Net |
| CEC isolado | DeepLabV3 ligeiramente superior no teste citado |
| Triagem de cápsula | Consenso U-Net × DeepLabV3 |

## Próxima leitura

Para entender o que o projeto faz **depois da segmentação**, siga para [Ecogenicidade renal quantitativa](ecogenicidade.md).
