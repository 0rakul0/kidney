# Modelos e resultados

[← Wiki](README.md)

## Cápsula renal

No teste deduplicado de 70 imagens do kidneyUS:

| Modelo | Dice | IoU | Precisão | Recall | FPS médio |
|---|---:|---:|---:|---:|---:|
| U-Net | **0,9290** | **0,8722** | **0,9236** | **0,9440** | 18,11 |
| SegFormer-B0 | 0,9223 | 0,8598 | 0,9117 | 0,9426 | **31,34** |
| DeepLabV3 R50 | 0,9203 | 0,8606 | 0,9219 | 0,9338 | 22,26 |
| UNet++ | 0,9099 | 0,8419 | 0,8910 | 0,9419 | 16,57 |

A U-Net foi escolhida pela melhor qualidade agregada.

## Consenso U-Net × DeepLabV3

Calibração out-of-fold em cinco folds:

- consenso médio: `0,9416`;
- mediana: `0,9627`;
- correlação entre consenso e menor Dice frente à referência: `0,840`.

O consenso serve para triagem e prioridade de revisão.

## Segmentação intrarrenal

Teste manual com 50 imagens:

| Modelo | Cortex | Medulla | CEC | Dice médio | IoU médio |
|---|---:|---:|---:|---:|---:|
| U-Net | **0,7075** | **0,7237** | 0,8469 | **0,7594** | **0,6163** |
| DeepLabV3 R50 | 0,6822 | 0,7156 | **0,8497** | 0,7492 | 0,6045 |

## Expansão experimental de Medulla

| Configuração | Dice teste |
|---|---:|
| Baseline manual | 0,7528 |
| Expansão consenso v1 | 0,7523 |

A expansão aumentou cobertura, mas não demonstrou ganho no Dice de teste.

## Resultado histórico no dataset_geral

`docs/resultados_deeplab_dataset_geral_cv.md` registra Dice médio de teste `0,9366` em uma base expandida. Esse número não é diretamente comparável ao benchmark de 0,9290 do kidneyUS porque dataset, composição das máscaras e protocolo diferem.

## Reprodutibilidade

Ao citar um resultado, registre dataset, split, checkpoint, limiar, pré-processamento e commit.
