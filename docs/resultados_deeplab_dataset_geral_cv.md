# Resultados DeepLabV3 no dataset_geral

Este experimento treinou o DeepLabV3 com backbone ResNet50 sobre o
`dataset_geral_cv`, derivado das imagens com mascara aceita em
`dataset_geral`.

## Protocolo

- Imagens com mascara aceita: 3.962.
- Desenvolvimento: 2.774 imagens, correspondentes a 70% da base supervisionada.
- Teste final fixo: 1.188 imagens, correspondentes a 30% da base supervisionada.
- Validacao cruzada: 5 folds dentro dos 70% de desenvolvimento.
- Modelo: DeepLabV3 ResNet50.
- Entrada: 256 x 256 pixels.
- Batch size: 8.
- Funcao de perda: BCE + Dice.
- Aumento de dados: ativo no treino.
- Busca de limiar: ativa entre 0.35 e 0.65.

## Resultado por fold

| Fold | Melhor epoca | Limiar | Dice validacao | IoU validacao | Dice teste | IoU teste |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25 | 0.50 | 0.9298 | 0.8822 | 0.9263 | 0.8782 |
| 2 | 16 | 0.50 | 0.9358 | 0.8793 | 0.9371 | 0.8817 |
| 3 | 19 | 0.55 | 0.9401 | 0.8870 | 0.9391 | 0.8851 |
| 4 | 18 | 0.50 | 0.9467 | 0.8987 | 0.9420 | 0.8904 |
| 5 | 17 | 0.35 | 0.9400 | 0.8868 | 0.9385 | 0.8841 |

## Consolidado

- Dice medio de validacao: 0.9385.
- IoU medio de validacao: 0.8868.
- Dice medio no teste fixo: 0.9366.
- IoU medio no teste fixo: 0.8839.
- Melhor fold por Dice de teste: fold 4, com Dice 0.9420 e IoU 0.8904.

## Artefatos

- Resumo consolidado: `results/segmentation_experiments/dataset_geral_deeplab_resnet50_cv_summary.json`.
- Melhor checkpoint escolhido: `models/dataset_geral_deeplab_resnet50_best.pth`.
- Checkpoint de origem do melhor modelo: `models/dataset_geral_deeplab_resnet50_cv_fold_04.pth`.

## Interpretacao

O resultado supera o melhor benchmark anterior do projeto, que havia obtido
Dice 0.8472 e IoU 0.7732 com DeepLabV3 no `dataset_augmented`. A melhora deve
ser interpretada considerando que o novo treinamento usa uma base maior, com
mascaras manuais e pseudo-mascaras aceitas por criterios de qualidade. Na
redacao da tese, os dados pseudo-rotulados devem ser reportados separadamente
dos dados com anotacao manual.
