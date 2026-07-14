# Segmentacao

Esta pasta concentra os novos codigos de segmentacao do projeto.

A construcao do `dataset_geral` faz parte da engenharia de dataset documentada
em:

```text
docs/narrativa_engenharia_dataset.md
```

## `dataset_geral_v2`

O script `build_dataset_geral.py` monta a base auditada com o formato:

```text
dataset_aumentado/dataset_geral_v2/
  imagens/
  mascaras/
  manifest.csv
  summary.json
  relatorios/
    duplicadas_por_hash.csv
    faltando_mascara.csv
    pseudomascaras_nao_validadas.csv
    predicoes_vazias.csv
    fila_revisao.csv
```

Ele procura imagens em:

- `dataset_aumentado/dataset_intrarrenal/supervisionado/capsule_annotator_1_deduplicated/`
- `dataset_aumentado/fontes/external_data/processed/*/images/`

Quando a mascara ja existe, ela e copiada para
`dataset_aumentado/dataset_geral_v2/mascaras/`.
Quando a mascara nao existe, o script usa o segmentador configurado para gerar
uma pseudomascara candidata. Toda predicao nao vazia e preservada na resolucao
original, mas permanece com estado `pending_human_review`.

Comando principal:

```powershell
.\.venv\Scripts\python.exe src\segmentation\build_dataset_geral.py --clear-output
```

Por padrao, o script usa:

- modelo: `unet`;
- checkpoint: `models/kidneyus_capsule_dedup_unet.pth`;
- CLAHE, como no treinamento;
- limiar de segmentacao `0.35`, selecionado na validacao;
- media das predicoes original e espelhada horizontalmente.

A base supervisionada canonica para segmentacao externa do rim agora e criada
com `engenharia_dataset/create_capsule_splits.py`, usando a classe `Capsule` do
kidneyUS. O `dataset_inicial/flood_1` foi retirado do fluxo de treinamento.

A quantidade absoluta de pixels e a area relativa nao sao criterios de
aceitacao, pois dependem da resolucao e do enquadramento. Confianca media e
concordancia entre a predicao original e a espelhada sao registradas somente
para ordenar a fila de revisao. Os valores de referencia correspondem ao
percentil 5 observado na validacao manual e nao comprovam que o rim foi
localizado corretamente.

## Split 70/30 e Validacao Cruzada

O novo modelo de segmentacao deve ser treinado a partir das imagens com mascara
aceita em `dataset_geral`. Para isso, o projeto usa
`engenharia_dataset/create_dataset_geral_splits.py`, que cria
`dataset_geral_cv/` com 30% das imagens em um holdout final fixo e os 70%
restantes divididos em 5 folds de validacao cruzada.

Cada fold segue o formato esperado pelos scripts de treino:

```text
dataset_geral_cv/folds/fold_01/
  train/image/
  train/mask/
  val/image/
  val/mask/
  test/image/
  test/mask/
```

Comando para criar os splits:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\create_dataset_geral_splits.py --clear-output --link-mode hardlink --folds 5 --test-ratio 0.30 --seed 42
```

Comando para treinar DeepLabV3 nos folds:

```powershell
.\.venv\Scripts\python.exe src\segmentation\experiments\run_dataset_geral_cv.py --epochs 30 --batch-size 8 --backbone resnet50
```

## Estado Atual

Ultima montagem executada:

```powershell
.\.venv\Scripts\python.exe src\segmentation\calibrate_capsule_review.py
.\.venv\Scripts\python.exe src\segmentation\build_dataset_geral.py --clear-output
```

Resultado:

- 4.947 imagens unicas: 468 `kidneyUS` e 4.479 MONAI;
- 468 mascaras manuais;
- 3.534 pseudomascaras externas nao validadas;
- 945 imagens externas sem predicao;
- fila das pseudomascaras: 1.024 prioridade alta, 189 media e 2.321 rotina.

Nenhuma pseudomascara MONAI e tratada automaticamente como referencia correta.

## Consenso U-Net e DeepLabV3

O script `run_capsule_model_consensus.py` usa a U-Net selecionada no teste
deduplicado e a DeepLabV3-ResNet50 como modelo independente de comparacao:

```powershell
.\.venv\Scripts\python.exe src\segmentation\run_capsule_model_consensus.py
```

Os limiares de consenso foram calibrados por validacao cruzada em 5 folds,
agrupada por exame. Cada uma das 468 imagens manuais foi avaliada uma vez por
modelos que nao a utilizaram no treinamento. O Dice entre os modelos apresentou
correlacao de `0.840` com o menor Dice em relacao a referencia. Os valores
`0.89` e `0.94` correspondem aos percentis 10 e 25 arredondados.

Resultado externo:

- 1.529 pseudomascaras U-Net com consenso alto;
- 699 com consenso intermediario;
- 1.306 com baixo consenso ou sem confirmacao da DeepLabV3;
- 435 candidatos produzidos apenas pela DeepLabV3;
- 510 imagens sem predicao por nenhum dos dois modelos.

O consenso e um indicador para priorizar revisao, nao uma validacao automatica.

