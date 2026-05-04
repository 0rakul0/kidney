# Segmentacao

Esta pasta concentra os novos codigos de segmentacao do projeto.

A construcao do `dataset_geral` faz parte da engenharia de dataset documentada
em:

```text
docs/narrativa_engenharia_dataset.md
```

## `dataset_geral`

O script `build_dataset_geral.py` monta a base geral com o formato:

```text
dataset_geral/
  imagens/
  mascaras/
  manifest.csv
  summary.json
  relatorios/
    duplicadas_por_hash.csv
    faltando_mascara.csv
    mascaras_geradas.csv
```

Ele procura imagens em:

- `dataset/`
- `dataset_augmented/`
- `identificada/image/`
- `pseudo_labels/accepted/image/`
- `dataset_loader/`
- `kidneyUS_images_25_june_2025/`
- `external_data/processed/*/images/`

Quando a mascara ja existe, ela e copiada para `dataset_geral/mascaras/`.
Quando a mascara nao existe, o script usa o segmentador configurado para gerar
uma pseudo-mascara e so aceita se passar pelos criterios de qualidade.

Comando principal:

```powershell
.\.venv\Scripts\python.exe src\segmentation\build_dataset_geral.py --clear-output
```

Por padrao, o script usa:

- modelo: `deeplab`;
- checkpoint: `models/augmented_deeplab_resnet50_baseline.pth`;
- limiar minimo de confianca: `0.90`.

Essa confianca e operacional: significa que a pseudo-mascara precisa passar por
filtros de confianca media do modelo, area plausivel, quantidade minima de
pixels e numero maximo de componentes. Ela nao substitui revisao humana.

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
.\.venv\Scripts\python.exe src\segmentation\build_dataset_geral.py --clear-output --confidence-threshold 0.90
```

Resultado:

- imagens unicas: 5.994;
- mascaras existentes copiadas: 1.001;
- pseudo-mascaras geradas e aceitas: 2.961;
- imagens sem mascara aceita: 2.032;
- total com mascara aceita: 3.962.

Pendencias:

- `dataset_loader`: 53 imagens rejeitadas;
- `monai_curated_png`: 1.979 imagens rejeitadas.

Principais motivos de rejeicao:

- baixa confianca com area renal pequena;
- poucos pixels de primeiro plano;
- area relativa abaixo do limite minimo;
- excesso de componentes em casos isolados.

