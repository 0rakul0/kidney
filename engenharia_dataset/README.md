# Engenharia de Dataset

Esta pasta agrupa os scripts de engenharia, tratamento, manipulacao e extracao
de dados usados no projeto. A ideia e separar a construcao da base dos scripts de
treino, avaliacao e escrita do artigo.

Os dados foram reorganizados em `dataset_inicial/` e `dataset_aumentado/`.
Fontes externas e MONAI agora ficam sob
`dataset_aumentado/fontes/external_data/`, e a base consolidada fica em
`dataset_aumentado/dataset_geral/`. Veja
`docs/organizacao_datasets_curadoria.md`.

A narrativa completa da busca de datasets externos, da estrategia incremental
para MONAI/NVIDIA e da montagem do `dataset_geral` esta em:

```text
docs/narrativa_engenharia_dataset.md
```

O guia de download manual/automatico para os demais datasets esta em:

```text
docs/download_manual_datasets.md
```

## Fluxo MONAI/NVIDIA

Fluxo adotado para fontes externas sem mascaras manuais:

```text
zip bruto MONAI
-> extrair DICOMs
-> converter so frames uteis para PNG
-> filtrar imagens renais B-mode
-> gerar pseudo-mascaras
-> manter PNG + mascara + manifesto
-> apagar DICOM/zip bruto se tudo estiver validado
```

O MONAI foi curado em lotes neste formato:

- total processado: 238 estudos renais/retroperitoneais;
- entrada bruta temporaria acumulada: aproximadamente 68,82 GB;
- saida curada: 4.487 PNGs em B-mode/escala de cinza;
- tamanho processado: aproximadamente 676 MB;
- rejeicoes: 5.716 entradas RGB/coloridas, preservadas no manifesto como rejeitadas;
- zips brutos removidos apos validacao de cada conversao.

Como o MONAI/NVIDIA e amplo e pesado, a regra adotada e baixar em partes:

1. baixar metadados;
2. filtrar estudos candidatos;
3. estimar tamanho;
4. baixar lote pequeno;
5. converter para PNG leve;
6. validar manifestos;
7. apagar os zips brutos.

Arquivos gerados:

```text
external_data/processed/monai_renal_png/images/
external_data/processed/monai_renal_png/metadata/
external_data/processed/monai_renal_png/manifest.csv
external_data/processed/monai_renal_png/summary.json
```

## Scripts

| Script | Funcao |
| --- | --- |
| `download_monai_renal_subset.py` | Baixa metadados MONAI/NVIDIA, identifica estudos renais/retroperitoneais, estima tamanho dos zips e baixa um subconjunto controlado. |
| `curate_monai_renal_dicoms.py` | Converte DICOMs MONAI para PNG leve, reduz cines, pula RGB/coloridos por padrao e gera manifesto. |
| `download_kaggle_datasets.py` | Baixa datasets Kaggle definidos em `config/kaggle_datasets.csv`, quando o token Kaggle estiver configurado. |
| `curate_external_image_folder.py` | Curadoria generica de pastas externas de imagens 2D para PNG B-mode/escala de cinza. |
| `create_dataset_geral_splits.py` | Cria a divisao 70/30 do `dataset_geral` e os folds de validacao cruzada dentro dos 70% de treino/desenvolvimento. |
| `expand_dataset_from_loader.py` | Gera pseudo-mascaras e monta `dataset_augmented/` a partir de uma pasta de imagens. |
| `divisor_segmentation.py` | Fluxo legado de pseudo-rotulacao/separacao de imagens identificadas e nao identificadas. |
| `annotate_reference_roi.py` | Interface simples para anotar ROI de referencia externa ao rim. |
| `extract_renal_features.py` | Extrai atributos quantitativos do rim, regiao interna, cortex, referencia e candidatos a piramides. |
| `prepare_renal_labels_template.py` | Prepara template CSV para rotulos da etapa de classificacao renal. |
| `suggest_renal_labels.py` | Sugere rotulos heuristicos iniciais a partir das features extraidas. |
| `build_intrarenal_kidneyus_dataset.py` | Converte os poligonos multiclasse do kidneyUS em mascaras e ROIs supervisionadas para o modelo 3. |
| `create_medulla_splits.py` | Cria `train`, `val` e `test` com ROIs e mascaras de `Medulla` para treinar o segmentador do modelo 3. |
| `create_intrarenal_multiclass_splits.py` | Cria a etapa 2 multiclasse dentro da ROI renal: fundo, `Cortex`, `Medulla` e `Central Echo Complex`. |
| `build_medulla_consensus_expanded_dataset.py` | Materializa expansao pseudo-rotulada de `Medulla`, mantendo validacao/teste manuais e gerando folhas de auditoria. |
| `../src/segmentation/tools/evaluate_medulla_stability.py` | Mede a estabilidade da cascata rim-medula e das medidas iniciais de opacidade. |
| `../src/segmentation/tools/select_medulla_consensus_candidates.py` | Prioriza pseudo-mascaras de medula para revisao por consenso entre modelos. |

## Comandos principais

Baixar/atualizar metadados MONAI e preparar um subconjunto de ate 2 GB:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\download_monai_renal_subset.py --max-gb 2
```

Converter zips MONAI para PNGs curados:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\curate_monai_renal_dicoms.py --clear-output
```

Baixar datasets Kaggle configurados:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\download_kaggle_datasets.py
```

Criar splits 70/30 com validacao cruzada em 5 folds:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\create_dataset_geral_splits.py --clear-output --link-mode hardlink --folds 5 --test-ratio 0.30 --seed 42
```

Gerar pseudo-mascaras para uma pasta de imagens e montar dataset aumentado:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\expand_dataset_from_loader.py full --clear-output
```

Extrair features renais:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\extract_renal_features.py
```

Construir a base supervisionada de medula/piramides do modelo 3:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\build_intrarenal_kidneyus_dataset.py --clear-output
```

Essa base rasteriza as anotacoes `Medulla`, `Capsule`, `Cortex` e
`Central Echo Complex` dos dois revisores do kidneyUS. Os recortes iniciais
usam a capsula manual para isolar a dificuldade da segmentacao interna; a
avaliacao em cascata devera repetir o experimento usando a mascara produzida
pelo modelo 2 como ROI. Para imagens que possuem `Medulla`, tambem sao salvas
as imagens isoladas em
`dataset_intrarrenal/intermediario/kidneyus_regions/roi/<anotador>/medulla_image/`.

Avaliar a heuristica atual de piramides contra o alvo supervisionado `Medulla`:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\evaluate_pyramid_heuristic.py
```

Criar os splits de treino da medula com o anotador 1:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\create_medulla_splits.py --clear-output
```

Treinar um primeiro DeepLab binario sobre esses recortes:

```powershell
.\.venv\Scripts\python.exe src\segmentation\experiments\train_deeplab.py `
  --dataset-path dataset_aumentado\dataset_intrarrenal\supervisionado\medulla_annotator_1 `
  --experiment-name medulla_deeplab_resnet50_annotator1_baseline `
  --checkpoint-name medulla_deeplab_resnet50_annotator1_baseline.pth `
  --epochs 30 --batch-size 8 --augment --clahe `
  --loss focal_tversky --early-stopping 8
```

Gerar pseudo-mascaras candidatas de medula dentro dos rins ja segmentados:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\generate_medulla_masks_from_kidney_roi.py
```

As predicoes sao restritas pela mascara renal e salvas em `results/` como
candidatas para revisao; elas nao substituem automaticamente anotacoes manuais.

Treinar a arquitetura dedicada condicionada pela mascara renal:

```powershell
.\.venv\Scripts\python.exe src\segmentation\experiments\train_medulla_roi_unet.py
```

Aplicar a arquitetura dedicada em uma saida separada:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\generate_medulla_masks_from_kidney_roi.py `
  --architecture roi_unet `
  --output-root results\intrarenal_model3\medulla_roi_unet_predictions_dataset_geral
```

Avaliar se a segmentacao e as medidas de opacidade permanecem estaveis quando
a ROI manual e substituida pela mascara produzida pelo modelo do rim:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\evaluate_medulla_stability.py --architecture deeplab
.\.venv\Scripts\python.exe src\segmentation\tools\evaluate_medulla_stability.py --architecture roi_unet
```

Selecionar a fila inicial de revisao por concordancia entre os dois modelos de
medula, somente sobre mascaras renais existentes:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\select_medulla_consensus_candidates.py
```

O limiar padrao e `Dice >= 0.75` entre os modelos. A saida e salva em
`results/intrarenal_model3/medulla_consensus_review/`; o resultado continua
sendo pseudo-rotulo e requer revisao visual antes de qualquer retreinamento.

Criar a expansao conservadora de treino e o pacote visual de auditoria:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\build_medulla_consensus_expanded_dataset.py --clear-output
```

O script mantem `val` e `test` manuais sem alteracao. Por padrao, somente
pseudo-mascaras com `Dice >= 0.78`, area relativa minima, baixa fragmentacao e
componente dominante entram no treino experimental; as demais permanecem na
fila de auditoria.

Treinar o modelo expandido v1:

```powershell
.\.venv\Scripts\python.exe src\segmentation\experiments\train_deeplab.py `
  --dataset-path dataset_aumentado\dataset_intrarrenal\pseudo_expandido\medulla_expanded_consensus_v1 `
  --experiment-name medulla_deeplab_resnet50_consensus_v1 `
  --checkpoint-name medulla_deeplab_resnet50_consensus_v1.pth `
  --epochs 30 --batch-size 8 --augment --clahe `
  --loss focal_tversky --early-stopping 8 --num-workers 0
```

Gerar a nova rodada de candidatas e preparar a fila v2:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\generate_medulla_masks_from_kidney_roi.py `
  --architecture deeplab `
  --checkpoint models\medulla_deeplab_resnet50_consensus_v1.pth `
  --output-root results\intrarenal_model3\medulla_predictions_consensus_v1_dataset_geral

.\.venv\Scripts\python.exe src\segmentation\tools\select_medulla_consensus_candidates.py `
  --deeplab-root results\intrarenal_model3\medulla_predictions_consensus_v1_dataset_geral `
  --output-root results\intrarenal_model3\medulla_consensus_review_v2

.\.venv\Scripts\python.exe engenharia_dataset\build_medulla_consensus_expanded_dataset.py `
  --selected-manifest results\intrarenal_model3\medulla_consensus_review_v2\selected_for_review.csv `
  --output-root dataset_aumentado\dataset_intrarrenal\pseudo_expandido\medulla_expanded_consensus_v2 `
  --review-root results\intrarenal_model3\medulla_consensus_review_v2\audit_packet_v2 `
  --clear-output
```

## Regras de proveniencia

Todo dado externo usado em experimento deve manter:

- fonte e URL;
- licenca;
- data de download;
- caminho local;
- rotulo original, quando existir;
- indicacao se o rotulo e manual, clinico, heuristico ou pseudo-rotulo;
- finalidade: segmentacao, pseudo-labeling, pre-treino, classificacao ou validacao qualitativa.

Dados externos com pseudo-mascaras devem ser reportados separadamente dos dados
com mascaras manuais.

### Etapa 2 multiclasse com DeepLabV3

O caminho recomendado para aproximar o fluxo do `kidneyUS` e reduzir modelos
separados e:

1. segmentar `Rim/Capsule` com o DeepLab binario ja existente;
2. usar a ROI renal para segmentar, em uma unica inferencia multiclasse,
   `Cortex`, `Medulla` e `Central Echo Complex`.

Preparar o dataset supervisionado, com separacao por paciente:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\create_intrarenal_multiclass_splits.py --clear-output
```

Treinar o DeepLabV3 multiclasse:

```powershell
.\.venv\Scripts\python.exe src\segmentation\experiments\train_deeplab_intrarenal_multiclass.py --epochs 50 --batch-size 4
```

Aplicar o modelo no `dataset_geral`:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\generate_intrarenal_multiclass_masks_from_kidney_roi.py
```

Resultado inicial do checkpoint
`models/intrarenal_deeplab_resnet50_multiclass_annotator1.pth`, treinado com
`235/50/50` imagens e sem vazamento de pacientes entre os splits:

| Classe | Dice teste | IoU teste |
|---|---:|---:|
| `Cortex` | 0.682169 | 0.517645 |
| `Medulla` | 0.715567 | 0.557108 |
| `Central Echo Complex` | 0.849745 | 0.738745 |
| Media das classes internas | 0.749160 | 0.604499 |
