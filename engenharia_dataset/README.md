# Engenharia de Dataset

Esta pasta agrupa os scripts de engenharia, tratamento, manipulacao e extracao
de dados usados no projeto. A ideia e separar a construcao da base dos scripts de
treino, avaliacao e escrita do artigo.

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

