# Como baixar os demais datasets

Este guia define onde salvar cada dataset externo para que ele possa entrar no
mesmo fluxo de engenharia usado no MONAI: baixar bruto, curar para PNG B-mode,
gerar pseudo-mascaras, atualizar `dataset_geral` e manter manifestos.

## Regra de pastas

Dados brutos ficam em:

```text
external_data/raw/<NOME_DO_DATASET>/
```

Dados curados ficam em:

```text
external_data/processed/<NOME_DO_DATASET>/images/
external_data/processed/<NOME_DO_DATASET>/manifest.csv
external_data/processed/<NOME_DO_DATASET>/summary.json
```

Depois de curar qualquer dataset externo, remonte o `dataset_geral`:

```powershell
.\.venv\Scripts\python.exe src\segmentation\build_dataset_geral.py --clear-output --confidence-threshold 0.90
```

## Kaggle

Antes de baixar datasets Kaggle, crie o token da API em sua conta Kaggle e salve:

```text
C:\Users\jeffe\.kaggle\kaggle.json
```

Depois rode:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\download_kaggle_datasets.py
```

Datasets configurados:

| Dataset | Link | Pasta esperada |
| --- | --- | --- |
| Kidney Ultrasound Images Stone/No Stone | <https://www.kaggle.com/datasets/gurjeetkaurmangat/kidney-ultrasound-images-stone-and-no-stone> | `external_data/raw/gurjeetkaurmangat__kidney-ultrasound-images-stone-and-no-stone/` |
| Kidney Failure Ultrasound / renal-data | <https://www.kaggle.com/datasets/zaynebnouiri/renal-data> | `external_data/raw/zaynebnouiri__renal-data/` |
| CT2USforKidneySeg | <https://www.kaggle.com/datasets/siatsyx/ct2usforkidneyseg> | `external_data/raw/siatsyx__ct2usforkidneyseg/` |
| AbdomenUS / ussimandsegm | <https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm> | `external_data/raw/ignaciorlando__ussimandsegm/` |

Se o download for manual pelo navegador, extraia o zip exatamente na pasta
indicada acima.

## Mississippi State abdominal ultrasound

Link:

<https://scholarsjunction.msstate.edu/research-data/5/>

DOI:

<https://doi.org/10.54718/LZXF6315>

O download automatizado retornou `403 Forbidden`, entao provavelmente deve ser
baixado manualmente pelo navegador.

Salve/extrai em:

```text
external_data/raw/MSU_AbdominalUltrasound/
```

Depois rode a curadoria generica:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\curate_external_image_folder.py --input-dir external_data\raw\MSU_AbdominalUltrasound --dataset-name msu_abdominal_ultrasound --clear-output
```

## TRUSTED

Link:

<https://springernature.figshare.com/articles/dataset/TRUSTED_The_Paired_3D_Ultrasound_and_CT_Human_Data_for_Kidney_Segmentation_and_Registration_Research/27981050>

Salve/extrai em:

```text
external_data/raw/TRUSTED/
```

Observacao: e um dataset 3D US + CT de aproximadamente 14,89 GB. Ele nao entra
diretamente no fluxo 2D B-mode atual sem uma etapa propria de conversao de
volumes para fatias/imagens 2D.

## CGPxy Kidney Ultrasound Dataset

Repositorio:

<https://github.com/CGPxy/Ultrasound-Dataset>

Status atual: o repositorio clonado contem README e termo de liberacao, nao as
imagens. Para obter os dados, preencha o `KUS_ReleaseAgreement.pdf` e envie aos
autores conforme instrucoes do repositorio.

Quando receber os arquivos, salve/extrai em:

```text
external_data/raw/CGPxy_KidneyUltrasound/
```

Depois rode:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\curate_external_image_folder.py --input-dir external_data\raw\CGPxy_KidneyUltrasound --dataset-name cgpxy_kidney_ultrasound --clear-output
```

## kidneyUS

Repositorio:

<https://github.com/rsingla92/kidneyUS>

Pagina relacionada:

<https://rsingla.ca/kidneyUS/>

Parte dos arquivos ja existe localmente em:

```text
kidneyUS_images_25_june_2025/
```

Se novos arquivos forem baixados, salve em:

```text
external_data/raw/kidneyUS/
```

Como o `kidneyUS` pode trazer anotacoes multiclasses, ele deve ser tratado com
mais cuidado do que uma pasta generica de imagens. Nao misture mascaras
multiclasses com mascaras binarias sem converter e documentar a classe usada.

## Curadoria generica de imagens

Para qualquer pasta externa que contenha imagens 2D comuns (`png`, `jpg`,
`jpeg`, `bmp`, `tif`, `tiff`), use:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\curate_external_image_folder.py --input-dir external_data\raw\<PASTA> --dataset-name <nome_curto> --clear-output
```

Por padrao, a curadoria:

- aceita imagens em escala de cinza;
- aceita RGB quase cinza, quando os canais sao praticamente iguais;
- rejeita RGB/coloridas, para evitar Doppler/overlays;
- salva tudo como PNG em `external_data/processed/<nome_curto>/images/`;
- gera `manifest.csv` e `summary.json`.

