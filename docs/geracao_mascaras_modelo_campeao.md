# Geracao de mascaras faltantes com o modelo campeao

Depois da validacao cruzada no `dataset_geral`, o modelo selecionado como campeao foi o
DeepLabV3 com backbone ResNet-50, correspondente ao fold 4. O checkpoint final usado nesta
etapa foi:

`D:\kidney\models\dataset_geral_deeplab_resnet50_best.pth`

## Objetivo

O objetivo desta etapa foi usar o modelo campeao para tentar segmentar as imagens do
`dataset_geral` que ainda estavam sem mascara. As mascaras ja existentes nao foram
sobrescritas. Apenas imagens marcadas no manifesto como `has_mask=false`, ou com caminho de
mascara ausente, foram processadas.

## Script

O script criado para esta engenharia foi:

`D:\kidney\src\segmentation\tools\generate_missing_dataset_geral_masks.py`

Comando executado:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\generate_missing_dataset_geral_masks.py
```

O script usa o limiar salvo no checkpoint campeao (`best_threshold=0.5`) e aplica um filtro
adicional de qualidade com confianca minima de `0.90`. A mascara so e salva quando passa pelos
criterios de area minima, area maxima, numero de componentes conectados e quantidade minima de
pixels de primeiro plano.

## Resultado

Antes desta etapa, o `dataset_geral` possuia 5994 imagens unicas, sendo 3962 com mascara e
2032 sem mascara.

O modelo campeao processou as 2032 imagens sem mascara:

- 891 pseudo-mascaras foram aceitas e salvas.
- 1141 imagens continuaram sem mascara por nao passarem no controle de qualidade.

Depois da etapa, o `dataset_geral` passou a ter:

- 5994 imagens unicas.
- 4853 imagens com mascara.
- 1141 imagens sem mascara.
- 1001 mascaras existentes ou primarias.
- 3852 mascaras geradas aceitas, incluindo as novas do modelo campeao.

## Arquivos de rastreabilidade

O manifesto principal foi atualizado em:

`D:\kidney\dataset_aumentado\dataset_geral\manifest.csv`

O resumo consolidado foi atualizado em:

`D:\kidney\dataset_aumentado\dataset_geral\summary.json`

O relatorio especifico da rodada com o modelo campeao foi salvo em:

`D:\kidney\dataset_aumentado\dataset_geral\relatorios\mascaras_geradas_modelo_campeao.csv`

As imagens que ainda permanecem sem mascara foram registradas em:

`D:\kidney\dataset_aumentado\dataset_geral\relatorios\faltando_mascara.csv`

## Atualizacao dos splits

Como a quantidade de imagens com mascara aumentou, a divisao 70/30 e os folds de validacao
cruzada foram recriados com hardlinks para reduzir uso de espaco:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\create_dataset_geral_splits.py --clear-output --link-mode hardlink
```

Nova divisao:

- 4853 imagens elegiveis com mascara.
- 3398 imagens no conjunto de desenvolvimento de 70%.
- 1455 imagens no teste fixo de 30%.
- 5 folds internos dentro do conjunto de desenvolvimento.

Esses dados ficam em:

`D:\kidney\dataset_geral_cv`

## Proximo passo metodologico

A etapa de pseudo-rotulagem mostrou que a confianca numerica do segmentador nao
e suficiente para garantir que uma imagem seja anatomicamente adequada. Algumas
imagens podem receber mascara aceita pelo modelo, mas ainda assim nao mostrar um
rim claro ou nao apresentar parenquima/piramides renais avaliaveis.

Por isso, o proximo passo do projeto sera criar um terceiro modelo, voltado para
avaliacao intrarrenal. Esse modelo usara a saida do DeepLab campeao como ROI
renal e devera se concentrar no conteudo interno do rim, com foco em qualidade
anatomica, parenquima visivel e padroes texturais sugestivos de alteracao
parenquimatosa.

O plano detalhado esta documentado em:

`D:\kidney\docs\proximos_passos_modelo_intrarrenal.md`
