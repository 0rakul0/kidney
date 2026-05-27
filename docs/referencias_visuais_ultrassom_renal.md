# Referencias visuais para curadoria intrarrenal

Este documento lista fontes publicas para comparar exemplos do `dataset_geral`
com imagens de referencia de ultrassom renal normal e imagens com alteracoes
parenquimatosas. A finalidade e apoiar a curadoria do modelo 3, nao diagnosticar
fibrose diretamente.

## Caracteristicas de rim com aspecto preservado

Referencia principal:

- Loyola University Medical Education Network. Normal Renal Ultrasound.
  https://www.meddean.luc.edu/lumen/meded/urology/uskidnl.htm

Caracteristicas descritas:

- rim em corte longitudinal;
- parenquima renal nao homogeneo;
- cortex periferico acinzentado;
- piramides renais mais escuras e distribuidas ao redor da regiao interna;
- hilo/seio renal central mais claro.

Outra fonte aberta:

- Wikimedia Commons. Normal adult kidney.jpg.
  https://commons.wikimedia.org/wiki/File:Normal_adult_kidney.jpg

Essa imagem esta sob licenca Creative Commons Attribution 4.0 e pode ser usada
como referencia visual com citacao adequada.

## Caracteristicas sugestivas de alteracao parenquimatosa

Referencias:

- NephroPOCUS. Increased renal cortical echogenicity does not always indicate
  chronic kidney disease.
  https://nephropocus.com/2019/06/02/increased-renal-cortical-echogenicity-does-not-always-indicate-chronic-kidney-disease/amp/

- O'Neill WC. Renal Relevant Radiology: Use of Ultrasound in Kidney Disease and
  Nephrology Procedures. Clinical Journal of the American Society of Nephrology.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3913230/

Caracteristicas descritas:

- aumento de ecogenicidade cortical;
- reducao ou perda de diferenciacao cortico-medular;
- parenquima fino em doenca cronica avancada;
- rins menores ou atroficos em alguns cenarios cronicos;
- hiperecogenicidade cortical correlacionada na literatura com alteracoes como
  fibrose intersticial, atrofia tubular e glomeruloesclerose, embora nao seja
  especifica.

## Como salvar referencias externas

As imagens externas baixadas manualmente devem ser salvas em:

`D:\kidney\dataset_aumentado\fontes\external_data\reference_ultrasound\images`

Sugestao de subpastas:

```text
D:\kidney\dataset_aumentado\fontes\external_data\reference_ultrasound\images
|-- healthy
|-- suspicious
`-- unknown
```

Use nomes de arquivo que preservem a origem, por exemplo:

```text
healthy\wikimedia_normal_adult_kidney_hansen_2015.png
suspicious\nephropocus_ckd_hyperechoic_cortex_example_01.png
```

## Aplicar o modelo 2 nas imagens externas

Depois de colocar as imagens na pasta acima, rode:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\segment_external_reference_images.py
```

Saidas:

- `D:\kidney\results\external_reference_segmentation\images`
- `D:\kidney\results\external_reference_segmentation\masks_model2`
- `D:\kidney\results\external_reference_segmentation\panels_model2`
- `D:\kidney\results\external_reference_segmentation\external_reference_model2_manifest.csv`

Nos paineis gerados, a linha amarela corresponde ao contorno da mascara prevista
pelo modelo 2, DeepLabV3-ResNet50 campeao.

## Observacao metodologica

As imagens externas servem como referencia visual e como apoio a construcao dos
criterios de curadoria. Elas devem ser separadas do treinamento principal, a
menos que a licenca permita uso em treinamento e a origem seja documentada.
