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
- checkpoint: `models/kidneyus_capsule_student_unet_deeplab.pth`;
- CLAHE, como no treinamento;
- limiar de segmentacao `0.60`, selecionado na validacao do modelo vencedor;
- media das predicoes original e espelhada horizontalmente.

Em 28/10/2026, o comando foi executado sobre o conjunto MONAI disponível no
projeto. O resultado registrado em `dataset_aumentado/dataset_geral_v2/summary.json`
contém 4.479 imagens MONAI, das quais 3.707 receberam pseudomáscaras não vazias
com esse checkpoint; todas permanecem marcadas para revisão humana.

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

## Destilacao de conhecimento para a U-Net operacional

O caminho experimental para transformar o consenso em um modelo único está
implementado em `experiments/train_distilled_unet.py`. A U-Net e a DeepLabV3
atuam como professores; suas probabilidades são combinadas pela média
aritmética e usadas, junto às máscaras manuais, para treinar uma terceira U-Net
estudante. A temperatura e os pesos das perdas são registrados no metadado do
checkpoint. O parâmetro `--consensus-only` permite restringir o treinamento às
imagens cujo Dice entre professores seja pelo menos o valor de
`--min-teacher-agreement`.

O comando reproduzível é:

```powershell
.\.venv\Scripts\python.exe src\segmentation\experiments\train_distilled_unet.py `
  --epochs 12 `
  --early-stopping 4 `
  --batch-size 16 `
  --base-channels 32 `
  --experiment-name capsule_distilled_unet_v1
```

O script gera um cache dos professores em
`results/segmentation_experiments/capsule_distilled_unet/teacher_cache/`, salva
o checkpoint estudante em `models/kidneyus_capsule_distilled_unet.pth` e grava
histórico e resumo JSON. O conjunto deduplicado mantém 328 imagens para
treino, 70 para validação e 70 para teste; o teste manual só é consultado após
a seleção do checkpoint.

Na primeira execução registrada, a estudante de 32 canais obteve Dice de
`0,8847` e IoU de `0,7932` no teste independente, enquanto os professores
obtiveram Dice de `0,9302` (U-Net) e `0,9182` (DeepLabV3). A estudante foi
adotada como modelo operacional único da Curadoria Web para concentrar o
conhecimento dos professores e reduzir o custo de memória da cascata. Essa
adoção é uma decisão de implantação e não uma alegação de superioridade: os
professores continuam sendo referências de controle e a qualidade deve ser
monitorada em avaliações posteriores.

O checkpoint vencedor da matriz operacional é
`models/kidneyus_capsule_student_unet_deeplab.pth`, treinado com U-Net e
DeepLabV3 como professores. Ele é o padrão usado pela Curadoria Web e pelo
`build_dataset_geral.py` para gerar pseudomáscaras das imagens do MONAI.

### Estudante intrarrenal destilada

A etapa interna também possui uma estudante destilada em
`experiments/train_distilled_inner_unet.py`. Ela recebe três canais (imagem em
tons de cinza, ROI renal mascarada e máscara do rim), produz quatro classes
(fundo, córtex, medula e CEC) e aprende com as máscaras manuais e com a média
das probabilidades dos professores intrarrenais U-Net e DeepLabV3.

Comando usado na RTX 3050:

```powershell
.\.venv\Scripts\python.exe src\segmentation\experiments\train_distilled_inner_unet.py `
  --epochs 50 `
  --batch-size 8 `
  --teacher-batch-size 1 `
  --early-stopping 10
```

O checkpoint operacional é
`models/intrarenal_student_unet_deeplab.pth`. No teste independente de 50
imagens, a estudante obteve Dice médio foreground `0,750075` e IoU médio
foreground `0,605106`. Por estrutura, os Dice foram `0,695894` para córtex,
`0,707624` para medula e `0,846706` para CEC. As métricas são de segmentação
anatômica no conjunto manual; as máscaras produzidas em novas imagens pela
Curadoria Web continuam sendo propostas para revisão humana.

No benchmark controlado da RTX 3050, com lote 8, entrada 256x256 e medição
somente do `forward`, a estudante alcançou 153,51 FPS (6,51 ms por imagem),
enquanto o SegFormer-B0 alcançou 365,61 FPS (2,74 ms por imagem). O SegFormer
continua sendo o mais rápido, mas a estudante é mais leve que os dois
professores e mantém a arquitetura convolucional escolhida para a aplicação.

