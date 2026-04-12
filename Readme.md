# Identificacao de Fibrose Renal por Visao Computacional

Este projeto investiga a identificacao de sinais de fibrose renal em imagens de ultrassonografia usando visao computacional e aprendizado profundo.

O estado atual do projeto ja resolve uma etapa importante: a segmentacao automatica do rim. A nova proposta transforma essa segmentacao em uma etapa intermediaria de um pipeline maior, cujo objetivo final e analisar a regiao interna do rim, especialmente as piramides renais, para detectar padroes de opacidade e hiperecogenicidade associados a doenca.

## Objetivo Atual

O objetivo do projeto nao e mais apenas localizar o rim na imagem.

O objetivo passou a ser:

1. segmentar o rim com boa precisao;
2. usar essa mascara para isolar o orgao;
3. segmentar internamente as piramides renais;
4. medir opacidade, brilho e concentracao de pontos brancos dentro do rim;
5. comparar, quando possivel, a ecogenicidade do rim com a de orgaos proximos, como figado ou pancreas;
6. apoiar a identificacao de rins com sinais compativeis com fibrose ou alteracao patologica.

## Motivacao Clinica

Na ultrassonografia, rins doentes podem apresentar alteracoes visuais relevantes no parenquima, como:

- aumento de brilho difuso;
- presenca de muitos pontos brancos;
- alteracao de contraste interno;
- mudanca relativa de ecogenicidade em comparacao com estruturas adjacentes.

Dentro da proposta deste projeto, esses sinais serao investigados de forma computacional a partir da segmentacao do rim e da analise das suas estruturas internas.

## Estado Atual

O projeto ja possui:

- dataset organizado em `train`, `val` e `test`;
- mascaras binarias do rim;
- pipeline de treinamento para segmentacao semantica;
- benchmark entre multiplas arquiteturas;
- scripts de visualizacao e avaliacao;
- pseudo-labeling para ampliar a base com novas mascaras do rim.

O projeto agora tambem possui:

- extracao automatica de features intrarrenais;
- mascara interna candidata das piramides renais por heuristica;
- exportacao de CSV para analise quantitativa da ROI renal;
- suporte a mascara manual de orgao de referencia, como figado.

O projeto ainda nao possui:

- mascaras das piramides renais;
- rotulos clinicos de fibrose;
- classificador final de rim saudavel versus rim doente;
- comparacao automatica com figado ou pancreas;
- metrica final clinica para a nova tarefa.

Quando houver uma ROI manual do figado, o projeto ja consegue comparar rim e figado de forma quantitativa.

## Estrutura do Projeto

```text
D:\kidney
|-- dataset/
|   |-- train/
|   |   |-- image/
|   |   `-- mask/
|   |-- val/
|   |   |-- image/
|   |   `-- mask/
|   `-- test/
|       |-- image/
|       `-- mask/
|-- dataset_loader/
|-- experiments/
|   |-- train_deeplab.py
|   |-- train_segformer.py
|   |-- train_unet.py
|   `-- train_unetplusplus.py
|-- models/
|-- results/
|-- utils/
|   |-- dataset.py
|   |-- losses.py
|   `-- metrics.py
|-- artigo/
|-- annotate_reference_roi.py
|-- benchmark_models.py
|-- divisor_segmentation.py
|-- evaluate_models.py
|-- extract_renal_features.py
|-- generate_prediction_samples.py
|-- prepare_renal_labels_template.py
|-- run_pipeline.py
|-- train_renal_classifier.py
|-- visualizar_resultados.py
`-- Readme.md
```

## Dataset

O dataset atual foi preparado para segmentacao do rim, nao para segmentacao interna nem para classificacao de fibrose.

## Proveniencia dos Dados

As imagens utilizadas neste repositorio foram organizadas localmente a partir de material recebido no arquivo `flood_1.zip`. Conforme a comunicacao recebida junto ao compartilhamento desse arquivo, o conjunto de imagens usado no estudo tem como origem o projeto publico [The Open Kidney Ultrasound Data Set (`rsingla92/kidneyUS`)](https://github.com/rsingla92/kidneyUS), que tambem serviu como principal referencia tecnica para a etapa de segmentacao renal.

Para fins de documentacao e publicacao deste projeto, a cadeia de proveniencia dos dados e descrita da seguinte forma:

```text
Open Kidney Ultrasound Dataset (fonte publica citada pelo autor)
-> arquivo flood_1.zip recebido por e-mail
-> organizacao local em dataset_loader/, identificada/ e dataset/
-> split final train/val/test usado neste repositorio
```

Registro de procedencia local:

- a pasta `dataset/` usada neste repositorio foi montada a partir do arquivo `flood_1.zip`;
- esse arquivo foi recebido por e-mail no contexto de compartilhamento do material de pesquisa;
- a mensagem recebida informa que os dados do estudo foram obtidos do Open Kidney Ultrasound Dataset.

Caracteristicas descritas pelo acervo de referencia:

- mais de 500 imagens bidimensionais de ultrassom abdominal em modo B;
- imagens renais anotadas por especialista;
- uma imagem por paciente;
- aquisicoes clinicas realizadas entre janeiro de 2015 e setembro de 2019;
- acesso aos dados mediante registro;
- uso nao comercial, conforme a documentacao do projeto original.

### Catalogacao Local das Imagens

Catalogacao atual das pastas locais:

- `dataset_loader/`: 536 imagens PNG do acervo bruto/local de trabalho;
- `identificada/image/`: 498 imagens com identificacao consistente para anotacao;
- `identificada/mask/`: 498 mascaras binarias correspondentes;
- `dataset/train/image/` e `dataset/train/mask/`: 360 pares imagem-mascara;
- `dataset/val/image/` e `dataset/val/mask/`: 51 pares imagem-mascara;
- `dataset/test/image/` e `dataset/test/mask/`: 102 pares imagem-mascara;
- `reference_masks/train/`: 7 mascaras manuais de orgao de referencia;
- `nao_identificada/`: 36 imagens ainda fora do split principal.

Isso significa que o split principal atualmente usado pelos scripts de treino e avaliacao contem `513` pares imagem-mascara (`360 + 51 + 102`).

### Fonte e Citacao

- repositorio: [https://github.com/rsingla92/kidneyUS](https://github.com/rsingla92/kidneyUS)
- artigo: Singla, R. et al. "The Open Kidney Ultrasound Data Set". International Workshop on Advances in Simplifying Medical Ultrasound, 2023.
- observacao de licenca: o repositorio original informa disponibilizacao sob `CC BY-NC-SA`, com restricao para uso comercial e liberacao dos dados mediante cadastro.

Formato esperado:

```text
dataset/
    train/
        image/
        mask/
    val/
        image/
        mask/
    test/
        image/
        mask/
```

Cada imagem possui:

- uma imagem de ultrassom em tons de cinza;
- uma mascara binaria delimitando o rim;
- o mesmo nome de arquivo em `image/` e `mask/`.

### Limitacao do dataset atual

Hoje a mascara marca apenas o contorno externo do rim.

Para atacar o novo problema, idealmente sera necessario adicionar pelo menos um destes tipos de anotacao:

1. mascara das piramides renais;
2. mascara de regioes suspeitas de alteracao;
3. rotulo por imagem ou por rim, como `saudavel`, `suspeito`, `doente`;
4. grau ordinal de alteracao, como `leve`, `moderado`, `grave`.

## Pipeline Atual

O pipeline implementado hoje e:

```text
Imagem de ultrassom
-> pre-processamento basico
-> redimensionamento
-> conversao para 3 canais
-> segmentacao do rim
-> mascara binaria do orgao
```

Esse pipeline ja atende bem a etapa de isolamento anatomico.

## Novo Pipeline Proposto

Com a redefinicao do projeto, o pipeline alvo passa a ser:

```text
Imagem de ultrassom
-> pre-processamento
-> segmentacao do rim
-> recorte da ROI renal
-> segmentacao interna das piramides renais
-> extracao de atributos de opacidade e ecogenicidade
-> comparacao com orgaos adjacentes, quando possivel
-> classificacao de rim saudavel ou com sinais de alteracao
```

## Desafio Central

O gargalo tecnico agora nao e mais encontrar o rim.

O gargalo e:

1. localizar estruturas internas do rim;
2. separar as piramides renais do restante do parenquima;
3. medir quantitativamente o quanto a regiao esta clara, heterogenea ou pontilhada;
4. transformar essa observacao visual em criterio computacional robusto.

Em termos práticos, a mascara do rim resolve a pergunta:

```text
onde esta o rim?
```

Mas a nova proposta precisa responder:

```text
como esta o interior do rim?
```

## Modelos Ja Implementados

O projeto ja possui treinamento e avaliacao para os seguintes modelos de segmentacao:

- U-Net
- U-Net++
- DeepLabV3
- SegFormer

Arquivos principais:

- `experiments/train_unet.py`
- `experiments/train_unetplusplus.py`
- `experiments/train_deeplab.py`
- `experiments/train_segformer.py`
- `evaluate_models.py`
- `benchmark_models.py`

### Inspiracao externa para segmentacao

A etapa de segmentacao renal deste repositorio foi inspirada pelo projeto `kidneyUS`, tanto pela organizacao do acervo quanto pela proposta de benchmarking em ultrassom renal. Por isso, o README agora registra explicitamente essa dependencia intelectual e a fonte das imagens.

Os scripts `evaluate_models.py` e `benchmark_models.py` tambem passaram a aceitar checkpoints externos, o que facilita testar um modelo vindo do projeto de referencia quando o arquivo de pesos estiver disponivel localmente.

Exemplos:

```bash
python evaluate_models.py --model deeplab --checkpoint models\meu_modelo_externo.pth --backbone resnet50
python benchmark_models.py --model segformer --checkpoint models\meu_modelo_externo.pth --segformer-backbone nvidia/segformer-b0-finetuned-ade-512-512
```

Se o checkpoint externo vier acompanhado de um arquivo `.meta.json`, o projeto tambem aproveita automaticamente metadados como `best_threshold` e configuracoes de arquitetura. Exemplo:

```json
{
  "best_threshold": 0.5,
  "model_kwargs": {
    "backbone": "resnet50"
  }
}
```

## Parte Tecnica

### Pre-processamento atual

O projeto aplica, dependendo do script:

- resize para `256x256`;
- normalizacao para intervalo `[0, 1]`;
- replicacao do canal grayscale para 3 canais;
- remocao de bordas pretas;
- reducao de ruido com `fastNlMeansDenoising`;
- melhoria de contraste com `CLAHE`.

### Funcoes de perda

Na etapa de segmentacao do rim, a combinacao usada e:

```text
BCEWithLogitsLoss + DiceLoss
```

Essa escolha e adequada para segmentacao binaria com desbalanceamento entre fundo e objeto.

### Metricas atuais

Para segmentacao do rim, o projeto ja utiliza ou ja calcula:

- Dice Score;
- IoU;
- Precision;
- Recall;
- F1;
- Hausdorff Distance;
- FPS em benchmark.

### O que falta tecnicamente para a nova proposta

Para que o projeto fique aderente ao novo objetivo, ainda sera necessario implementar:

1. uma estrategia de segmentacao interna das piramides renais;
2. uma etapa de extracao de caracteristicas radiomicas ou estatisticas da ROI interna;
3. uma metrica de opacidade/ecogenicidade relativa;
4. uma rotina de deteccao de pontos hiperecogenicos;
5. uma logica de comparacao com figado ou pancreas, quando presentes;
6. um classificador final baseado em regras, aprendizado supervisionado ou abordagem hibrida.

### O que ja foi implementado com base na literatura

Com base em trabalhos que usam a ROI do rim como entrada para analise de textura e ecogenicidade, o projeto passou a incluir uma etapa intermediaria executavel:

- extracao de estatisticas de intensidade da mascara renal;
- extracao de estatisticas da regiao interna erodida;
- deteccao de pontos brilhantes dentro do rim;
- metricas de textura por GLCM;
- comparacao entre o rim e uma banda externa de referencia;
- comparacao entre o rim e uma ROI manual de figado, quando disponivel;
- geracao de uma mascara candidata de estruturas internas escuras compativeis com piramides.

Essa etapa nao substitui a anotacao manual das piramides, mas reduz a distancia entre a segmentacao externa atual e a classificacao final do orgao.

## Hipotese Computacional Atual

A hipotese de trabalho do projeto e:

- se o rim apresentar regiao interna excessivamente clara;
- se houver muitos pontos brancos distribuidos no interior do orgao;
- se a ecogenicidade renal estiver elevada em relacao a estruturas de referencia;

entao pode haver sinal compativel com alteracao patologica, incluindo suspeita de fibrose.

Essa hipotese ainda precisa ser validada com rotulos apropriados e criterios mais objetivos.

## Caminhos Tecnicos Possiveis Para a Etapa Interna

Existem pelo menos quatro caminhos viaveis para a proxima fase:

1. Segmentacao supervisionada das piramides.
Requer novas mascaras manuais e permite usar U-Net, DeepLab ou SegFormer tambem para a estrutura interna.

2. Segmentacao fraca ou semi-supervisionada.
Pode combinar heuristicas de intensidade, watershed, clustering ou pseudo-labeling com revisao manual.

3. Analise por textura sem segmentacao perfeita.
Depois de segmentar o rim, e possivel extrair descritores como histograma, entropia, LBP, GLCM e distribuicao de pixels brilhantes.

4. Pipeline hibrido.
Segmenta o rim com rede neural e usa regras classicas de processamento de imagem para localizar as piramides e medir opacidade.

## Comparacao Com Orgaos Adjuntos

Uma extensao importante da proposta e comparar a ecogenicidade do rim com a de outros orgaos proximos.

Na pratica, isso significa:

- detectar se figado ou pancreas aparecem no enquadramento;
- segmentar ou aproximar uma ROI desses orgaos;
- medir intensidade media, contraste e distribuicao tonal;
- comparar com a intensidade do rim ou das piramides.

Esse passo pode ser muito util, mas nao deve ser obrigatorio em todas as imagens, porque nem sempre esses orgaos estao bem visiveis.

No estado atual do projeto, a forma mais confiavel de usar essa comparacao e com anotacao manual da ROI de referencia.

## Scripts Principais

### Treinamento

- `experiments/train_unet.py`
- `experiments/train_unetplusplus.py`
- `experiments/train_deeplab.py`
- `experiments/train_segformer.py`

### Avaliacao e benchmark

- `evaluate_models.py`
- `benchmark_models.py`

### Visualizacao

- `generate_prediction_samples.py`
- `visualizar_resultados.py`

### Pseudo-labeling

- `divisor_segmentation.py`

### Analise intrarrenal

- `extract_renal_features.py`
- `utils/renal_features.py`

### Anotacao de ROI de referencia

- `annotate_reference_roi.py`

### Classificacao experimental

- `prepare_renal_labels_template.py`
- `run_pipeline.py`
- `train_renal_classifier.py`

## Requisitos

Dependencias principais:

```bash
pip install torch torchvision opencv-python numpy matplotlib pandas tqdm transformers scipy
```

Versao recomendada:

- Python 3.9 ou superior

## Como Executar

### Fluxo simplificado

Para evitar decorar varios scripts, use:

```bash
python run_pipeline.py status
python run_pipeline.py init-labels
python run_pipeline.py extract
python run_pipeline.py train
```

Significado:

- `status`: mostra quantas imagens ja tem features, quantos labels foram preenchidos e quantas mascaras de referencia existem;
- `init-labels`: cria o template inicial de labels;
- `extract`: recalcula as features intrarrenais;
- `train`: treina o classificador experimental.

### Treinar modelos

Os scripts de treinamento estao em `experiments/`.

Exemplos:

```bash
python experiments/train_unet.py
python experiments/train_unetplusplus.py
python experiments/train_deeplab.py
python experiments/train_segformer.py
```

Agora os scripts aceitam hiperparametros por linha de comando para testar aumento de capacidade, regularizacao e limiar de segmentacao.

Exemplos:

```bash
python experiments/train_unet.py --base-channels 96 --augment --auto-pos-weight --scheduler cosine --optimizer adamw
python experiments/train_unetplusplus.py --base-channels 96 --augment --weight-decay 1e-4
python experiments/train_deeplab.py --backbone resnet101 --auto-pos-weight --scheduler plateau
python experiments/train_segformer.py --backbone-name nvidia/segformer-b2-finetuned-ade-512-512 --batch-size 4 --augment
```

Os treinos agora tambem salvam:

- historico por epoca em `results/segmentation_experiments/`;
- resumo JSON do experimento;
- metadados do checkpoint com melhor `threshold` e configuracao usada.

### Busca rapida de hiperparametros

Para rodar uma busca inicial com presets focados em capacidade de segmentacao:

```bash
python experiments/run_hyperparameter_search.py --model unet --epochs 40
python experiments/run_hyperparameter_search.py --model all --epochs 30
```

Os resultados consolidados sao salvos em:

```text
results/segmentation_experiments/hyperparameter_search_summary.csv
```

### Avaliar modelos

```bash
python evaluate_models.py
python benchmark_models.py
```

Se existir um arquivo de metadados ao lado do checkpoint, os scripts de avaliacao passam a reutilizar automaticamente o `threshold` e a variante do modelo usados no melhor experimento.

### Gerar amostras visuais

```bash
python generate_prediction_samples.py
python visualizar_resultados.py
```

### Gerar pseudo-labels

```bash
python divisor_segmentation.py
```

### Anotar ROI de figado ou outro orgao de referencia

```bash
python annotate_reference_roi.py
```

Teclas:

- clique esquerdo adiciona pontos do poligono;
- clique direito fecha o poligono;
- `s` salva a mascara poligonal da imagem atual;
- `r` reinicia o poligono da imagem atual;
- `n` pula para a proxima imagem sem salvar ROI;
- `q` encerra a anotacao;
- `Ctrl+C` no terminal tambem encerra a anotacao;
- fechar a janela encerra a anotacao.

As mascaras devem ser salvas em:

```text
reference_masks/
    train/
    val/
    test/
```

Cada mascara deve ter o mesmo nome do arquivo da imagem original.

### Extrair features intrarrenais

```bash
python extract_renal_features.py
```

Saidas geradas:

- `results/renal_feature_analysis/renal_features.csv`
- `results/renal_feature_analysis/renal_features_summary.csv`
- `results/renal_feature_analysis/candidate_masks/`

O CSV contem, por imagem:

- intensidade media e dispersao do rim;
- intensidade da regiao interna e da banda cortical;
- razao de intensidade entre rim e referencia externa;
- proporcao de pixels brilhantes;
- numero de componentes brilhantes;
- features de textura por coocorrencia;
- proporcao da mascara candidata das piramides.

Quando existir uma mascara em `reference_masks/<split>/`, o script tambem calcula:

- intensidade media da ROI de referencia;
- razao `rim / figado` ou `rim / referencia`;
- razao `interior do rim / referencia`.

### Preparar template de rotulos

```bash
python prepare_renal_labels_template.py
```

Esse comando gera:

- `results/renal_feature_analysis/renal_labels.csv`

Preencha manualmente:

- `label = 0` para rim saudavel;
- `label = 1` para rim doente;
- `label_name` com um texto opcional como `healthy` ou `diseased`.

### Treinar classificador experimental

```bash
python train_renal_classifier.py
```

Saidas geradas:

- `results/renal_classifier/metrics.json`
- `results/renal_classifier/classification_report.json`
- `results/renal_classifier/confusion_matrix.json`
- `results/renal_classifier/feature_importance.csv`
- `results/renal_classifier/test_predictions.csv`

O classificador atual usa `RandomForest` sobre as features extraidas da ROI renal.

## Roadmap

Proxima etapa recomendada:

1. padronizar os scripts para rodar da raiz do projeto;
2. criar um subconjunto anotado das piramides renais;
3. definir uma metrica objetiva de opacidade interna;
4. testar um primeiro pipeline hibrido para a ROI interna;
5. construir um classificador inicial de rins com e sem sinal de alteracao usando tambem a razao rim/figado quando disponivel;
6. revisar o artigo com base nos primeiros resultados da etapa interna.

## Resumo da Situacao Atual

Hoje o projeto ja faz bem a segmentacao externa do rim.

O problema cientifico principal, a partir de agora, e avancar da pergunta:

```text
onde esta o rim?
```

para a pergunta:

```text
o interior desse rim apresenta sinais visuais compativeis com fibrose?
```

Essa transicao define a nova fase do projeto.

## Referencias e Inspiracao

Os seguintes trabalhos inspiram diretamente a formulacao atual do projeto:

- Development and Validation of a Deep Learning Model to Quantify Interstitial Fibrosis and Tubular Atrophy From Kidney Ultrasonography Images. Mostra um pipeline de `segmentacao do rim -> extracao de features -> classificacao`. Fonte: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8144924/)
- Ultrasound-based radiomics analysis in the assessment of renal fibrosis in patients with chronic kidney disease. Inspira o uso de radiomics e variaveis quantitativas da ROI renal. Fonte: [PubMed](https://pubmed.ncbi.nlm.nih.gov/37256330/)
- Diagnostic accuracy of ultrasound-based multimodal radiomics modeling for fibrosis detection in chronic kidney disease. Reforca o uso de ultrassom quantitativo e modelos multimodais para fibrose. Fonte: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10017610/)
- How echogenic is echogenic? Quantitative acoustics of the renal cortex. Sustenta a comparacao quantitativa entre ecogenicidade renal e hepática. Fonte: [PubMed](https://pubmed.ncbi.nlm.nih.gov/11273869/)
- Sonographically determined kidney measurements are better able to predict histological changes and a low CKD-EPI eGFR when weighted towards cortical echogenicity. Apoia a ecogenicidade cortical como indicador relevante em doenca renal. Fonte: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7137523/)
- Renal pyramid echogenicity in ureteropelvic junction obstruction: correlation between altered echogenicity and differential renal function. Justifica a observacao das piramides renais como alvo anatomico importante. Fonte: [PubMed](https://pubmed.ncbi.nlm.nih.gov/18633607/)
- A novel convolutional neural network for kidney ultrasound images segmentation. Da suporte ao uso de segmentacao profunda do rim como primeira etapa do pipeline. Fonte: [PubMed](https://pubmed.ncbi.nlm.nih.gov/35248816/)
