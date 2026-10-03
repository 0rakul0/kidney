# Scripts auxiliares, features e figuras

[← Wiki](README.md)

## Visão geral

Além do pipeline principal, o repositório contém scripts para classificação exploratória, extração de features, geração de figuras e compatibilidade com código antigo.

---

# `run_pipeline.py`

## Função

Runner simplificado do eixo de atributos/classificação renal.

### Comandos

```text
status
init-labels
extract
train
```

---

## `status`

Verifica:
- existência de `renal_features.csv`;
- existência de `renal_labels.csv`;
- quantidade de labels preenchidos;
- distribuição por split e classe;
- quantidade de máscaras de referência.

Depois sugere a próxima ação.

---

## `init-labels`

Executa:
`engenharia_dataset/prepare_renal_labels_template.py`

Cria o template de rótulos.

---

## `extract`

Executa:
`engenharia_dataset/extract_renal_features.py`

Produz features quantitativas.

---

## `train`

Executa:
`train_renal_classifier.py`

Treina o classificador tabular.

---

# `utils/renal_features.py`

É o núcleo da extração de descritores.

## Carregamento
- `load_grayscale`: lê imagem em escala de cinza;
- `load_binary_mask`: lê máscara binária;
- `normalize_image`: normaliza intensidades.

## Máscaras derivadas

### `get_inner_mask`
Erode a máscara renal para criar uma região interna.

### `get_cortex_band`
Cria uma banda periférica aproximada entre a máscara renal e a região interna.

### `get_reference_band`
Cria outra região de referência morfológica.

Essas regiões são heurísticas e não substituem as máscaras anatômicas multiclasse atuais.

---

## Features de intensidade

`intensity_features` calcula descritores estatísticos dentro da máscara.

---

## Pontos brilhantes

`bright_spot_features` usa um percentil de intensidade para medir:
- quantidade;
- proporção;
- intensidade de pixels muito brilhantes.

---

## Textura

`texture_features` usa uma matriz de coocorrência de níveis de cinza simplificada para gerar descritores de textura.

---

## Razões

`ratio_features` calcula relações entre medidas de regiões distintas.

---

## `heuristic_pyramid_mask`

Cria uma máscara heurística de regiões medulares/piramidais com base em intensidade e geometria interna.

Esse algoritmo serviu como baseline e foi posteriormente superado pelos modelos supervisionados de Medulla.

---

## `extract_renal_features`

Função agregadora.

Recebe:
- imagem;
- máscara renal;
- opcionalmente máscara de referência.

Retorna o conjunto completo de descritores derivados.

---

# `train_renal_classifier.py`

## Função

Treina um classificador usando as features tabulares.

### Entrada
- `renal_features.csv`;
- `renal_labels.csv`.

### Comportamento
1. junta features e rótulos;
2. remove linhas sem label;
3. seleciona colunas numéricas úteis;
4. separa dados;
5. treina modelos/classificador definido no script;
6. calcula métricas;
7. salva resultados.

### Saída
`results/renal_classifier/`

### Status científico
É um eixo exploratório. A qualidade do classificador depende da validade dos rótulos clínicos fornecidos.

---

# Figuras do artigo

## `build_article_capsule_figure.py`

Gera figura específica do artigo para segmentação da cápsula.

### Processamento
- carrega casos definidos no script;
- carrega checkpoint U-Net;
- executa predição;
- calcula Dice/IoU;
- sobrepõe referência e predição;
- monta a figura final.

---

## `build_article_quality_figures.py`

Transforma os painéis qualitativos produzidos pelos experimentos em figuras prontas para o artigo.

### Comportamento
- remove margens;
- recorta a linha principal;
- separa painéis;
- lê métricas;
- adiciona títulos;
- compõe canvas final.

O script usa caminhos locais históricos de `D:\kidney`, portanto pode precisar de adaptação se o projeto for executado em outro ambiente.

---

## `build_discussion_quality_figure.py`

Monta uma figura de discussão contrastando caso de falha e caso de sucesso.

Ele lê:
- imagem problemática;
- imagem bem segmentada;
- máscaras U-Net/DeepLab;
- contornos.

Também possui caminhos locais históricos e é voltado à produção do material do artigo.

---

# Compatibilidade em `utils/`

Arquivos como:
- `utils/dataset.py`;
- `utils/losses.py`;
- `utils/metrics.py`;
- `utils/model_loader.py`;
- `utils/segmentation_training.py`;
- `utils/segmentation_evaluation.py`

funcionam como camada de compatibilidade/reexportação para os módulos equivalentes em `src/segmentation/core/`.

O código novo deve preferir:

```text
src/segmentation/core/
```

---

# Scripts de referência/benchmark externo

## `experiments/run_kidneyus_nnunet_inference.py`

Executa modelos nnU-Net externos sobre o conjunto local.

### Comportamentos importantes
- corrige compatibilidade de `torch.load` com checkpoints antigos;
- adapta pré-processamento do nnU-Net para Windows;
- detecta folds disponíveis;
- converte PNG para NIfTI;
- executa inferência;
- reconverte a máscara;
- calcula Dice, IoU, precisão, recall, F1 e Hausdorff;
- gera overlays;
- consolida métricas.

### Uso
Serve como referência externa de arquitetura/pesos, não como componente obrigatório do pipeline principal.

---

# Como interpretar os auxiliares

Os scripts auxiliares podem ser divididos em:

| Tipo | Papel |
|---|---|
| Runner | facilitar execução |
| Feature extraction | explorar descritores tabulares |
| Classificação | testar hipótese com labels |
| Figuras | produzir material científico |
| Compatibilidade | manter código antigo funcionando |
| Benchmark externo | comparar com modelos de terceiros |

Eles não devem ser confundidos com o núcleo atual de segmentação + curadoria + ecogenicidade.
