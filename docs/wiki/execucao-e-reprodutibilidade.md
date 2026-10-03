# Execução e reprodutibilidade

[← Wiki](README.md)

## Objetivo desta página

Esta página funciona como guia operacional para reproduzir os principais fluxos do projeto.

Ela não substitui os parâmetros específicos de cada experimento, mas indica:

- onde estão os scripts;
- qual comando inicia cada etapa;
- quais artefatos devem ser preservados;
- quais informações devem acompanhar um resultado científico.

## Estrutura operacional

```text
kidney/
├── src/
│   └── segmentation/
├── engenharia_dataset/
├── curadoria_web/
├── models/
├── results/
├── docs/
├── artigo/
├── config/
└── run_pipeline.py
```

## 1. Verificar o ambiente

O projeto depende de Python e, para os modelos, PyTorch.

A Curadoria Web possui dependências próprias em:

```text
curadoria_web/requirements.txt
```

Os checkpoints usados em inferência devem estar disponíveis em:

```text
models/
```

## 2. Curadoria Web

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe .\curadoria_web\app.py
```

A aplicação local fica em:

```text
http://127.0.0.1:8765
```

### Funções principais

- revisar máscaras;
- corrigir por polígono;
- alterar modelo intrarrenal exibido;
- registrar observações;
- exportar CSV/JSON;
- ajustar ROIs de ecogenicidade.

## 3. Inferência em nova imagem

A interface está disponível em:

```text
http://127.0.0.1:8765/inferencia.html
```

Ela recebe PNG, JPG ou JPEG.

O fluxo executa:

```text
imagem
→ U-Net de cápsula
→ ROI renal
→ U-Net multiclasse intrarrenal
→ Cortex / Medulla / CEC
→ painel de ecogenicidade
```

As inferências não entram automaticamente no conjunto curado.

## 4. Construir dataset geral

Comando:

```powershell
.\.venv\Scripts\python.exe src\segmentation\build_dataset_geral.py --clear-output
```

O processo consolida imagens, procura máscaras existentes e gera candidatos quando necessário.

### Artefatos esperados

```text
dataset_geral_v2/
├── imagens/
├── mascaras/
├── manifest.csv
├── summary.json
└── relatorios/
```

## 5. Criar splits

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\create_dataset_geral_splits.py --clear-output --link-mode hardlink --folds 5 --test-ratio 0.30 --seed 42
```

A estratégia cria:

- holdout final;
- conjunto de desenvolvimento;
- cinco folds.

## 6. Consenso da cápsula

```powershell
.\.venv\Scripts\python.exe src\segmentation\run_capsule_model_consensus.py
```

Esse script compara as predições dos modelos selecionados e gera indicadores para revisão.

## 7. Calibração

O repositório contém scripts para calibração de:

- revisão da cápsula;
- consenso OOF;
- limiares;
- priorização.

Eles ficam em:

```text
src/segmentation/
```

## 8. Engenharia de dataset

Os scripts em:

```text
engenharia_dataset/
```

cobrem:

- download;
- conversão;
- curadoria;
- criação de manifestos;
- splits;
- pseudo-rótulos;
- extração de atributos.

Cada script deve ser executado considerando a versão do dataset para o qual foi criado.

## 9. Runner de atributos/classificação

O arquivo:

```text
run_pipeline.py
```

oferece quatro comandos:

```powershell
python run_pipeline.py status
python run_pipeline.py init-labels
python run_pipeline.py extract
python run_pipeline.py train
```

### status

Mostra a situação de:

- features;
- labels;
- máscaras de referência.

### init-labels

Cria template de rótulos.

### extract

Extrai atributos renais.

### train

Treina o classificador baseado nos atributos.

## 10. Checkpoints

Os modelos devem ser acompanhados por metadados.

A pasta `models/` contém arquivos `.meta.json` associados aos experimentos.

Isso permite registrar:

- arquitetura;
- conjunto de dados;
- hiperparâmetros;
- época;
- limiar;
- métricas.

## 11. Resultados

Resultados experimentais devem ser mantidos em:

```text
results/
```

Evite depender apenas de números copiados para README ou artigo. O ideal é manter:

- JSON;
- CSV;
- logs;
- predições;
- métricas;
- figuras.

## 12. Checklist de reprodutibilidade

Antes de considerar um resultado reproduzível, registrar:

1. commit Git;
2. dataset;
3. versão do manifesto;
4. regra de split;
5. seed;
6. arquitetura;
7. checkpoint;
8. tamanho da entrada;
9. CLAHE ou outro pré-processamento;
10. augmentation;
11. loss;
12. otimizador;
13. learning rate;
14. número de épocas;
15. limiar;
16. pós-processamento;
17. métricas;
18. status de curadoria;
19. hardware relevante.

## 13. Artefatos canônicos

| Tipo | Caminho |
|---|---|
| Código de segmentação | `src/segmentation/` |
| Engenharia de dados | `engenharia_dataset/` |
| Interface | `curadoria_web/` |
| Modelos | `models/` |
| Resultados | `results/` |
| Documentação | `docs/` |
| Wiki | `docs/wiki/` |
| Artigo ativo | `artigo/SBBD_2026___Jefferson/` |

## Regra de versionamento

Ao alterar:

- split;
- dataset;
- checkpoint;
- protocolo de curadoria;
- medida de ecogenicidade;

o resultado deve ser tratado como uma nova configuração experimental.

## Próxima leitura

Para entender por que algumas decisões antigas diferem do fluxo atual, consulte [Histórico metodológico](historico-metodologico.md).
