# Execução e reprodutibilidade

[← Wiki](README.md)

## Curadoria Web

```powershell
.\.venv\Scripts\python.exe .\curadoria_web\app.py
```

Abra `http://127.0.0.1:8765`. A página `/inferencia.html` executa a cascata local quando os checkpoints estão disponíveis.

## Construir dataset geral

```powershell
.\.venv\Scripts\python.exe src\segmentation\build_dataset_geral.py --clear-output
```

## Criar splits

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\create_dataset_geral_splits.py --clear-output --link-mode hardlink --folds 5 --test-ratio 0.30 --seed 42
```

## Consenso da cápsula

```powershell
.\.venv\Scripts\python.exe src\segmentation\run_capsule_model_consensus.py
```

## Runner de atributos/classificação

```powershell
python run_pipeline.py status
python run_pipeline.py init-labels
python run_pipeline.py extract
python run_pipeline.py train
```

## Checklist mínimo

Registre commit, dataset, manifesto, split, seed, arquitetura, checkpoint, tamanho de entrada, pré-processamento, aumento de dados, loss, otimizador, limiar, pós-processamento, conjunto de avaliação e status de curadoria.

## Artefatos canônicos

- `docs/`
- `src/segmentation/`
- `engenharia_dataset/`
- `models/`
- `results/`
- `artigo/SBBD_2026___Jefferson/`
