# Organizacao do repositorio

Este documento define a estrutura oficial do projeto e separa arquivos fonte,
datasets consolidados, artefatos experimentais e saidas temporarias. A regra
geral e preservar rastreabilidade: nada de dataset, resultado ou checkpoint deve
ser removido sem antes registrar o motivo.

## Estrutura principal

```text
D:\kidney
|-- dataset_inicial/
|-- dataset_aumentado/
|-- src/
|-- engenharia_dataset/
|-- curadoria_web/
|-- models/
|-- results/
|-- docs/
|-- artigo/
|-- config/
|-- scripts/
|-- archive_local/
```

## Papel de cada pasta

| Pasta | Papel | Status |
| --- | --- | --- |
| `dataset_inicial/` | Base original com splits `train`, `val` e `test`, derivada do Open Kidney/kidneyUS. | Manter |
| `dataset_aumentado/fontes/` | Fontes brutas, externas ou complementares usadas para montar bases derivadas. | Manter |
| `dataset_aumentado/pseudo_labels/` | Pseudo-mascaras antigas da primeira expansao. | Manter como historico |
| `dataset_aumentado/expansao_pseudorrotulada/` | Versao intermediaria da base expandida. | Manter como historico metodologico |
| `dataset_aumentado/dataset_geral/` | Base consolidada atual: imagens, mascaras aceitas, manifestos e relatorios. | Fonte principal atual |
| `dataset_aumentado/dataset_geral_cv/` | Folds e holdout derivados do `dataset_geral`; usa hardlinks quando criado com `--link-mode hardlink`. | Manter |
| `dataset_aumentado/dataset_intrarrenal/` | Bases derivadas para cortex, medula e segmentacao intrarrenal; inclui intermediarios, bases supervisionadas e pseudo-expansoes. | Manter |
| `dataset_aumentado/curadoria/` | Manifestos, miniaturas, respostas e artefatos de revisao humana. | Manter |
| `src/` | Codigo principal de segmentacao, treino, avaliacao e ferramentas. | Manter |
| `engenharia_dataset/` | Scripts de montagem, curadoria, conversao e expansao de datasets. | Manter |
| `curadoria_web/` | Interface local de curadoria das mascaras. | Manter |
| `models/` | Checkpoints e metadados de modelos. | Manter; arquivar apenas checkpoints obsoletos depois de decisao manual |
| `results/` | Resultados experimentais, metricas, predicoes e auditorias. | Manter |
| `docs/` | Documentacao tecnica, narrativa metodologica e relatorios. | Manter |
| `artigo/` | Material do artigo. A versao ativa e `artigo/SBBD_2026___Jefferson/`. | Manter |
| `out/` | Saidas temporarias de testes, screenshots e verificacoes locais. | Pode arquivar |
| `archive_local/` | Arquivo local ignorado pelo Git para itens antigos ou temporarios. | Nao versionar |

## Observacoes sobre duplicacao aparente

`dataset_geral_cv/` parece duplicar muitas imagens porque cada fold possui
subpastas de treino, validacao e teste. No ambiente atual, os arquivos foram
criados como hardlinks para `dataset_geral/`, portanto representam a mesma base
em diferentes splits experimentais. A pasta deve ser tratada como derivada, mas
nao como copia solta.

`dataset_inicial/`, `expansao_pseudorrotulada/` e `dataset_geral/` tambem podem
parecer versoes repetidas. Elas correspondem a momentos diferentes da
metodologia:

1. base original com mascaras iniciais;
2. primeira expansao por pseudo-rotulagem;
3. base consolidada atual para treinamento e avaliacao.

As pseudo-mascaras foram geradas por modelos treinados com anotacoes manuais
do Open Kidney Ultrasound Data Set (`kidneyUS`). Elas servem para ampliar o
dataset de forma controlada: o modelo prediz mascaras em imagens adicionais,
os filtros automaticos removem casos fracos e a curadoria humana deve validar
ou corrigir as candidatas antes de usa-las como referencia forte.

Uma pseudo-mascara renal e aceita automaticamente apenas se atingir confianca
media minima de `0.90`, area relativa entre `0.03` e `0.75`, pelo menos `800`
pixels positivos e no maximo `3` componentes conectados. Esses valores ficam
registrados no `manifest.csv`, junto do motivo de rejeicao quando a mascara
nao passa nos filtros.

## Dataset intrarrenal

`dataset_aumentado/dataset_intrarrenal/` possui uma organizacao propria:

- `intermediario/kidneyus_regions/`: intermediario derivado do
  OpenKidney/kidneyUS, com ROIs e mascaras por classe/anotador;
- `supervisionado/`: bases supervisionadas finais, incluindo
  `medulla_annotator_1/`, `cortex_annotator_1/` e
  `regions_multiclass_annotator_1/`;
- `pseudo_expandido/`: bases experimentais com pseudo-mascaras adicionadas ao
  treino, incluindo as expansoes de medula.

As subpastas foram agrupadas por objetivo para reduzir a quantidade de pastas
no primeiro nivel. A estrutura esta detalhada em
`dataset_aumentado/dataset_intrarrenal/README.md`.

## Itens candidatos a arquivo local

Podem ir para `archive_local/` quando nao forem necessarios para rodar o fluxo
principal:

- `out/`, por conter saidas temporarias;
- versoes antigas do artigo, quando houver uma versao ativa definida;
- screenshots de teste, caches de navegador e previews intermediarios;
- checkpoints antigos que nao entram em tabelas, relatorios ou reproducao.

Nao arquivar automaticamente:

- `dataset_inicial/`;
- `dataset_aumentado/dataset_geral/`;
- `dataset_aumentado/dataset_geral_cv/`;
- `dataset_aumentado/dataset_intrarrenal/`;
- `results/`;
- checkpoints referenciados em relatorios ou no artigo.

## Versao ativa do artigo

A versao ativa do artigo fica em:

```text
artigo/SBBD_2026___Jefferson/
```

Versoes antigas ou bases de edicao podem ser guardadas em `archive_local/`,
mantendo o artigo ativo sem pastas paralelas no fluxo principal.
