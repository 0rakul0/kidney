# Organizacao dos datasets e planilha de curadoria

## Estrutura na raiz

A raiz do projeto mantem uma pasta principal de dados:

```text
dataset_aumentado/
```

`dataset_aumentado/` concentra fontes adicionais, pseudorrotulos, conjuntos
derivados e artefatos preparados para curadoria:

```text
dataset_aumentado/
|-- fontes/
|   |-- dataset_loader/
|   |-- external_data/
|   |-- identificada/
|   |-- kidneyUS_images_25_june_2025/
|   |-- nao_identificada/
|   `-- reference_masks/
|-- pseudo_labels/
|-- expansao_pseudorrotulada/
|-- dataset_geral/
|-- dataset_geral_cv/
|-- dataset_intrarrenal/
`-- curadoria/
```

O kidneyUS e a fonte canonica do novo ciclo. A base supervisionada para
segmentacao externa do rim e
`dataset_aumentado/dataset_intrarrenal/supervisionado/capsule_annotator_1/`,
criada a partir da classe `Capsule`.

O subconjunto MONAI bruto/processado esta em
`dataset_aumentado/fontes/external_data/`. As imagens consolidadas para uso
do pipeline podem ser montadas em `dataset_aumentado/dataset_geral/`, junto do
manifesto que registra a origem de cada imagem.

## Organizacao do dataset intrarrenal

`dataset_aumentado/dataset_intrarrenal/` agrupa bases derivadas para a etapa
intrarrenal. Ela nao deve ser lida como uma unica pasta plana de imagens.

Estrutura conceitual:

```text
dataset_intrarrenal/
|-- intermediario/
|   `-- kidneyus_regions/                       # OpenKidney/kidneyUS
|-- supervisionado/
|   |-- capsule_annotator_1/                    # Capsule binario
|   |-- medulla_annotator_1/                    # Medulla binario
|   |-- cortex_annotator_1/                     # Cortex binario
|   `-- regions_multiclass_annotator_1/         # Cortex, Medulla e CEC
`-- pseudo_expandido/
    |-- medulla_expanded_consensus_v1/          # pseudo-expansao experimental
    |-- medulla_expanded_consensus_v2/          # pseudo-expansao experimental
    `-- medulla_generated_kidney_consensus_v1/  # pseudo-expansao com ROI gerada
```

`intermediario/kidneyus_regions/` pode conter varias representacoes da mesma
imagem, pois armazena ROIs, mascaras por classe e dados por anotador. Ja as
bases finais em `supervisionado/` sao filtradas por classe, por anotador e por
split.

Contagens principais:

| Base | Total | Train | Val | Test |
| --- | ---: | ---: | ---: | ---: |
| `supervisionado/capsule_annotator_1/` | gerado por `create_capsule_splits.py` | - | - | - |
| `supervisionado/medulla_annotator_1/` | 336 | 236 | 50 | 50 |
| `supervisionado/cortex_annotator_1/` | 336 | 236 | 50 | 50 |
| `supervisionado/regions_multiclass_annotator_1/` | 335 | 235 | 50 | 50 |

As bases expandidas de medula adicionam pseudo-mascaras apenas ao treino e
mantem validacao/teste manuais:

| Base expandida | Treino total | Pseudo-mascaras adicionadas |
| --- | ---: | ---: |
| `pseudo_expandido/medulla_expanded_consensus_v1/` | 578 | 342 |
| `pseudo_expandido/medulla_expanded_consensus_v2/` | 710 | 474 |
| `pseudo_expandido/medulla_generated_kidney_consensus_v1/` | 1347 | 1111 |

Um guia local mais detalhado tambem foi salvo em:

```text
dataset_aumentado/dataset_intrarrenal/README.md
```

## Proveniencia das pseudo-mascaras

As pseudo-mascaras usadas para ampliar o dataset foram geradas por modelos de
segmentacao treinados com anotacoes manuais do Open Kidney Ultrasound Data Set
(`kidneyUS`).

Na etapa renal externa, o modelo aprendeu a segmentar o rim/capsula a partir
das mascaras manuais disponiveis no conjunto inicial derivado do
OpenKidney/kidneyUS. Em seguida, esse modelo foi aplicado sobre imagens sem
mascara para produzir novas mascaras candidatas. Apenas as predicoes que
passaram por limiar de confianca e filtros geometricos entraram no
`dataset_geral` como pseudo-mascaras aceitas.

Na etapa intrarrenal, as anotacoes multiclasse do OpenKidney/kidneyUS
(`Capsule`, `Cortex`, `Medulla` e `Central Echo Complex`) foram usadas para
treinar segmentadores internos. As predicoes desses modelos sobre imagens do
`dataset_geral` sao pseudo-mascaras candidatas de estruturas internas e entram
na fila de curadoria humana.

Portanto, a cadeia metodologica e:

```text
OpenKidney/kidneyUS com anotacoes manuais
-> treinamento dos modelos de segmentacao
-> geracao de pseudo-mascaras em imagens adicionais
-> filtros automaticos de qualidade
-> curadoria humana
-> aumento progressivo do dataset
```

As pseudo-mascaras nao substituem as anotacoes manuais originais. Elas sao um
mecanismo de expansao controlada da base, criado para aumentar o volume de
dados disponivel e permitir retreinamento com maior confiabilidade apos revisao.

## Criterios de aceite automatico

Uma mascara gerada por modelo so e aceita automaticamente quando passa por
todos os criterios de qualidade definidos no script
`src/segmentation/build_dataset_geral.py` ou no script de geracao de mascaras
faltantes. Os valores atuais sao:

| Criterio | Valor |
| --- | ---: |
| Confianca media dos pixels previstos como rim | `>= 0.90` |
| Area minima da mascara em relacao a imagem | `>= 0.03` |
| Area maxima da mascara em relacao a imagem | `<= 0.75` |
| Pixels positivos minimos | `>= 800` |
| Numero maximo de componentes conectados | `<= 3` |

O aceite automatico segue a seguinte sequencia:

```text
1. a imagem sem mascara e processada pelo segmentador;
2. o mapa de probabilidade e convertido em mascara binaria;
3. a mascara e redimensionada para o tamanho original da imagem;
4. sao calculados confianca media, area relativa, pixels positivos e componentes;
5. a mascara so e salva se todos os criterios forem satisfeitos;
6. o resultado e registrado no manifest.csv e nos relatorios do dataset_geral.
```

Os motivos de rejeicao possiveis incluem:

- `low_confidence`;
- `too_few_foreground_pixels`;
- `area_ratio_too_low`;
- `area_ratio_too_high`;
- `too_many_components`.

Esses filtros nao substituem avaliacao humana. Eles apenas impedem que
predicoes evidentemente fracas entrem automaticamente na base expandida.

## Planilha de curadoria

A fila de revisao e gerada por:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\build_curation_manifest.py
```

Arquivo tabular base:

```text
dataset_aumentado/curadoria/curadoria_mascaras.csv
```

Planilha formatada:

```text
dataset_aumentado/curadoria/outputs/curadoria_20260527/planilha_curadoria_mascaras.xlsx
```

Para avaliacao direta, foi criada uma planilha visual completa com todas as
imagens e com sobreposicoes das mascaras sobre a ultrassonografia:

```text
dataset_aumentado/curadoria/outputs/curadoria_visual_completa/curadoria_visual_completa.xlsx
```

A planilha visual possui as `5.994` imagens do manifesto, contorno vermelho
para revisao da mascara renal e contorno amarelo para revisao de `Medulla`,
quando disponivel. Os caminhos dos arquivos permanecem na aba `Referencias`,
enquanto a aba `Curadoria Visual` e destinada ao preenchimento humano.
O arquivo completo tem aproximadamente `84 MB`, devido as miniaturas
embutidas para compartilhamento e avaliacao direta.

## Abas da planilha visual

A aba `Curadoria Visual` exibe cada imagem junto aos contornos produzidos para
revisao:

| Coluna | Conteudo |
| --- | --- |
| `image_id` | Identificador unico da imagem |
| `origem` | Base de origem da imagem |
| `patient_id` | Identificador anonimo, quando disponivel |
| `imagem` | Miniatura da ultrassonografia |
| `rim_contorno_vermelho` | Mascara renal sobreposta para revisao |
| `origem_mascara_rim` | `manual/existente`, `pseudo` ou `sem_mascara` |
| `status_rim` | `pendente`, `aceita`, `corrigir`, `rejeitada` ou `nao_disponivel` |
| `cortex` | Espaco reservado para futura mascara de `Cortex` |
| `status_cortex` | Situacao da anotacao de `Cortex` |
| `medulla_contorno_amarelo` | Mascara de `Medulla` sobreposta para revisao |
| `status_medulla` | Situacao da anotacao de `Medulla` |
| `fibrose` | `0` ou `1`, somente quando houver referencia clinica |
| `fonte_fibrose` | Fonte do rotulo clinico, quando existente |
| `revisor` | Identificador do avaliador |
| `tipo_revisor` | `especialista` ou `nao_especialista` |
| `observacao` | Registro livre da curadoria |

A aba `Referencias` preserva os caminhos dos arquivos originais e das
mascaras. A aba `Instrucoes` orienta o preenchimento.

Os campos de decisao humana iniciam como `pendente` ou vazios. Um valor
`0` em `fibrose` significa paciente/imagem saudavel com referencia clinica,
nao ausencia de avaliacao.

## Ressalva sobre Cortex e fibrose

No estado atual, `anot2` nao e uma mascara consolidada de `Cortex`; ela
representa `Medulla`, priorizando anotacao manual quando disponivel e usando
pseudomascara candidata nos demais casos. Caso a curadoria tambem deva
abranger `Cortex`, recomenda-se adicionar uma coluna separada para essa
estrutura, sem reutilizar a mesma anotacao.

O campo `fibrose` nao deve ser preenchido por inferencia visual automatica
nem pela qualidade da mascara. Ele exige confirmacao clinica ou avaliacao
especializada definida no protocolo do estudo.
