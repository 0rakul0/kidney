# Organizacao dos datasets e planilha de curadoria

## Estrutura na raiz

A raiz do projeto mantem somente duas pastas de dados:

```text
dataset_inicial/
dataset_aumentado/
```

`dataset_inicial/` contem a divisao original em `train`, `val` e `test`.

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

O subconjunto MONAI bruto/processado esta em
`dataset_aumentado/fontes/external_data/`. As imagens consolidadas para uso
do pipeline permanecem em `dataset_aumentado/dataset_geral/`, junto do
manifesto que registra a origem de cada imagem.

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

A aba `Curadoria` possui exatamente as colunas:

| Coluna | Conteudo |
| --- | --- |
| `imagem` | Caminho da imagem consolidada no `dataset_geral` |
| `anot1` | Mascara renal disponivel para avaliacao |
| `anot2` | Mascara intrarrenal disponivel; atualmente `Medulla` |
| `cl1` | Revisao da mascara renal: `1` correta, `0` incorreta |
| `cl2` | Revisao da mascara de `Medulla`: `1` correta, `0` incorreta |
| `fibrose` | `1` fibrose confirmada, `0` saudavel, apenas com referencia clinica |

Os campos `cl1`, `cl2` e `fibrose` sao deixados vazios na geracao inicial:
um valor `0` ja significa avaliacao concluida com resultado negativo.

## Ressalva sobre Cortex e fibrose

No estado atual, `anot2` nao e uma mascara consolidada de `Cortex`; ela
representa `Medulla`, priorizando anotacao manual quando disponivel e usando
pseudomascara candidata nos demais casos. Caso a curadoria tambem deva
abranger `Cortex`, recomenda-se adicionar uma coluna separada para essa
estrutura, sem reutilizar a mesma anotacao.

O campo `fibrose` nao deve ser preenchido por inferencia visual automatica
nem pela qualidade da mascara. Ele exige confirmacao clinica ou avaliacao
especializada definida no protocolo do estudo.
