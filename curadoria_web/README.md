# Curadoria Renal Web

Aplicacao local para revisar as mascaras de rim, cortex e medulla sem incorporar
as imagens em uma planilha. O servidor usa o manifesto visual validado, exibe
os contornos na mesma dimensao da imagem e grava uma avaliacao por revisor.

A tela oferece filtro exclusivo para pseudo-mascaras. Quando uma mascara foi
produzida automaticamente, ela informa o checkpoint utilizado e as metricas
documentadas do modelo (`Dice`, `IoU` e `F1`), deixando claro que esses valores
medem desempenho em teste e nao aprovam automaticamente o caso visualizado.

O caminho atual usa uma etapa intrarrenal multiclasse com DeepLabV3 dentro da
ROI renal. O checkpoint
`intrarenal_deeplab_resnet50_multiclass_annotator1.pth` segmenta `Cortex`,
`Medulla` e `Central Echo Complex` em uma unica inferencia. No teste separado
por paciente, obteve `Dice=0.682169` para `Cortex`, `0.715567` para `Medulla`
e `0.849745` para `Central Echo Complex`; portanto, suas mascaras aparecem
apenas como propostas para revisao.

## Executar

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe .\curadoria_web\app.py
```

Abra `http://127.0.0.1:8765`.

## Inferência em nova imagem

Use `http://127.0.0.1:8765/inferencia.html` (ou o botão **Inferir imagem** na
tela principal) para enviar uma imagem PNG, JPG ou JPEG de até 20 MB. A página
executa localmente a cascata U-Net de cápsula renal e U-Net multiclasse
intrarrenal, gerando máscaras candidatas de rim, córtex, medula e CEC, além do
painel de ecogenicidade. Os resultados são gravados em
`dataset_aumentado/curadoria/respostas/inferencias/` e não entram
automaticamente na fila de curadoria nem constituem diagnóstico clínico.

Essa função requer o ambiente local do projeto com PyTorch e os checkpoints em
`models/`; a imagem Docker de curadoria, destinada apenas à revisão de
máscaras pré-computadas, não inclui os pesos de inferência.

## Dados gerados

As respostas sao persistidas em:

```text
dataset_aumentado/curadoria/respostas/curadoria.sqlite3
```

A ferramenta `Poligono` permite selecionar `Rim`, `Cortex`, `Medulla`,
`Central Echo Complex` ou o `Marcador de anomalia` e substituir, adicionar
ou apagar uma regiao. O marcador de anomalia e uma anotacao visual exploratoria.
O painel direito tambem exibe a diferenca de brilho medio, calculada na imagem
original sem CLAHE, entre cortex e CEC:
quanto mais a direita, mais proximo o brilho cortical esta do CEC. As
mascaras editadas sao salvas separadamente por revisor em:

```text
dataset_aumentado/curadoria/respostas/mascaras_corrigidas/
```

A interface oferece exportacao dos registros para `JSON` e `CSV`. Cada
revisor mantem a propria avaliacao da imagem; uma nova gravacao do mesmo
revisor atualiza sua resposta anterior, preservando os horarios. Cada
salvamento manual tambem preserva uma versao imutavel em `historico/`, para
auditoria. As
exportacoes incluem os caminhos e as operacoes das mascaras corrigidas de
`Rim`, `Cortex`, `Medulla`, `Central Echo Complex` e `Anomalia`.

## Fluxo recomendado

1. Informe o identificador do revisor e seu perfil.
2. Revise os contornos sobrepostos na imagem; use zoom e desligue camadas para
   examinar bordas.
3. Se necessario, escolha `Poligono`, a classe e a acao para corrigir a
   mascara exibida; a proposta original nao e sobrescrita. Para registrar uma
   regiao sugestiva, selecione `Marcador de anomalia` (roxo).
4. Classifique cada mascara como aceita, corrigir, rejeitada ou indisponivel.
5. Consulte o medidor de brilho cortex × CEC como marcador exploratorio.
6. Salve e avance para o proximo caso.

## Compartilhamento

O modo padrao escuta apenas o computador local. Para um piloto em rede
institucional, o servidor pode ser iniciado com `--host 0.0.0.0`, mas ele nao
inclui autenticacao. Antes de usar imagens clinicas com varios revisores,
publique a aplicacao em infraestrutura aprovada com autenticacao e controle
de acesso, e migre a persistencia para um banco central como PostgreSQL.

Evite usar email como armazenamento de respostas: ele dificulta consolidacao,
versionamento e auditoria.

## Guia de scripts e componentes

Esta tabela documenta os arquivos **confirmados no repositorio**. Alguns sao modulos ou recursos web, nao scripts executados diretamente. Os caminhos mencionados representam configuracoes do codigo, nao uma garantia de que checkpoints e datasets estejam versionados.

| Arquivo | O que faz | Objetivo | Entradas e saidas |
| --- | --- | --- | --- |
| `curadoria_web/app.py` | Servidor HTTP da curadoria: apresenta casos, imagens, mascaras, ferramentas de revisao e endpoints para exportacao. | Organizar revisao humana auditavel. | Le o manifesto visual e imagens; grava revisoes em `dataset_aumentado/curadoria/respostas/curadoria.sqlite3` e correcoes separadas. |
| `curadoria_web/inference.py` | Cascata de inferencia: segmenta a capsula, recorta a ROI renal, prediz cortex, medula e CEC e invoca medidas de ecogenicidade. | Gerar propostas de segmentacao de novas imagens, sem diagnostico clinico. | PNG/JPEG e os modelos `models/kidneyus_capsule_unet.pth` e `models/intrarenal_unet_multiclass_annotator1.pth`; grava imagens e mascaras em `dataset_aumentado/curadoria/respostas/inferencias/`. |
| `curadoria_web/static/app.js` | Logica do navegador: navegacao, zoom, camadas, desenho de correcoes e envio de revisoes ao servidor. | Oferecer ferramentas interativas para inspeção e curadoria. | Endpoints da aplicacao; altera o estado exibido e envia dados ao servidor. |
| `curadoria_web/static/inferencia.html` | Pagina de upload e visualizacao da inferencia. | Permitir testar uma imagem sem incorpora-la automaticamente ao treino. | Usa a API de `app.py` e apresenta camadas e resultados no navegador. |
| `src/segmentation/build_dataset_geral.py` | Organiza fontes de imagens e produz pseudomascaras candidatas com checkpoint configuravel; registra manifesto. | Preparar dados com proveniencia para revisao e possivel expansao supervisionada. | Imagens e modelo; padrao do checkpoint: `models/kidneyus_capsule_dedup_unet.pth`; produz `manifest.csv` no diretorio de saida definido. |
| `src/segmentation/experiments/train_unet.py` | Treina U-Net usando a infraestrutura compartilhada de treinamento. | Aprender segmentacao renal e avaliar o modelo. | Conjuntos particionados e argumentos de treino; gera checkpoint e metricas conforme configuracao. |
| `src/segmentation/experiments/train_inner_deeplab.py` | Treina DeepLabV3 multiclasse na ROI renal. | Identificar cortex, medula e complexo ecogenico central. | Dataset intrarrenal e parametros; checkpoint padrao `models/intrarenal_deeplab_resnet50_multiclass_annotator1.pth`, metadados e resultados. |
| `artigo/artigo_base/compilar_artigo.ps1` | Executa LaTeX, BibTeX e repeticoes do LaTeX; verifica citacoes e rotulos das figuras. | Gerar um artigo com bibliografia e referencias cruzadas resolvidas. | `main.tex`, `.bib`, figuras e distribuicao LaTeX; produz `main.pdf`. |

### Sequencia de utilizacao

1. **Preparacao:** `build_dataset_geral.py` prepara o acervo e diferencia pseudomascaras de referencia manual.
2. **Treinamento:** os scripts de `src/segmentation/experiments/` ajustam e avaliam redes e salvam checkpoints.
3. **Inferencia:** `inference.py`, chamado pelo servidor, aplica modelos previamente treinados a uma imagem nova.
4. **Curadoria:** `app.py` com a interface em `static/` permite aceitar, corrigir ou rejeitar mascaras e armazenar o historico.
5. **Artigo:** `compilar_artigo.ps1` recompila o texto, atualiza as referencias e valida a saida.

### Comandos de consulta e execucao (Windows, a partir da raiz)

```powershell
.\.venv\Scripts\python.exe .\curadoria_web\app.py
.\.venv\Scripts\python.exe .\src\segmentation\build_dataset_geral.py --help
.\.venv\Scripts\python.exe .\src\segmentation\experiments\train_unet.py --help
.\.venv\Scripts\python.exe .\src\segmentation\experiments\train_inner_deeplab.py --help
powershell -NoProfile -ExecutionPolicy Bypass -File .\artigo\artigo_base\compilar_artigo.ps1
```

Antes de preparar dados ou treinar modelos, confira argumentos, manifestos, particoes e checkpoints. Nem todos os scripts utilizam o mesmo checkpoint como padrao.

### Geracao das quatro figuras cientificas

O arquivo **`scripts/figuras/gerar_figuras.py`** implementa cinco subcomandos para produzir PNGs de alta resolucao (300 DPI por padrao) a partir de **imagens e mascaras ja existentes**. Este utilitario nao executa inferencia, nao carrega checkpoints e nao altera os dados originais. Para produzir mascaras novas, use antes a inferencia disponibilizada pela Curadoria Web.

| Modo | Objetivo | Entradas obrigatorias |
| --- | --- | --- |
| `revisao` | Mostrar original, mascara automatica, revisada e pixels adicionados/removidos. | `--image`, `--automatic`, `--reviewed`, `--output`. Use **somente revisoes realmente registradas** no sistema. |
| `multiclasse` | Mostrar original, composicao de rim/cortex/medula/CEC e sobreposicao. | `--image`, `--rim`, `--cortex`, `--medulla`, `--cec`, `--output`. |
| `multiclasse_detalhada` | Mostrar original, cada mascara individual sobre a imagem e sobreposicao conjunta. | `--image`, `--rim`, `--cortex`, `--medulla`, `--cec`, `--output`. |
| `rois` | Mostrar as regioes selecionadas de cortex e CEC e calcular medias e a razao cortex/CEC. | `--image`, `--cortex`, `--cec`, `--output`. Os argumentos representam as **ROIs escolhidas**, nao necessariamente a mascara completa de cada classe. |
| `casos` | Apresentar dois exemplos lado a lado, cada um com imagem, referencia manual e predicao. | `--good-image`, `--good-reference`, `--good-prediction`, `--hard-image`, `--hard-reference`, `--hard-prediction`, `--output`. |

Exemplos de uso (substitua os caminhos entre aspas pelos arquivos reais de um mesmo caso):

```powershell
.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras.py --help

.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras.py revisao --image "imagem.png" --automatic "mascara_automatica.png" --reviewed "mascara_revisada.png" --output "artigo/artigo_base/figures/curadoria_antes_depois_revisao.png"

.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras.py multiclasse --image "imagem.png" --rim "rim.png" --cortex "cortex.png" --medulla "medulla.png" --cec "cec.png" --output "artigo/artigo_base/figures/segmentacao_anatomica_multiclasse.png"

.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras.py multiclasse_detalhada --image "imagem.png" --rim "rim.png" --cortex "cortex.png" --medulla "medulla.png" --cec "cec.png" --output "artigo/artigo_base/figures/segmentacao_anatomica_multiclasse_detalhada.png"

.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras.py rois --image "imagem.png" --cortex "roi_cortex.png" --cec "roi_cec.png" --output "artigo/artigo_base/figures/rois_ecogenicidade_exemplo.png"

.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras.py casos --good-image "imagem_favoravel.png" --good-reference "gt_favoravel.png" --good-prediction "pred_favoravel.png" --hard-image "imagem_dificil.png" --hard-reference "gt_dificil.png" --hard-prediction "pred_dificil.png" --output "artigo/artigo_base/figures/casos_sucesso_falha_segmentacao.png"
```

**Requisitos:** Python com `numpy`, `opencv-python` e `matplotlib`; todas as mascaras devem estar alinhadas a imagem original e ter exatamente a mesma resolucao. O programa interrompe a execucao quando recebe um caminho inexistente ou uma mascara em resolucao diferente.

**Interpretacao:** a razao de brilho calculada em `rois` usa intensidade B-mode original e nao constitui diagnostico. As mascaras de comparacao manual precisam pertencer ao mesmo caso e revisao documentada; o script nao autentica a procedencia de arquivos externos. Na figura de diferencas, verde significa adicao e vermelho significa remocao. Recomenda-se legenda LaTeX com identificacao de cada painel e `Fonte: elaboracao propria (resultados do sistema desenvolvido).`

### Selecao automatica de casos reais (manifesto + SQLite)

O auxiliar `scripts/figuras/gerar_figuras_curadoria.py` consulta o manifesto visual, os resultados dos segmentadores e o banco SQLite **sem editar os dados**. O comando `listar` identifica IDs e revisoes existentes; `revisao` exige o par de mascara anterior e mascara efetivamente corrigida; `multiclasse` busca as quatro camadas disponiveis. O gerador exporta PNG e um arquivo `.proveniencia.json` com caminhos usados, ID, camada, modelo e revisor (quando aplicavel).

Execute na raiz do repositorio:

```powershell
.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras_curadoria.py listar
.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras_curadoria.py revisao --image-id "ID_REAL" --reviewer "REVISOR" --layer cortex
.\.venv\Scripts\python.exe .\scripts\figuras\gerar_figuras_curadoria.py multiclasse --image-id "ID_REAL" --model unet
```

Diretorio padrao: `artigo/artigo_base/figures/geradas/`. Argumentos opcionais: `--manifest`, `--database`, `--output-dir` e `--model` (`unet` ou `deeplab`). Nao ha selecao arbitraria de "melhor caso": a revisao e escolhida pelo ID/revisor real. A aprovacao registrada e incluida na proveniencia e nao e interpretada automaticamente como validacao clinica.

**Limites:** a versao automatica cobre `revisao` e `multiclasse`. As figuras `rois` e `casos` continuam acessiveis pelo gerador base com mascaras explicitamente fornecidas; nenhuma selecao automatica de ROI ou referencial manual foi suposta. Um caminho ausente ou uma mascara inconsistente interrompe a geracao em vez de fabricar resultado.
