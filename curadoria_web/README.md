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

### Geradores de figuras planejados (ainda nao implementados)

| Script proposto | Figura e finalidade | Requisitos |
| --- | --- | --- |
| `gerar_figura_revisao_humana.py` | Original, mascara automatica, corrigida e diferencas; demonstrar a revisao humana real. | Caso com correcao persistida e versoes correspondentes. |
| `gerar_figura_multiclasse.py` | Capsula, cortex, medula, CEC e overlays; explicar a cascata anatomica. | Modelos, imagem e mascaras geradas. |
| `gerar_figura_rois_ecogenicidade.py` | ROIs e medidas cortex/CEC; vincular a imagem as equacoes. | Imagem B-mode original, mascaras e ROIs validas. |
| `gerar_figura_sucesso_falha.py` | Casos favoravel e dificil; demonstrar limites da segmentacao. | Imagens, predicoes e referencias manuais comparaveis. |

**Nota cientifica:** nao apresentar mascaras pseudorrotuladas como referencia manual, nem simular correcao humana para criar figura de resultado. As figuras devem conter descricao e fonte dos dados.
