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

## Dados gerados

As respostas sao persistidas em:

```text
dataset_aumentado/curadoria/respostas/curadoria.sqlite3
```

A ferramenta `Poligono` permite selecionar `Rim`, `Cortex`, `Medulla` ou
`Central Echo Complex` e substituir, adicionar ou apagar uma regiao. As
mascaras editadas sao salvas separadamente por revisor em:

```text
dataset_aumentado/curadoria/respostas/mascaras_corrigidas/
```

A interface oferece exportacao dos registros para `JSON` e `CSV`. Cada
revisor mantem a propria avaliacao da imagem; uma nova gravacao do mesmo
revisor atualiza sua resposta anterior, preservando os horarios. As
exportacoes incluem os caminhos e as operacoes das mascaras corrigidas de
`Rim`, `Cortex`, `Medulla` e `Central Echo Complex`.

## Fluxo recomendado

1. Informe o identificador do revisor e seu perfil.
2. Revise os contornos sobrepostos na imagem; use zoom e desligue camadas para
   examinar bordas.
3. Se necessario, escolha `Poligono`, a classe e a acao para corrigir a
   mascara exibida; a proposta original nao e sobrescrita.
4. Classifique cada mascara como aceita, corrigir, rejeitada ou indisponivel.
5. Registre fibrose somente quando existir referencia clinica rastreavel.
6. Salve e avance para o proximo caso.

## Compartilhamento

O modo padrao escuta apenas o computador local. Para um piloto em rede
institucional, o servidor pode ser iniciado com `--host 0.0.0.0`, mas ele nao
inclui autenticacao. Antes de usar imagens clinicas com varios revisores,
publique a aplicacao em infraestrutura aprovada com autenticacao e controle
de acesso, e migre a persistencia para um banco central como PostgreSQL.

Evite usar email como armazenamento de respostas: ele dificulta consolidacao,
versionamento e auditoria.
