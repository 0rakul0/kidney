# Reuniao de alinhamento: curadoria e pseudo-rotulacao human-in-the-loop

## Objetivo da reuniao

Definir um protocolo defensavel para revisar as mascaras renais e
intrarrenais geradas automaticamente, antes que sejam apresentadas como
anotacoes confiaveis ou utilizadas na etapa de segmentacao de `Medulla`.

Reuniao sugerida: **quinta-feira, 28/05/2026, das 17:00 as 17:45**.
Esse horario atende a disponibilidade informada por Amaro, Gabriel,
Jefferson e Felipe, considerando que Felipe precisa encerrar ate 18:15.

Participantes importantes:

- Felipe Henriques;
- Jefferson;
- Amaro Azevedo de Lima;
- Gabriel Matos Araujo;
- Nordeval, como potencial especialista na curadoria;
- Jorge Henriques Jr., pelo possivel uso futuro dos dados do HUPE e pela
  necessidade de avaliacao especializada.

## Situacao atual do projeto

O pipeline atual possui duas tarefas que precisam ser separadas:

```text
Etapa 1: imagem -> segmentacao do rim -> mascara/ROI renal
Etapa 2: ROI renal -> segmentacao de Medulla -> marcadores intrarrenais
```

Na etapa 1, o projeto possui `4.853` imagens com caminho de mascara renal:

| Origem da mascara renal | Imagens | Interpretacao atual |
| --- | ---: | --- |
| Mascara existente | 1.001 | Fonte inicial mais segura para iniciar revisao |
| Mascara gerada e aceita automaticamente | 3.852 | Pseudomascaras que precisam de auditoria humana |
| Total | 4.853 | Base disponivel para curadoria |

Na etapa 2, ja foi feito um experimento de expansao de `Medulla`:

| Grupo | Quantidade | Interpretacao atual |
| --- | ---: | --- |
| Candidatas iniciais para revisao com ROI renal existente | 457 | Priorizadas por consenso entre modelos |
| Pseudomascaras incorporadas ao treino experimental v1 | 342 | Resultado experimental, nao anotacao clinicamente validada |
| Nova fila de revisao sobre ROI renal existente | 599 | Inclui 162 novas candidatas |
| Candidatas filtradas sobre ROI renal gerada pelo modelo | 1.111 | Exigem revisar rim e medula antes de qualquer uso |

Assim, o trabalho ja mostra que e possivel gerar e priorizar candidatas, mas
nao demonstra que as pseudomascaras sejam equivalentes a rotulos revisados.

## Criticas e sugestoes recebidas

### 1. Curadoria das mascaras renais e indispensavel

O primeiro modelo foi treinado com imagens anotadas e usado para gerar
pseudomascaras em milhares de imagens adicionais. A critica central e que o
limiar numerico de confianca do modelo nao garante que o rim esteja
anatomicamente bem delimitado.

Consequencia: as `3.852` mascaras renais geradas automaticamente devem ser
tratadas como candidatas pendentes de revisao, nao como `ground truth`.

### 2. A segmentacao da medula depende da qualidade da ROI renal

A etapa de `Medulla` e coerente apenas se a mascara do rim que delimita a ROI
estiver correta. Uma mascara renal incorreta pode produzir uma pseudomascara
de medula aparentemente plausivel em uma regiao que nao corresponde ao rim.

Consequencia: candidatas de medula baseadas em ROI renal gerada devem ter
auditoria em dois niveis:

1. confirmar se a mascara externa realmente delimita o rim;
2. somente entao avaliar se a mascara interna de `Medulla` e aceitavel.

### 3. Confianca do modelo pode priorizar, mas nao aprovar sozinha

Foi sugerido usar confianca, concordancia entre modelos e filtros geometricos
para selecionar os casos mais promissores. Essa estrategia pode reduzir o
esforco de revisao, mas nao substitui a avaliacao humana.

Consequencia: os filtros automaticos devem ser descritos como mecanismo de
priorizacao de fila, nao como aprovacao final da anotacao.

### 4. O objetivo primario e construir um dataset anotado confiavel

Houve duas perspectivas complementares:

- avaliar iterativamente se novas pseudomascaras melhoram o modelo;
- priorizar a conclusao de um conjunto anotado confiavel e, depois, fazer o
  split definitivo e comparar modelos.

O ponto de convergencia e que a avaliacao final do desempenho deve usar rotulos
humanos confiaveis e que a curadoria continua obrigatoria, mesmo que exista um
estudo adicional de reducao de carga de especialista.

### 5. O HUPE pode ampliar o estudo em uma segunda fase

Foi sugerido incluir imagens do HUPE, com Nordeval apoiando a avaliacao das
anotacoes e possiveis variaveis clinicas por paciente. Essa extensao pode
fortalecer a relevancia clinica do trabalho, mas adiciona uma nova coorte,
novas regras de privacidade e possiveis desfechos clinicos.

Consequencia: recomenda-se tratar HUPE e dados clinicos como segunda fase,
sem alterar o protocolo principal antes de finalizar a curadoria da base atual.

## Proposta de protocolo para deliberacao

### Principio metodologico

Usar o modelo para acelerar a anotacao, nunca para substituir silenciosamente
a revisao humana. Toda mascara aceita para compor um dataset curado deve
possuir status de revisao e identidade do tipo de revisor.

### Fase A: congelar os conjuntos de origem

Antes da curadoria, registrar:

- as imagens originalmente anotadas;
- as mascaras renais geradas automaticamente;
- as pseudomascaras de `Medulla` ja produzidas;
- a versao do modelo e os filtros usados em cada geracao;
- a origem da imagem, separando base atual e, futuramente, HUPE.

O experimento existente com `342` pseudomascaras deve permanecer documentado
como expansao automatica experimental, nao como conjunto curado.

### Fase B: definir formulario de revisao

Para cada imagem da etapa renal, a revisao deve registrar no minimo:

| Campo | Opcoes sugeridas |
| --- | --- |
| Imagem contem rim avaliavel? | sim / nao / duvidoso |
| Mascara renal esta correta? | aceitar / corrigir / rejeitar |
| Qualidade da imagem | adequada / limitada / inadequada |
| Revisor | identificador do revisor |
| Nivel de expertise | especialista / nao especialista treinado |
| Observacao | texto curto opcional |

Para `Medulla`, somente imagens com ROI renal aceita devem seguir para:

| Campo | Opcoes sugeridas |
| --- | --- |
| Medulla avaliavel? | sim / nao / duvidoso |
| Mascara de Medulla | aceitar / corrigir / rejeitar |
| Dificuldade do caso | simples / intermediario / dificil |

### Fase C: lote piloto para medir concordancia

Selecionar um lote piloto de aproximadamente `100` imagens, contendo casos
faceis, intermediarios e dificeis, para revisao independente por:

- um especialista;
- um nao especialista treinado, se a hipotese de reducao de custo for estudada.

Medir concordancia de decisao (`aceitar`, `corrigir`, `rejeitar`) e, para
mascaras corrigidas, Dice/IoU entre revisores. Sem esse piloto, nao e possivel
sustentar a hipotese de que um nao especialista com apoio do modelo substitui
parte relevante do trabalho especializado.

### Fase D: curadoria em lotes

Caso o piloto seja aceitavel, revisar as candidatas em lotes de ate `500`
imagens, como sugerido na discussao:

```text
modelo gera/prioriza candidatas
-> humano aceita, corrige ou rejeita
-> casos aceitos/corrigidos entram na base curada
-> casos rejeitados retornam para anotacao manual ou nova avaliacao
```

Ordem conservadora de revisao:

1. mascaras renais existentes e pseudomascaras de `Medulla` ja priorizadas;
2. mascaras renais geradas automaticamente com maior prioridade de revisao;
3. pseudomascaras de `Medulla` cuja ROI renal tambem foi gerada;
4. casos rejeitados ou duvidosos para anotacao manual especializada.

### Fase E: treinamento durante a curadoria, sem contaminar avaliacao final

E possivel treinar modelos intermediarios apos cada lote aprovado para avaliar
se a curadoria assistida melhora a selecao dos lotes seguintes. Contudo, esse
resultado nao deve ser confundido com a avaliacao final do modelo.

Regras recomendadas:

- manter um pequeno conjunto sentinela revisado por especialista para
  monitoramento interno durante o processo;
- nao usar esse conjunto para escolher o resultado final do artigo;
- apos concluir a anotacao definida no protocolo, criar split
  treino/validacao/teste por paciente, quando houver identificador;
- realizar a comparacao final somente sobre teste humano congelado.

Essa solucao atende as duas preocupacoes levantadas: acompanha se o modelo
esta evoluindo durante o processo e preserva um dataset final confiavel.

## Comparacoes cientificas possiveis

Se a curadoria for registrada de forma rastreavel, o trabalho pode comparar:

| Experimento | Pergunta |
| --- | --- |
| Modelo treinado apenas com rotulos manuais iniciais | Qual e o baseline? |
| Modelo com pseudomascaras filtradas automaticamente | O filtro automatico ajuda sem validacao humana? |
| Modelo com pseudomascaras aceitas/corrigidas por humano | Qual e o ganho do human-in-the-loop? |
| Decisoes de nao especialista versus especialista no piloto | E possivel reduzir demanda do especialista com seguranca? |
| Base atual versus HUPE, em fase posterior | O metodo generaliza para outra origem? |

## Pontos que precisam de decisao na reuniao

1. A prioridade imediata e curar mascara renal, `Medulla`, ou ambas em
   sequencia?
2. Quem fara o papel de especialista de referencia: Nordeval, Jorge ou ambos?
3. O lote piloto tera `100` imagens ou outra quantidade viavel?
4. O nao especialista participara do piloto para medir concordancia?
5. Quais criterios tornam uma mascara `aceita`, `corrigida` ou `rejeitada`?
6. Os experimentos automaticos ja realizados entram no artigo apenas como
   resultado exploratorio pendente de auditoria?
7. Qual modificacao minima e necessaria no artigo do SBBD antes da submissao?
8. Quando e adequado iniciar uma fase HUPE com variaveis clinicas?

## Ajustes necessarios no artigo do SBBD

Com base na critica recebida, a redacao nao deve sugerir que a confianca de
`90%` validou automaticamente as pseudomascaras. A formulacao mais defensavel
e:

> As pseudomascaras foram selecionadas automaticamente por criterios
> operacionais para experimentacao e priorizacao de curadoria. Sua incorporacao
> em uma base anotada confiavel depende de auditoria humana.

Tambem e importante separar claramente:

- resultado experimental obtido ao treinar com pseudomascaras;
- protocolo futuro de curadoria human-in-the-loop;
- eventual validacao clinica com dados do HUPE.

## Pauta sugerida para 45 minutos

| Tempo | Item | Resultado esperado |
| ---: | --- | --- |
| 5 min | Apresentar pipeline atual e numeros da base | Contexto comum |
| 10 min | Confirmar risco de usar pseudomascaras sem auditoria | Consenso metodologico |
| 10 min | Definir protocolo piloto e papel do especialista | Desenho da curadoria |
| 10 min | Definir como alterar o artigo do SBBD | Lista de ajustes |
| 5 min | Discutir HUPE e informacoes clinicas como segunda fase | Escopo futuro |
| 5 min | Nomear responsaveis e datas | Plano executavel |

## Mensagem pronta para agendamento

```text
Pessoal, para alinharmos a curadoria das pseudomascaras, o protocolo
human-in-the-loop e os ajustes do artigo do SBBD, proponho uma reuniao na
quinta-feira (28/05), das 17h as 17h45. Esse horario permite a participacao
do Felipe antes do compromisso das 18h15.

Seria importante contarmos tambem com o Nordeval e, se possivel, com o Jorge,
pois precisamos definir o papel do especialista na curadoria e discutir, em
uma etapa posterior, o uso das imagens e informacoes clinicas do HUPE.

Pauta: (1) validar a necessidade de revisao humana das mascaras renais e de
Medulla; (2) definir um lote piloto e criterios de aceitar/corrigir/rejeitar;
(3) decidir os ajustes imediatos no artigo do SBBD; e (4) organizar a possivel
fase HUPE. Posso organizar o link da reuniao assim que confirmarem.
```

## Encaminhamentos recomendados

| Responsavel a confirmar | Acao |
| --- | --- |
| Jefferson / Felipe | Enviar convite para quinta-feira, 28/05/2026, 17:00-17:45 |
| Nordeval / Jorge | Confirmar disponibilidade e papel na revisao especializada |
| Amaro / Gabriel | Levar proposta de protocolo human-in-the-loop e criterios de avaliacao |
| Jefferson | Apresentar os numeros atuais, os experimentos automaticos e os pontos do artigo a revisar |
| Grupo | Aprovar lote piloto, formulario de curadoria e regra de uso das pseudomascaras |
