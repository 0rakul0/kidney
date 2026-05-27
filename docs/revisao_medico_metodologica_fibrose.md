# Revisao medico-metodologica: fibrose renal e visao computacional

## Motivo do ajuste

A critica recebida e pertinente: o projeto esta tecnicamente detalhado, mas a
narrativa medica precisa separar com clareza tres conceitos diferentes:

1. segmentacao anatomica do rim e de compartimentos intrarrenais;
2. caracterizacao ultrassonografica de ecogenicidade e textura;
3. estimativa de fibrose renal, cuja referencia clinica e histopatologica.

O trabalho ja produz resultados validos para os dois primeiros pontos. O
terceiro nao deve ser afirmado sem rotulos clinicos ou histologicos adequados.

## O que significa fibrose renal na literatura

Nos trabalhos clinicamente orientados, o desfecho mais usado nao e "opacidade
da medula", mas `interstitial fibrosis and tubular atrophy` (`IFTA`). IFTA e
avaliada em biopsia renal, geralmente pela porcentagem de cortex acometido.

Consequencias para o artigo:

- sem biopsia ou laudo clinico correspondente, nao ha rotulo de fibrose;
- ecogenicidade e textura no ultrassom sao marcadores indiretos e
  inespecificos;
- a saida atual deve ser denominada caracterizacao parenquimatosa ou
  marcador ultrassonografico exploratorio, nao diagnostico de fibrose.

## Evidencias relevantes encontradas

| Trabalho | Dados e alvo | O que ensina para o projeto |
| --- | --- | --- |
| Moghazi et al., Kidney International, 2005 | 207 pacientes com ultrassom e biopsia; comparou medidas ultrassonograficas com glomeruloesclerose, atrofia tubular, fibrose intersticial e inflamacao | A sustentacao clinica para fibrose passa por ecogenicidade e espessura cortical/parenquimatosa comparadas a histologia. |
| Athavale et al., JAMA Network Open, 2021 | 352 biopsias, 6.135 imagens de ultrassom com mascaras adequadas; classificacao de quatro faixas de IFTA | Mostra viabilidade de DL em ultrassom para IFTA, mas somente porque o alvo deriva de biopsia. Acuracia no teste por imagem: `0.8675`; no nivel do paciente: `0.8955`. |
| Ge et al., European Radiology, 2023 | Pacientes com CKD, biopsia, B-mode, elastografia e fatores clinicos | Modelo combinado para IFTA teve AUC de teste `0.85` para leve versus moderada/grave e `0.83` para leve/moderada versus grave; multimodalidade supera medidas isoladas. |
| Chen et al., Abdominal Radiology, 2023 | 160 pacientes com CKD e biopsia; radiomica de ultrassom mais variaveis clinicas | A assinatura radiomica isolada foi moderada (`AUC 0.72`), reforcando que B-mode sozinho nao resolve toda a tarefa. |
| Chen et al., Science Progress, 2025 | 146 pacientes com CKD e biopsia; comparou rim inteiro, parenquima e porcao media | A porcao media teve melhor AUC (`0.74`) que rim inteiro (`0.61`) e parenquima (`0.66`), sugerindo avaliar ROIs alternativas em vez de assumir a medula como unica regiao. |
| Qin et al., Renal Failure, 2024 | CKD com biopsia; escala de cinza, SMI e elastografia | Escala de cinza isolada obteve AUC `0.682`; fusao multimodal chegou a `0.86`, novamente limitando alegacoes baseadas apenas em intensidade B-mode. |

## Leitura do PDF recebido

O PDF local corresponde a:

> Obaid et al. *Noisy Ultrasound Kidney Image Classifications Using Deep
> Learning Ensembles and Grad-CAM Analysis*. AI, 2025;6:172.

O estudo classifica imagens de rim normal versus rim com calculo:

- `1.821` imagens normais e `2.592` com calculo;
- ensemble de `Darknet19`, `Darknet53` e `InceptionV3`;
- teste de robustez adicionando ruido sintetico, incluindo speckle, Poisson,
  Gaussian e salt-and-pepper;
- acuracia maxima relatada de `99.43%` em imagens originais e `99.21%` com
  ruido adicionado;
- uso de Grad-CAM para visualizar regioes consideradas pelo classificador.

Esse PDF pode apoiar duas decisoes tecnicas:

- incluir experimentos de robustez a ruido/speckle no ultrassom;
- adicionar mapas de explicabilidade para verificar se o modelo observa o rim.

Ele nao sustenta:

- identificacao de fibrose;
- segmentacao de medula ou piramides;
- associacao de ecogenicidade intrarrenal com IFTA;
- validade clinica de uma mascara de opacidade como desfecho.

## Ajuste recomendado do problema

### Contribuicao atualmente demonstravel

```text
segmentacao automatica do rim
-> segmentacao anatomica de Medulla dentro da ROI renal
-> extracao exploratoria de marcadores de ecogenicidade/textura
```

Essa contribuicao e defensavel com os dados atuais, desde que `Medulla` seja
tratada como a classe anotada do dataset e nao como prova de piramides
individuais nem de fibrose.

### Trilha clinicamente alinhada a fibrose

```text
imagem de ultrassom
-> segmentacao do rim
-> segmentacao ou definicao de ROI cortical/parenquimatosa
-> marcadores de ecogenicidade, textura e diferenciacao cortico-medular
-> rotulo de referencia clinica ou histologica (preferencialmente IFTA)
-> modelo de estratificacao de alteracao cronica/fibrose
```

A medula pode continuar na analise como componente secundario:

- contraste cortex-medula;
- preservacao/perda da diferenciacao cortico-medular;
- controle anatomico e explicabilidade;
- avaliacao exploratoria de ecotextura.

Ela nao deve ser apresentada como regiao primaria de fibrose enquanto nao
houver evidencias e rotulos que validem essa hipotese.

## Mudancas de nomenclatura para o artigo

| Evitar | Preferir agora | Usar apenas com referencia apropriada |
| --- | --- | --- |
| "detectar fibrose pela opacidade da medula" | "quantificar marcadores ultrassonograficos intrarrenais" | "predizer IFTA" com biopsia/IFTA como ground truth |
| "piramides segmentadas" quando o rotulo e `Medulla` | "segmentacao da classe Medulla anotada" | "piramides renais" apos revisao anatomica especifica |
| "rim com fibrose" em imagens sem laudo | "imagem com alteracao parenquimatosa suspeita" | "fibrose moderada/grave" com classe clinica/histologica |
| "opacidade" isoladamente | "ecogenicidade/intensidade relativa no B-mode" | "ecogenicidade cortical associada a IFTA" em conjunto biopsiado |

## Experimentos a incorporar

### Com os dados atuais

1. Manter o resultado da segmentacao renal e da classe `Medulla`.
2. Extrair tambem features do `Cortex`, ja disponivel no `kidneyUS`.
3. Comparar features de `Cortex`, `Medulla`, rim inteiro e contraste
   cortex-medula, sem chamar nenhuma delas de fibrose.
4. Incluir robustez a speckle/ruido e verificacao visual da regiao analisada,
   inspiradas no PDF recebido.
5. Reportar concordancia entre anotadores para cada estrutura anatomica.

### Para afirmar associacao com fibrose

1. Obter imagens vinculadas a biopsia com grau de IFTA ou laudos clinicos
   definidos por nefrologista.
2. Definir o desfecho antes do treinamento, por exemplo:
   `IFTA < 25%` versus `IFTA >= 25%`, ou quatro classes conforme Athavale.
3. Fazer divisao por paciente, nao por imagem, evitando vazamento entre
   treino e teste.
4. Testar modelos com ROI de cortex/parenquima, rim inteiro e porcao media.
5. Combinar imagem com dados clinicos, como eGFR, quando autorizados e
   disponiveis.
6. Validar externamente, preferencialmente em equipamento ou centro distinto.

## Recomendacao para a reuniao com o nefrologista

Levar quatro perguntas objetivas:

1. A classe `Medulla` do dataset tem valor clinico para a pergunta desejada,
   ou o alvo deveria ser `Cortex`/parenquima e diferenciacao
   cortico-medular?
2. Quais achados ultrassonograficos sao aceitaveis como marcadores de
   doenca cronica sem afirmar fibrose?
3. Existe acesso a imagens pareadas com biopsia, laudo de IFTA, eGFR ou
   estadiamento clinico?
4. Qual terminologia o grupo medico aceita para a contribuicao atual:
   caracterizacao, triagem de alteracao parenquimatosa ou predicao de IFTA?

## Fontes consultadas

- Athavale et al. (2021), JAMA Network Open:
  https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2780063
- Moghazi et al. (2005), Kidney International:
  https://pubmed.ncbi.nlm.nih.gov/15780105/
- Ge et al. (2023), Insights into Imaging:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10017610/
- Chen et al. (2023), Abdominal Radiology:
  https://pubmed.ncbi.nlm.nih.gov/37256330/
- Chen et al. (2025), Science Progress:
  https://journals.sagepub.com/doi/10.1177/00368504251399597
- Qin et al. (2024), multimodal ultrasound deep learning:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11497579/
- Obaid et al. (2025), PDF fornecido, AI:
  https://doi.org/10.3390/ai6080172
