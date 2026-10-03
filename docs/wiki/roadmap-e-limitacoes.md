# Roadmap e limitações

[← Wiki](README.md)

## Objetivo desta página

Esta página separa claramente:

- o que o projeto já demonstrou;
- o que ainda é limitado;
- o que precisa ser validado;
- quais são os próximos passos técnicos e científicos.

## O que já está estabelecido

O projeto já possui:

- pipeline de segmentação renal;
- comparação entre arquiteturas;
- segmentação intrarrenal;
- consenso entre modelos;
- expansão por pseudo-rótulos;
- Curadoria Web;
- edição de máscaras;
- rastreabilidade por revisor;
- inferência local;
- protocolo exploratório de ecogenicidade.

## Limitação 1 — propagação de erro

A arquitetura é em cascata.

Logo:

```text
erro na cápsula
→ erro na ROI
→ erro intrarrenal
→ erro na ecogenicidade
```

A qualidade da primeira etapa continua sendo crítica.

## Limitação 2 — pseudo-rótulos

Uma pseudo-máscara pode parecer plausível e ainda estar anatomicamente incorreta.

Por isso, pseudo-rótulos são úteis para:

- expansão;
- triagem;
- experimentos;
- preparação de fila.

Mas não devem ser confundidos com referência clínica revisada.

## Limitação 3 — segmentação intrarrenal

Córtex e medula são estruturas de limite menos definido que a cápsula.

Isso explica parte da diferença entre o Dice da cápsula e o Dice intrarrenal.

## Limitação 4 — ecogenicidade

A intensidade do ultrassom não depende apenas do tecido.

Pode variar com:

- ganho;
- TGC;
- profundidade;
- equipamento;
- preset;
- compressão;
- pós-processamento.

Portanto, as medidas atuais devem ser interpretadas como relativas e exploratórias.

## Limitação 5 — ausência de desfecho clínico

O pipeline atual não possui, de forma consolidada, referência clínica/histológica suficiente para afirmar associação causal com fibrose ou DRC.

Esse é um limite científico, não apenas técnico.

## Limitação 6 — generalização externa

Boa performance no kidneyUS não garante desempenho idêntico em:

- outro hospital;
- outro equipamento;
- outro operador;
- outra população.

A expansão MONAI ajuda na diversidade, mas não substitui validação externa controlada.

## Prioridade 1 — concluir base curada

Criar um conjunto em que cada máscara utilizada como referência final possua:

- origem;
- revisor;
- expertise;
- status;
- versão;
- data.

## Prioridade 2 — medir concordância

Um lote revisado independentemente pode permitir medir:

- concordância aceitar/corrigir/rejeitar;
- Dice entre revisores;
- IoU entre revisores;
- diferença entre especialista e não especialista treinado.

## Prioridade 3 — congelar avaliação final

Após a curadoria, criar um teste final:

- independente;
- por paciente/exame;
- não reutilizado para seleção de modelo;
- totalmente revisado.

## Prioridade 4 — comparar três cenários

Uma comparação cientificamente interessante é:

| Cenário | Pergunta |
|---|---|
| Manual inicial | Qual é o baseline? |
| Pseudo-rótulos automáticos | Quanto a automação ajuda? |
| Human-in-the-loop | Quanto a revisão melhora a expansão? |

## Prioridade 5 — validar ecogenicidade

O protocolo córtex–CEC deve passar por análise de sensibilidade.

Exemplos:

- raio da ROI;
- número de pares;
- profundidade;
- erosão;
- escolha automática versus manual;
- estabilidade entre revisores.

## Prioridade 6 — incorporar dados clínicos

Uma fase clínica futura pode incluir:

- eGFR;
- creatinina;
- laudo;
- diagnóstico;
- histopatologia, quando disponível.

Essa etapa exige governança e desenho de estudo próprios.

## Prioridade 7 — coorte externa

Uma possível fase HUPE pode funcionar como avaliação externa.

Idealmente, deve ser tratada como coorte separada.

## Evolução da Curadoria Web

Para uso local, SQLite é suficiente.

Para uso institucional multiusuário, considerar:

- autenticação;
- autorização;
- PostgreSQL;
- controle de concorrência;
- armazenamento central;
- trilha de auditoria;
- backups;
- versionamento das regras de revisão.

## Evolução do pipeline

Possíveis extensões técnicas:

- active learning;
- seleção de casos por incerteza;
- ensemble;
- calibração probabilística;
- análise de qualidade automática;
- modelos com atenção;
- segmentação hierárquica;
- aprendizado semi-supervisionado.

Essas extensões só devem ser priorizadas depois de estabilizar o protocolo de referência.

## Critério de resultado publicável

Um resultado forte deve ter:

1. pergunta científica definida;
2. população definida;
3. dataset identificado;
4. split reproduzível;
5. rótulo confiável;
6. baseline;
7. comparação justa;
8. métrica apropriada;
9. análise de erro;
10. limitações;
11. checkpoint;
12. commit;
13. protocolo de curadoria.

## Objetivo de longo prazo

O objetivo do projeto não é apenas produzir uma máscara.

A meta é construir uma infraestrutura capaz de conectar:

```text
imagem
→ anatomia
→ curadoria
→ medida quantitativa
→ hipótese clínica
```

mantendo rastreabilidade entre todas essas etapas.

## Próximo marco recomendado

O marco científico mais importante é concluir um conjunto curado e congelado que permita comparar, no mesmo protocolo:

- modelo manual;
- pseudo-rotulagem;
- human-in-the-loop.

Esse conjunto pode se tornar a base metodológica central da dissertação e de uma publicação posterior.
