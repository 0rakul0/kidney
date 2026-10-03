# Roadmap e limitações

[← Wiki](README.md)

## Limitações atuais

- erros da cápsula propagam-se para as etapas internas;
- pseudo-máscaras não são referência clínica;
- ecogenicidade depende de ganho, TGC, profundidade e pós-processamento;
- associação com DRC, fibrose ou IFTA requer desfechos clínicos/histológicos;
- resultados de datasets diferentes não devem ser comparados diretamente.

## Próximos passos

1. consolidar uma base curada com identidade e expertise do revisor;
2. congelar teste humano por paciente;
3. medir concordância interobservador;
4. comparar baseline manual × pseudo-rótulos × human-in-the-loop;
5. concluir auditoria das pseudo-máscaras;
6. validar sensibilidade das ROIs de ecogenicidade;
7. incorporar dados clínicos com governança definida;
8. avaliar generalização em coorte externa.

## Evolução técnica da Curadoria Web

Para uso institucional multiusuário: autenticação, autorização, banco central, armazenamento aprovado, trilha de auditoria e versionamento do protocolo.

## Critério para publicação

Cada resultado deve vir acompanhado por hipótese, dataset, split, status dos rótulos, métrica, comparação justa, limitações, commit e checkpoint.

## Objetivo final

Transformar a segmentação renal em infraestrutura confiável para investigação quantitativa de estruturas e padrões intrarrenais, separando claramente automação, curadoria e interpretação clínica.
