# Artigo-base — Qualificação de Mestrado

**Navegação rápida:** [PDF com sumário](main.pdf) · [Fonte LaTeX](main.tex) · [Como compilar](README-compilacao.md)

Na versão de qualificação, o PDF apresenta **Resumo**, **Abstract**, listas de figuras, tabelas, abreviaturas e siglas, símbolos e **Sumário** navegável, seguido das seções, referências e apêndice. As páginas são recalculadas automaticamente na compilação. Esses elementos pré-textuais suplementares são desativados no modo de artigo SBC.

## Sumário do trabalho

1. **Introdução** — contexto, hipótese, objetivos e contribuições.
2. **Fundamentação Teórica e Trabalhos Relacionados** — dados anotados, arquiteturas, métricas, destilação, curadoria e ecogenicidade.
3. **Metodologia** — fontes, separação experimental, segmentação, consenso, destilação e revisão humana.
4. **Resultados** — cápsula, imagens externas, estudantes, cascata, medula, curadoria e estudo piloto.
5. **Discussão** — interpretação crítica, comparação com a literatura e limites da contribuição.
6. **Limitações e Ameaças à Validade** — riscos experimentais, anatômicos, externos e clínicos.
7. **Continuidade da Dissertação** — experimentos planejados e critérios de avaliação.
8. **Conclusão** — contribuições demonstradas e próximos passos.
9. **Referências Bibliográficas**.
10. **Apêndice** — exemplo complementar do assistente multimodal.

### Se o GitHub mostrar um PDF antigo

O arquivo `main.tex` é a fonte do texto, mas o `main.pdf` precisa ser recompilado depois de cada alteração. A rotina [Artigo-base (PDF de qualificação)](https://github.com/0rakul0/kidney/actions/workflows/build-artigo-base.yml) recompila, verifica o conteúdo do sumário e publica o PDF. Abra o arquivo `main.pdf` e use **Download raw** para obter a versão do repositório, em vez de uma visualização anteriormente armazenada em cache.

Para a qualificação, o código utiliza `\versaoqualificacaotrue` e `tocdepth=2`, incluindo seções e subseções. A posição do sumário varia conforme a quantidade de páginas das listas de figuras e tabelas. A classificação formal da versão de **Proposta de Dissertação** (capa, folha de rosto, orientadores, paginação e margens) exige confirmação do modelo aprovado pelo PPCIC; a presente pasta permanece como artigo-base expandido. Para eventual submissão no modelo SBC, consulte as instruções de compilação antes de desativar esse modo.
