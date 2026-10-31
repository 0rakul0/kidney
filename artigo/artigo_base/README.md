# Artigo-base — Qualificação de Mestrado

**Navegação rápida:** [PDF com sumário](main.pdf#page=3) · [Fonte LaTeX](main.tex) · [Como compilar](README-compilacao.md)

O PDF apresenta um **Sumário** depois do resumo, com capítulos, subseções, referências bibliográficas e apêndice. Os números de página são atualizados automaticamente ao compilar o LaTeX.

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

Para a qualificação, o código utiliza `\versaoqualificacaotrue` e `tocdepth=2`, incluindo seções e subseções. Para eventual submissão no modelo SBC, consulte as instruções de compilação antes de desativar esse modo.
