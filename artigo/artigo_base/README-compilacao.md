# Artigo-base: qualificação e compilação

Fonte principal: [main.tex](main.tex). Bibliografia: [sbc-template.bib](sbc-template.bib). O arquivo `main.bbl` é um auxiliar derivado pelo BibTeX e deve ser atualizado junto da fonte quando houver mudança de referências.

## Modos de documento

O preâmbulo de `main.tex` contém um seletor:

```tex
\versaoqualificacaotrue   % sumário clicável e apêndice do assistente
% \versaoqualificacaofalse  % artigo sem sumário e sem apêndice de qualificação
```

Na **qualificação**, o documento exibe sumário navegável após o resumo, marcadores de navegação de seções no PDF e exemplo multimodal complementar como apêndice. Na **submissão em formato SBC**, desative a opção acima e confira também as exigências específicas do evento (número de páginas, margens e anonimização, se aplicável).

## Compilação

No Windows, a partir de `D:\kidney\artigo\artigo_base`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\compilar_artigo.ps1
```

Em Linux/macOS ou Overleaf, execute BibTeX entre passes de pdfLaTeX; o comando mais simples, com latexmk instalado, é:

```bash
cd artigo/artigo_base
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

Alternativa manual:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

O PDF `main.pdf` versionado pode estar defasado em relação ao `main.tex`. **Não considere a existência do arquivo PDF prova de que o conteúdo atual foi recompilado.** Após compilar, confira título, sumário, figuras, tabelas, páginas finais, apêndice e a lista de referências.

## Revisão para qualificação

- A **Introdução** define problema, hipótese, objetivos e contribuições.
- A **Fundamentação** está agrupada em oito eixos, sem explicações redundantes de curadoria e pseudorrotulação.
- A **Metodologia** separa bases, testes, máscaras manuais, pseudomáscaras e limites de seleção de modelos.
- Os **Resultados** seguem o processamento da cápsula até a cascata, curadoria e demonstração exploratória.
- A **Discussão** interpreta resultados e ameaça à validade, sem tratar cobertura do MONAI como acurácia anatômica.
- A **Continuidade** propõe experimentos para seleção independente, revisão humana, pseudo-expansão controlada e repetibilidade da ecogenicidade.
- O **assistente multimodal** permanece demonstrativo e não é usado como prova de diagnóstico.

## Conferências antes de apresentar à banca

1. Confirmar que todos os `\cite{...}` existem no arquivo BibTeX e que os `\ref{...}` possuem `\label{...}`.
2. Conferir a descrição das arquiteturas frente aos scripts e checkpoints **efetivamente executados**; resultados históricos devem continuar identificados.
3. Verificar se o teste da combinação de professores foi usado na seleção operacional; a qualificação atual explicita esse viés e propõe reavaliação.
4. Confirmar as versões reais dos modelos externos no estudo multimodal; sem ID técnico, a conclusão é apenas ilustrativa.
5. Não afirmar que a pseudomáscara do MONAI é correta sem revisão; 3.707/4.479 expressa cobertura operacional.
6. Conferir o sentido da razão córtex/CEC em cada tela e equação. A razão inversa deve ser nomeada explicitamente.
7. Revisar visualmente o PDF inteiro, inclusive diagramas, tabelas longas e páginas do sumário.

## Figuras

Os arquivos utilizados estão em `figures/`. As figuras de Curadoria Web são registros da interface real, não evidência isolada de ganho de qualidade. Em particular, o exemplo multimodal utiliza checkpoints descritos em sua legenda; ele não substitui o teste agrupado da cascata destilada.
