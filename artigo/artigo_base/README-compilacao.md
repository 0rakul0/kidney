# Compilação do artigo-base e referências bibliográficas

O artigo-base está em `main.tex`. A bibliografia está em `sbc-template.bib` e utiliza o estilo BibTeX da SBC (`sbc.bst`).

**Atenção:** o repositório continha os arquivos `main.aux`, `main.bbl` e `main.blg` produzidos para uma versão anterior do texto. Eles foram removidos do versionamento e ignorados para que não provoquem citações como `[?, ?]` ou referências como `Tabela ??`. Esses arquivos são produzidos novamente a cada compilação.

## Windows: compilação recomendada

Após instalar MiKTeX ou TeX Live com BibTeX:

```powershell
cd D:\kidney\artigo\artigo_base
powershell -NoProfile -ExecutionPolicy Bypass -File .\compilar_artigo.ps1
```

O script limpa os arquivos auxiliares, executa `pdflatex`, `bibtex`, `pdflatex` e `pdflatex`, e verifica se o log ainda reporta citações ou rótulos indefinidos.

## Linux, macOS ou terminal do Overleaf

A partir da pasta deste artigo:

```bash
rm -f main.aux main.bbl main.blg main.out main.fdb_latexmk main.fls
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

No Overleaf, configure o compilador **pdfLaTeX** e use **Recompile from scratch** (recompilar do zero). O Overleaf normalmente executa BibTeX automaticamente via `latexmk`.

## Como saber se corrigiu?

As referências do texto são chamadas por `\cite{chave}` e devem existir no `sbc-template.bib`. Os rótulos cruzados são chamados por `\ref{rotulo}` e devem ter um `\label{rotulo}`. As chaves citadas e os rótulos do `main.tex` foram confrontados com o `.bib` e com as definições de rótulos; não foram encontradas ausências.

Se houver erros, consulte `main.log` (LaTeX) e `main.blg` (BibTeX). Rodar somente `pdflatex` uma vez **não é suficiente** depois de uma alteração nas citações ou tabelas.

**Observação importante:** o arquivo `main.pdf` já versionado pode representar uma compilação anterior. Ele deve ser regenerado pela sequência acima antes de ser considerado a versão final do artigo.

## Imagens da Curadoria Web

Os blocos das quatro capturas são condicionais: elas são exibidas quando existirem, em `figures/`, os PNGs previstos pelo arquivo `figures/README-curadoria.md`. A ausência de qualquer captura não deve ser resolvida inventando resultados ou fotografias.
