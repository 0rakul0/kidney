# Compilação do artigo-base e referências bibliográficas

O artigo-base está em `main.tex`. A bibliografia fonte está em `sbc-template.bib` e utiliza o estilo BibTeX da SBC (`sbc.bst`). O arquivo `main.bbl`, gerado a partir do `.bib`, é mantido no repositório para evitar o desaparecimento das referências em compilações que não executam BibTeX.

**Correção de bibliografia:** o arquivo `main.bbl` é mantido no repositório para incluir todas as 19 referências do texto atual, inclusive quando o editor apenas executa `pdflatex`. Os arquivos temporários `main.aux` e `main.blg` continuam fora do versionamento por poderem conter referências obsoletas. A compilação completa regenera o `main.bbl` a partir do arquivo fonte `sbc-template.bib`.

## Windows: compilação recomendada

Após instalar MiKTeX ou TeX Live com BibTeX:

```powershell
cd D:\kidney\artigo\artigo_base
powershell -NoProfile -ExecutionPolicy Bypass -File .\compilar_artigo.ps1
```

O script limpa os arquivos auxiliares locais, executa `pdflatex`, `bibtex`, `pdflatex` e `pdflatex`, e verifica se o log ainda reporta citações ou rótulos indefinidos. Ele também regenera localmente o arquivo `main.bbl`; quando houver mudanças na bibliografia, inclua esse arquivo no mesmo commit do `main.tex` e do `.bib`.

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
