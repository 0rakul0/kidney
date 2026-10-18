# Compilacao limpa do artigo-base (SBC / BibTeX).
# Execute na pasta artigo/artigo_base ou de qualquer diretorio:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\compilar_artigo.ps1

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

try {
    foreach ($programa in @("pdflatex", "bibtex")) {
        if (-not (Get-Command $programa -ErrorAction SilentlyContinue)) {
            throw "Comando '$programa' nao encontrado. Instale MiKTeX ou TeX Live com BibTeX e adicione ao PATH."
        }
    }

    # Evita herdar citacoes e numeracoes de versoes anteriores do artigo.
    foreach ($arquivo in @("main.aux", "main.bbl", "main.blg", "main.out", "main.fdb_latexmk", "main.fls")) {
        if (Test-Path -LiteralPath $arquivo) {
            Remove-Item -LiteralPath $arquivo -Force
        }
    }

    Write-Host "1/4: LaTeX - gerar referencias auxiliares"
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "Falha na primeira execucao do pdflatex" }

    Write-Host "2/4: BibTeX - gerar bibliografia a partir de sbc-template.bib"
    & bibtex main
    if ($LASTEXITCODE -ne 0) { throw "Falha na execucao do bibtex" }

    Write-Host "3/4: LaTeX - associar bibliografia e rotulos"
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "Falha na segunda execucao do pdflatex" }

    Write-Host "4/4: LaTeX - atualizar referencias cruzadas"
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "Falha na terceira execucao do pdflatex" }

    $indefinidas = Select-String -Path "main.log" -Pattern "Citation .* undefined|Reference .* undefined|There were undefined references|There were undefined citations" -Quiet
    if ($indefinidas) {
        throw "Ainda ha referencias indefinidas em main.log. Verifique a chave de citacao ou o rotulo."
    }

    if (-not (Test-Path "main.pdf")) {
        throw "A compilacao terminou sem criar main.pdf"
    }

    Write-Host ""
    Write-Host "Concluido: $((Resolve-Path 'main.pdf').Path)"
    Write-Host "Bibliografia gerada de novo; citacoes e referencias cruzadas conferidas."
}
finally {
    Pop-Location
}
