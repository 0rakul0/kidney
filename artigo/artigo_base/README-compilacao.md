# Qualificação e artigo SBC: arquivos e compilação

## Entradas de compilação

| Fonte | Finalidade |
|---|---|
| [main.tex](main.tex) | **Proposta de Dissertação de Mestrado** (modo padrão; capa, folha de rosto, Resumo, Abstract, listas, sumário e cronograma) |
| [sbc.tex](sbc.tex) | Artigo independente no formato SBC; reutiliza o texto, os dados e as referências de `main.tex`, sem capa, listas ou cronograma |
| [qualificacao_capa.tex](qualificacao_capa.tex) | Identificação acadêmica editável; não altera o bloco de coautores do artigo SBC |
| [sbc-template.bib](sbc-template.bib) | Base bibliográfica compartilhada |
| [CHECKLIST-QUALIFICACAO.md](CHECKLIST-QUALIFICACAO.md) | Pendências documentais e experimentais |

O PPCIC é o **Programa de Pós-Graduação em Ciência da Computação do CEFET/RJ**. A folha de rosto trata o documento como **Proposta de Dissertação de Mestrado**, não como trabalho já aprovado nem como dissertação defendida. O orientador informado é **Felipe da R. Henriques**, já identificado em `qualificacao_capa.tex`. Antes da entrega, conferir se o PPCIC exige o nome por extenso. Coorientação só deve ser indicada quando confirmada.

A formatação acadêmica segue, de forma adaptada, o [Manual do CEFET/RJ](https://eic.cefet-rj.br/ppcic/wp-content/uploads/2018/07/Manual-de-Elaboracao-de-Teses-Dissertacoes-e-Monografias.pdf): formato A4, margens de 3 cm, fonte Times 12 aproximada pelo pacote `times`, entrelinhas 1,5; pré-textuais contados desde a folha de rosto sem numeração visível; números exibidos na primeira página da Introdução, no alto à direita. Foi empregado `abntex2cite` para citações autor-data no modo de qualificação. **Não é uma certificação de conformidade total à norma ABNT atualizada**; revise campos bibliográficos, apresentação das citações, legendas e detalhes institucionais com a orientação.

## Compilação

No terminal, dentro de `artigo/artigo_base`:

```bash
# Proposta de dissertação, modo padrão
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

# Artigo SBC independente
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error sbc.tex
```

A versão SBC gera `sbc.pdf`, enquanto a qualificação gera `main.pdf`. Cada compilação usa seus próprios arquivos auxiliares. Não é preciso alterar um seletor interno. No Windows, `compilar_artigo.ps1` continua compilando `main.tex`, a versão de qualificação.

O [workflow de compilação](https://github.com/0rakul0/kidney/actions/workflows/build-artigo-base.yml) atualiza o `main.pdf` da branch `main` quando o código é alterado e valida se o sumário, as 16 figuras, as 17 tabelas, os elementos pré-textuais e o cronograma foram gerados. Em caso de falha, verifique o log da execução; não trate um PDF anteriormente versionado como recém-compilado.

## Pré-textuais e navegação

No modo de qualificação, a sequência é:

1. Capa e folha de rosto, sem números impressos.
2. Resumo em português (com palavras-chave), em página própria.
3. Abstract em inglês (com keywords), em página própria.
4. Lista de Figuras, Lista de Tabelas, Lista de Abreviaturas e Siglas, Lista de Símbolos.
5. Sumário clicável com seções e subseções.
6. Introdução, capítulos, referências e apêndice, com numeração visível.

Os textos dos resumos são definidos uma única vez e reutilizados no modo SBC, impedindo divergência acidental.

## Cronograma de pesquisa

A seção de Continuidade incorpora cronograma **indicativo de novembro de 2026 a abril de 2027** com entregáveis: congelamento dos protocolos, anotação externa, avaliação da curadoria humana, comparação dos regimes de treinamento, estudo de ecogenicidade e versão revisada da dissertação. **Essas datas são propostas editoriais, não datas de qualificação/defesa confirmadas.** Devem ser ajustadas com a orientação e com o calendário do PPCIC.

## Não incluído antecipadamente

Não foram criadas ficha catalográfica, folha de aprovação assinada, lista fictícia de banca nem datas oficiais de defesa. Esses dados dependem de confirmação e das etapas institucionais aplicáveis.

## Verificações recomendadas

- [x] Incluir Felipe da R. Henriques como orientador na folha de rosto.
- [ ] Confirmar a grafia por extenso do nome do orientador para a versão entregue à banca.
- [ ] Conferir a grafia completa do nome do discente na capa/folha de rosto.
- [ ] Confirmar com a orientação o modelo efetivamente exigido para a qualificação do PPCIC.
- [ ] Validar os meses e entregáveis do cronograma.
- [ ] Inspecionar o PDF completo após compilar: páginas de título, resumos, listas, equações, tabelas, citações, paginação e sumário.
- [ ] Rever as referências em `sbc-template.bib` quanto à completude dos campos e às normas institucionais.

Fontes oficiais: [Documentos PPCIC](https://eic.cefet-rj.br/ppcic/documentos/), [divulgação de qualificação no PPCIC](https://eic.cefet-rj.br/ppcic/divulgacao-no-site/) e [Manual CEFET/RJ](https://eic.cefet-rj.br/ppcic/wp-content/uploads/2018/07/Manual-de-Elaboracao-de-Teses-Dissertacoes-e-Monografias.pdf).
