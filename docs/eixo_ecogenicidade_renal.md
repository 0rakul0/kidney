# Eixo central: ecogenicidade renal quantitativa

Esta nota preserva a fundamentação e as decisões metodológicas do marcador de
ecogenicidade intrarrenal para a dissertação. O marcador é exploratório: ele
descreve diferenças de intensidade no B-mode e não fornece, isoladamente, um
diagnóstico de fibrose, DRC ou IFTA.

## Pergunta metodológica

Dadas máscaras de `Cortex`, `Medulla` e `Central Echo Complex` (CEC), é
possível extrair medidas reproduzíveis de ecogenicidade relativa sem aplicar
CLAHE à imagem que entra no cálculo? A resposta é investigada como
caracterização quantitativa da imagem; a associação clínica depende de
validação com eGFR, laudo ou histopatologia.

## Medidas implementadas

Sejam `mu_C`, `mu_M` e `mu_CEC` as intensidades médias da imagem original em
tons de cinza, nas máscaras de córtex, medula e CEC, respectivamente. A faixa
digital é 0--255.

| Medida | Fórmula | Interpretação operacional |
|---|---|---|
| Brilho relativo córtex/CEC | `100 * mu_C / mu_CEC` | 100% indica médias iguais; valores menores indicam córtex mais escuro que o CEC. |
| Diferença CEC--córtex | `mu_CEC - mu_C` | Menor separação de intensidade quando o valor se aproxima de zero. |
| Contraste normalizado | `100 * (mu_CEC - mu_C) / (mu_CEC + mu_C)` | Forma adimensional que atenua, mas não elimina, variações globais de ganho. |
| Brilho relativo córtex/medula | `100 * mu_C / mu_M` | Quantifica a diferenciação córtico-medular. |
| Mediana e IQR | percentis 50, 75 e 25 por região | Resumo robusto da distribuição, menos sensível a pixels extremos. |
| Cobertura e saturação | número de pixels; proporção em `<=1` ou `>=254` | Controle de qualidade operacional da ROI e da imagem. |

As medidas são calculadas somente nas máscaras efetivas: a máscara manual do
revisor, quando existir, prevalece sobre a proposta automática. O arquivo de
imagem usado é sempre o original, sem CLAHE e sem super-resolução. Isso evita
que o realce local altere artificialmente os níveis de cinza usados pelo
marcador.

## Protocolo de amostragem pareada córtex--CEC

A razão calculada sobre toda a máscara continua disponível, mas passa a ser
classificada como **descritor global secundário**. A medida principal é a
amostragem local, pareada e em profundidade aproximadamente equivalente entre
córtex e CEC. A mudança reduz a influência de regiões extensas, heterogêneas
ou com distribuição desigual de pixels em uma das duas estruturas.

Sejam `M_C` e `M_CEC` as máscaras de córtex e CEC. Antes da amostragem, ambas
são erodidas com elemento estruturante de 5 x 5 pixels:

`E(M) = erode(M, K_5x5)`.

Essa etapa exclui pixels de contorno e diminui o risco de misturar tecidos
vizinhos por erro de segmentação ou volume parcial. Na implementação atual,
a faixa vertical que contém as duas estruturas é dividida em três estratos:
superior, médio e inferior. Em cada estrato, no máximo um par é aceito, e
apenas se houver disco circular integralmente contido nas duas máscaras
erodidas. Assim, o procedimento pode produzir de um a três pares; uma faixa
sem suporte anatômico válido não é preenchida artificialmente.

Para cada faixa válida, primeiro é escolhido um centro interno no CEC. O
centro cortical é então selecionado na mesma faixa com coordenada vertical
mais próxima e, em caso de empate, menor distância horizontal. Essa regra usa
a coordenada vertical da imagem como aproximação de mesma profundidade. Os dois
discos têm o mesmo raio e área. O raio operacional é adaptativo: 2,5% da menor
dimensão da imagem, limitado entre 6 e 24 pixels. As coordenadas dos centros,
o raio e a faixa são armazenados para auditoria e para uma futura análise de
sensibilidade do tamanho da ROI.

Na interface de curadoria, os pares efetivamente aceitos também são desenhados
sobre a imagem: a ROI cortical aparece em ciano, a ROI do CEC em laranja e uma
linha tracejada conecta o par. O rótulo informa o número do par, a faixa e a
razão calculada. A sobreposição pode ser ocultada sem alterar o cálculo, para
permitir tanto a inspeção rastreável quanto a leitura limpa da imagem.

O revisor pode ativar o modo de ajuste e mover cada círculo dentro da estrutura
correspondente. Após o reposicionamento, a medida é recalculada na imagem
original antes de ser aceita. O sistema rejeita ROIs fora da máscara erodida,
com raio diferente do par ou em profundidades excessivamente distintas. A
configuração só se torna parte do registro do revisor após a ação explícita de
salvar; a seleção automática pode ser restaurada a qualquer momento.

Para cada par `i`, com médias locais `mu_C,i` e `mu_CEC,i`, são calculadas:

| Medida por par | Fórmula |
|---|---|
| Razão córtex/CEC | `R_i = 100 * mu_C,i / mu_CEC,i` |
| Diferença CEC--córtex | `D_i = mu_CEC,i - mu_C,i` |
| Contraste normalizado | `CN_i = 100 * (mu_CEC,i - mu_C,i) / (mu_CEC,i + mu_C,i)` |

O resultado principal exibido é `mediana(R_i)`. Também são exibidos o IQR,
o número de pares válidos e os resultados individuais das faixas superior,
média e inferior. `mediana(D_i)` e `mediana(CN_i)` podem ser incluídas na
tabela de análise. O cálculo de máscara completa permanece separado e deve
ser relatado como análise descritiva ou de sensibilidade, nunca misturado à
estimativa pareada principal.

Não se deve interpretar a ausência de três pares como falha automaticamente:
em um corte ultrassonográfico, partes do córtex ou do CEC podem não coexistir
nas três faixas após a exclusão das bordas. O relatório deve informar `N`
válido, sem imputar uma ROI ausente. Como a geometria DICOM, a profundidade
física e a curva de TGC não estão disponíveis na imagem rasterizada, “mesma
profundidade” é uma aproximação por linha de imagem; essa limitação deve ser
declarada e reavaliada quando dados de aquisição estiverem disponíveis.

## Texto-base para a dissertação

> A ecogenicidade relativa intrarrenal foi estimada por amostragem pareada
> entre córtex e complexo ecogênico central (CEC), na imagem B-mode original,
> sem aplicação de CLAHE. As máscaras foram erodidas para excluir regiões de
> fronteira. Foram buscados até três pares de ROIs circulares de mesma área,
> posicionados nos estratos superior, médio e inferior, com a ROI cortical
> selecionada na coordenada vertical mais próxima da ROI do CEC. Para cada par
> foram calculadas a razão córtex/CEC, a diferença CEC--córtex e o contraste
> normalizado; a mediana das razões válidas foi adotada como estimativa
> principal, acompanhada do IQR e dos resultados individuais. A razão obtida
> sobre toda a máscara foi preservada como descritor global secundário. A
> escolha de medidas relativas é consistente com a literatura de ecogenicidade
> renal quantitativa, que também ressalta a influência de condições de
> aquisição e a necessidade de interpretação não absoluta [Manley2001]. A
> comparação entre seio renal/CEC e córtex é apoiada pela associação dessa
> diferença com a função renal reportada por Zhao et al. [Zhao2025].

O protocolo de três pares, a erosão de bordas e o raio adaptativo constituem
uma **decisão metodológica operacional deste trabalho**, e não devem ser
apresentados como um protocolo clínico externo já validado. A validação deverá
comparar a medida com os desfechos clínicos ou histopatológicos definidos no
estudo, considerando que ecogenicidade isolada não diagnostica fibrose
[Moghazi2005].


## Fundamentação e interpretação

Manley e O'Neill quantificaram a ecogenicidade cortical pela razão de
intensidades entre córtex e fígado, demonstrando reprodutibilidade da medida,
mas também influência do ganho e do estado de diurese. Esse resultado sustenta
o uso de razões de intensidade, mas exige registrar parâmetros de aquisição e
não tratar uma razão como valor absoluto universal [Manley2001].

Zhao et al. avaliaram diretamente a diferença entre seio renal e córtex em
339 participantes. A diferença de ecogenicidade foi associada à classificação
por eGFR e apresentou AUC de 0,838 para a medida no rim direito. Esse trabalho
fundamenta a inclusão de `mu_CEC - mu_C`, pois o CEC empregado no projeto é a
representação segmentada do complexo ecogênico central/seio renal
[Zhao2025].

Como a alteração de ecogenicidade não é específica de fibrose, o sistema usa o
termo **marcador de anomalia**. Em estudo com 207 pacientes submetidos a
biópsia, a ecogenicidade foi o parâmetro ultrassonográfico com maior correlação
com os achados histológicos, mas as correlações foram moderadas e a fibrose
intersticial não foi determinante independente no modelo multivariado
[Moghazi2005]. Portanto, qualquer desfecho de fibrose/IFTA requer referência
histológica ou clínica apropriada.

## Limitações e controles experimentais

- O desligamento do CLAHE não remove o efeito de ganho, TGC, profundidade,
  compressão e pós-processamento do aparelho.
- O CEC/seio renal é heterogêneo e pode conter gordura, vasos e sistema
  coletor; a máscara deve ser auditada antes de usar o marcador.
- Razões e diferenças dependem da qualidade da segmentação. Por isso, devem
  ser guardadas cobertura, saturação, modelo selecionado e versão da máscara.
- Para comparação clínica convencional, córtex versus fígado (rim direito) ou
  baço (rim esquerdo), na mesma profundidade, continua sendo uma referência
  importante quando esses órgãos estiverem disponíveis [Manley2001;
  Beutler2024].

## Leitura do comparador livre no artefato Curadoria Web

O comparador livre calcula a média dos níveis de cinza originais, sem CLAHE,
em duas ROIs circulares de mesmo raio marcadas pelo revisor: ponto A e ponto
B. A leitura é apresentada por cinco elementos: (1) razão A/B em porcentagem,
que expressa quanto A é mais claro ou escuro que B; (2) diferença `B - A`, em
níveis de cinza; (3) contraste normalizado `(B - A) / (B + A)`; (4) diferença
de profundidade entre os centros das ROIs, em pixels, que deve ser considerada
como possível fonte de atenuação; e (5) IQR entre os pares. Com apenas um par,
o IQR é zero por definição e não representa estabilidade. Recomenda-se usar
três pares quando a análise exigir estimativa de variabilidade.

## Referências a citar

- Manley JA, O'Neill WC. *How echogenic is echogenic? Quantitative acoustics
  of the renal cortex*. American Journal of Kidney Diseases, 2001.
  DOI: [10.1016/S0272-6386(01)80118-9](https://doi.org/10.1016/S0272-6386(01)80118-9).
- Zhao L et al. *Quantifying ultrasound echogenicity difference for accurate
  chronic kidney disease diagnosis*. Journal of Nephrology, 2025.
  DOI: [10.1007/s40620-025-02352-z](https://doi.org/10.1007/s40620-025-02352-z).
- Moghazi S et al. *Correlation of renal histopathology with sonographic
  findings*. Kidney International, 2005.
  DOI: [10.1111/j.1523-1755.2005.00230.x](https://doi.org/10.1111/j.1523-1755.2005.00230.x).
- Beutler BD et al. *An Objective Computer-Assisted Measurement of Sonographic
  Renal Cortical Echogenicity: The Splenorenal Index*. Ultrasound Quarterly,
  2024. DOI: [10.1097/RUQ.0000000000000646](https://doi.org/10.1097/RUQ.0000000000000646).
