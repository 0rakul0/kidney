# Proximos passos: modelo intrarrenal para avaliacao do parenquima

## Contexto

O modelo campeao atual, DeepLabV3 com backbone ResNet-50, resolve a etapa de
segmentacao renal externa. Ele localiza o rim e produz uma mascara binaria da
regiao renal. No entanto, a tarefa central da tese nao deve depender apenas da
borda do rim. A identificacao de possiveis alteracoes associadas a fibrose exige
olhar para dentro da regiao renal, especialmente para o parenquima e para a
organizacao visual das estruturas internas.

Por isso, o proximo passo metodologico sera usar a saida do segmentador como
entrada para um terceiro modelo. Esse terceiro modelo sera responsavel por
avaliar se a imagem contem uma regiao renal anatomicamente util e se ha padroes
texturais intrarrenais sugestivos de alteracao parenquimatosa.

## Papel de cada modelo

O pipeline proposto passa a ter tres etapas:

```text
Imagem de ultrassom
    -> Modelo 1 / bases iniciais: segmentacao renal de referencia e pseudo-rotulagem inicial
    -> Modelo 2: DeepLabV3 campeao para segmentar o rim
    -> Modelo 3: avaliacao intrarrenal concentrada na ROI renal
```

O modelo 2 atua como localizador anatomico. Ele reduz o campo de busca e remove
regioes externas ao rim, como textos da tela, bordas, outros orgaos e artefatos.

O modelo 3 devera receber a imagem renal ja isolada ou mascarada. Assim, o
aprendizado fica concentrado no parenquima renal e em sinais internos, em vez de
apenas reconhecer o contorno do orgao.

## Entrada proposta para o modelo 3

A entrada do modelo 3 pode ser composta por tres canais:

- Canal 1: imagem original em escala de cinza.
- Canal 2: imagem recortada ou mascarada pela ROI renal gerada pelo DeepLab.
- Canal 3: mascara binaria do rim.

Tambem sera criada uma versao de recorte ao redor do rim, usando a bounding box
da mascara com uma margem fixa. Esse recorte deve preservar o contexto imediato,
mas reduzir informacao irrelevante fora da regiao renal.

## Saidas esperadas

O modelo 3 deve ser inicialmente formulado como um classificador de qualidade
anatomica e textura intrarrenal, evitando afirmar fibrose diretamente sem
confirmacao clinica ou histologica.

Classes iniciais sugeridas:

- `rim_bom`: rim visivel, anatomia compativel e parenquima avaliavel.
- `rim_duvidoso`: imagem possivelmente renal, mas com qualidade ou corte
  insuficiente.
- `nao_rim`: imagem nao adequada para analise renal.
- `parenquima_visivel`: estrutura interna suficiente para analise.
- `parenquima_nao_visivel`: estrutura interna insuficiente.
- `alteracao_textural_sugestiva`: padrao interno possivelmente associado a
  alteracao parenquimatosa.
- `sem_alteracao_textural_evidente`: textura sem suspeita visual evidente.

Na redacao cientifica, a saida principal deve ser descrita como:

> padrao ultrassonografico sugestivo de alteracao parenquimatosa compativel com
> fibrose renal

Essa formulacao e mais defensavel do que afirmar diagnostico direto de fibrose.

## Curadoria necessaria

Antes do treino do modelo 3, sera necessario montar uma base revisada com
exemplos positivos e negativos.

Fontes positivas fortes:

- imagens com mascaras manuais ou primarias do dataset original;
- imagens em que a anatomia renal e claramente visivel;
- imagens com parenquima avaliavel.

Fontes negativas:

- imagens rejeitadas pelo modelo campeao;
- imagens MONAI com alta confianca numerica, mas sem aspecto renal claro;
- imagens sem piramides, sem parenquima avaliavel ou com corte inadequado;
- imagens com artefatos, Doppler/coloridas ou estruturas nao renais dominantes.

Uma etapa manual pequena, com revisao de algumas centenas de exemplos, deve ser
priorizada. Essa revisao servira como conjunto inicial de treinamento e validacao
do modelo 3.

## Evidencia visual

Como o modelo 3 sera o componente mais critico para a tese, ele nao deve ser
apresentado apenas como uma caixa preta. O projeto deve incluir evidencias
visuais da regiao usada para decisao.

Metodos recomendados:

- Grad-CAM ou mapa de atencao para classificadores CNN;
- mapa de ativacao sobreposto ao recorte renal;
- comparacao entre imagem original, mascara renal, ROI e mapa de importancia;
- features radiomicas ou texturais calculadas dentro da mascara renal.

Essas evidencias ajudam a mostrar se o modelo esta olhando para o parenquima ou
se esta aprendendo algum artefato externo.

## Implementacao planejada

Etapas tecnicas:

1. Criar script para gerar ROIs renais a partir do `dataset_geral`.
2. Salvar, para cada imagem, a imagem original, mascara, recorte renal e imagem
   mascarada.
3. Criar manifesto de revisao manual com campos para qualidade anatomica e
   textura intrarrenal.
4. Rotular um primeiro lote de imagens como `rim_bom`, `rim_duvidoso`,
   `nao_rim` e, quando possivel, `alteracao_textural_sugestiva`.
5. Treinar um classificador inicial usando a ROI renal.
6. Avaliar o classificador separadamente da segmentacao.
7. Gerar mapas de explicabilidade para os melhores e piores casos.
8. Usar o classificador como filtro anatomico antes de aceitar novas
   pseudo-mascaras ou novas imagens para treino.

## Triagem automatica de candidatos

Foi criado um script inicial para buscar candidatos dentro do `dataset_geral`
com base na ROI renal e em descritores visuais simples:

`D:\kidney\src\segmentation\tools\find_intrarenal_candidates.py`

O script calcula, dentro da mascara renal, medidas como brilho medio, contraste
entre percentis, variacao interna, textura por Laplaciano e componentes escuros
candidatos a piramides renais. A partir desses valores, ele gera dois rankings:

- candidatos com aspecto mais proximo de rim com diferenciacao interna visivel;
- candidatos com maior suspeita de alteracao textural, aumento de ecogenicidade
  ou perda de contraste interno.

Comando padrao:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\find_intrarenal_candidates.py --top-k 30
```

Saidas principais:

- `D:\kidney\results\reference_matching\intrarenal_candidate_scores.csv`
- `D:\kidney\results\reference_matching\intrarenal_healthy_candidates.csv`
- `D:\kidney\results\reference_matching\intrarenal_suspicious_candidates.csv`
- `D:\kidney\results\reference_matching\intrarenal_candidate_panels`

Esses rankings sao apenas uma triagem visual. Eles nao substituem revisao
anatomica nem confirmam fibrose. O objetivo e reduzir o espaco de busca e
acelerar a construcao do conjunto de revisao manual para o modelo 3.

## Impacto esperado na tese

Essa etapa fortalece a metodologia porque separa dois problemas diferentes:

- localizar o rim;
- analisar o conteudo interno do rim.

O DeepLab campeao resolve a primeira etapa. O modelo intrarrenal passa a atacar
o problema mais proximo da pergunta clinica da tese: identificar padroes
ultrassonograficos internos que possam sugerir alteracao parenquimatosa
compativel com fibrose.
