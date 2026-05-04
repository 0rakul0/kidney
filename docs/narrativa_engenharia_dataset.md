# Narrativa da engenharia de dataset

## Motivacao

O projeto comecou com uma base local preparada para segmentacao renal, com
imagens e mascaras binarias organizadas em treino, validacao e teste. Essa base
foi suficiente para treinar os primeiros segmentadores e comparar arquiteturas,
mas ainda era pequena para sustentar uma etapa posterior de identificacao de
alteracoes parenquimatosas associadas a fibrose renal.

Por isso, a estrategia passou a ser ampliar a diversidade de imagens de
ultrassom renal sem misturar automaticamente qualquer dado externo ao conjunto
principal. A ampliacao foi tratada como uma etapa separada de engenharia de
dataset, com tres objetivos:

1. encontrar fontes externas relevantes;
2. preservar a proveniencia de cada imagem;
3. aceitar mascaras geradas automaticamente apenas quando houvesse confianca
   operacional suficiente.

## Busca por bases externas

A busca considerou Kaggle, Hugging Face, repositorios GitHub, Figshare, indices
curados de dados abertos e repositorios publicos de imagem medica.

No Kaggle foram identificados conjuntos uteis para aumento de robustez, como
imagens de ultrassom renal com classes `stone` e `no stone`, alem de bases
abdominais com segmentacao de orgaos. Esses dados ainda dependem do token
`kaggle.json` para download automatizado.

No Hugging Face, a primeira busca por `kidney`, `renal ultrasound` e
`kidney ultrasound` nao encontrou uma base forte de ultrassom renal com
segmentacao. A maior parte dos resultados era tabular, CT, texto ou pouco
relacionada ao objetivo da tese.

Entre as fontes fora do Kaggle, os melhores candidatos foram:

- `kidneyUS`, por ser diretamente relacionado a ultrassom renal e conter
  anotacoes multiclasses do rim;
- o diretorio NIDUS/RadOSS, usado como indice curado para localizar bases de
  ultrassom;
- o dataset abdominal da Mississippi State, por conter classe de rim e
  organizacao normal/anormal;
- o repositorio CGPxy, que descreve 200 imagens de ultrassom renal, mas exige
  termo de liberacao;
- o repositorio MONAI/NVIDIA Clinical Ultrasound Image Repository, por ser uma
  base ampla, publica e com muitos exames abdominais/retroperitoneais.

## Estrategia incremental para MONAI/NVIDIA

O MONAI/NVIDIA Clinical Ultrasound Image Repository nao e um dataset pequeno nem
especifico apenas de rim. Ele e um repositorio clinico amplo, composto por
estudos DICOM de ultrassom de diferentes regioes e protocolos. A vantagem e a
diversidade: ha muitos estudos abdominais e retroperitoneais que podem conter
rim. A desvantagem e o volume: baixar tudo seria desnecessario, pesado e pouco
controlado.

Por isso, a estrategia adotada foi incremental:

```text
baixar metadados globais
-> filtrar estudos candidatos por termos renais/retroperitoneais
-> estimar o tamanho de cada estudo
-> baixar apenas um lote inicial dos menores estudos
-> converter e filtrar localmente
-> apagar zips brutos apos validar a conversao
```

Essa abordagem permite crescer a base aos poucos. Se o lote inicial gerar boas
imagens e boas mascaras, novos lotes podem ser baixados repetindo o mesmo
processo. Se o lote tiver muito ruido, o custo computacional e de armazenamento
fica limitado.

## Curadoria MONAI/NVIDIA executada

Primeiro foram baixados os metadados globais:

```text
external_data/raw/MONAI_ClinicalUltrasoundRepository/meta-only/all-meta.json
external_data/raw/MONAI_ClinicalUltrasoundRepository/meta-only/all-meta.csv
```

Em seguida, os metadados foram filtrados por termos como `RENAL`, `KIDNEY` e
`RETROPERITONEAL`. Esse filtro encontrou 238 estudos candidatos.

Como os 238 estudos somavam aproximadamente 68,82 GB, a base passou a ser
baixada em partes. O primeiro lote controlado trouxe os 45 menores estudos
renais/retroperitoneais, totalizando aproximadamente 1,97 GB em arquivos zip.
Depois da validacao desse fluxo, os lotes seguintes foram baixados pulando os
estudos ja processados ate esgotar os 238 candidatos.

Depois disso foi feita a etapa de alivio da carga:

```text
zip bruto MONAI
-> ler DICOMs
-> converter so frames uteis para PNG
-> pular RGB/coloridos por padrao
-> reduzir cines para poucos frames representativos
-> manter metadados e manifesto
-> remover zips brutos apos validar a conversao
```

Resultado acumulado apos todos os lotes:

- 238 estudos renais/retroperitoneais candidatos processados;
- 68,82 GB aproximados de zips brutos baixados temporariamente em lotes;
- 4.487 imagens PNG B-mode/escala de cinza curadas;
- 5.716 entradas rejeitadas por serem RGB/coloridas;
- pasta processada com aproximadamente 676 MB no total;
- nenhum zip bruto mantido localmente apos a conversao.

Os arquivos curados ficaram em:

```text
external_data/processed/monai_renal_png/images/
external_data/processed/monai_renal_png/metadata/
external_data/processed/monai_renal_png/manifest.csv
external_data/processed/monai_renal_png/summary.json
```

## Construcao do `dataset_geral`

Depois da curadoria das fontes externas, foi criada uma base consolidada chamada
`dataset_geral/`. Ela nao substitui os datasets originais; funciona como uma
visao unificada das imagens disponiveis e das mascaras aceitas.

O formato e:

```text
dataset_geral/
    imagens/
    mascaras/
    manifest.csv
    summary.json
    relatorios/
        duplicadas_por_hash.csv
        faltando_mascara.csv
        mascaras_geradas.csv
```

O script de construcao percorre as fontes disponiveis, remove duplicatas por
hash, copia imagens para `dataset_geral/imagens/` e procura mascaras existentes.
Quando a mascara ja existe, ela e copiada. Quando nao existe, o segmentador
renal atual tenta gerar uma pseudo-mascara.

Pseudo-mascaras novas so entram em `dataset_geral/mascaras/` quando passam por
criterios de qualidade:

- confianca media minima do modelo;
- area relativa plausivel;
- quantidade minima de pixels de primeiro plano;
- limite de componentes conectados.

Na ultima montagem, com limiar operacional de confianca `0.90`, o resultado foi:

- 5.994 imagens unicas;
- 1.001 mascaras existentes copiadas;
- 2.961 pseudo-mascaras geradas e aceitas;
- 2.032 imagens ainda sem mascara aceita;
- 3.962 imagens com mascara em `dataset_geral/mascaras/`.

As imagens sem mascara aceita nao foram descartadas. Elas permanecem no
manifesto e no relatorio `faltando_mascara.csv`, pois representam o proximo
conjunto de casos para revisao manual, ajuste de segmentador ou nova tentativa
com outro modelo.

## Papel na tese

Essa etapa cria uma base mais ampla e rastreavel para a segmentacao renal. Ela
nao transforma automaticamente imagens externas em evidencia clinica de fibrose.
O papel dela e aumentar a diversidade de imagens para treinar e testar
segmentadores, mantendo separacao entre:

- mascaras manuais ou ja existentes;
- pseudo-mascaras aceitas por criterios automaticos;
- imagens sem mascara confiavel.

Essa separacao e importante para a tese, porque permite relatar claramente quais
resultados vieram da base original, quais vieram de pseudo-labeling e quais
dados externos foram usados apenas como reforco de robustez.
