# Conteúdo técnico consolidado

[← Wiki](README.md)

## Objetivo

Os antigos arquivos Markdown soltos em `docs/` foram incorporados à wiki e removidos para evitar duplicação e divergência entre versões da documentação.

Esta página registra **o que foi absorvido** e onde esse conteúdo passou a viver.

## Engenharia de dados e fontes externas

O conteúdo sobre:
- busca de datasets externos;
- MONAI/NVIDIA;
- Kaggle;
- organização de fontes brutas e processadas;
- estratégia incremental de download;
- conversão DICOM→PNG;
- proveniência;
- expansão de dataset;

foi incorporado em:

- [Dados e curadoria](dados-e-curadoria.md)
- [Scripts de engenharia de dataset](scripts-engenharia-dataset.md)
- [Execução e reprodutibilidade](execucao-e-reprodutibilidade.md)

## Organização do repositório e datasets

O conteúdo sobre:
- função de cada pasta;
- datasets intermediários;
- bases supervisionadas;
- pseudo-expansões;
- hardlinks;
- artefatos canônicos;
- arquivos temporários;

foi incorporado em:

- [Arquitetura do pipeline](arquitetura.md)
- [Dados e curadoria](dados-e-curadoria.md)
- [Execução e reprodutibilidade](execucao-e-reprodutibilidade.md)

## Segmentação renal

O conteúdo sobre:
- DeepLabV3 no `dataset_geral`;
- validação cruzada;
- modelo campeão;
- geração de máscaras faltantes;
- thresholds;
- filtros morfológicos;
- pseudo-máscaras;

foi incorporado em:

- [Modelos e resultados](modelos-e-resultados.md)
- [Scripts de segmentação](scripts-segmentacao.md)
- [Histórico metodológico](historico-metodologico.md)

## Segmentação intrarrenal e Medulla

O conteúdo sobre:
- baseline heurístico;
- DeepLab de Medulla;
- MedullaROIUNet;
- estabilidade da cascata;
- consenso entre modelos;
- expansão pseudo-rotulada;
- modelo intrarrenal multiclasse;
- WiSARD/WNN como comparador;

foi incorporado em:

- [Modelos e resultados](modelos-e-resultados.md)
- [Scripts de segmentação](scripts-segmentacao.md)
- [Histórico metodológico](historico-metodologico.md)
- [Roadmap e limitações](roadmap-e-limitacoes.md)

## Curadoria human-in-the-loop

O conteúdo sobre:
- necessidade de revisão humana;
- piloto de concordância;
- aceitar/corrigir/rejeitar;
- revisão em lotes;
- separação entre treino intermediário e teste final;
- rastreabilidade por revisor;

foi incorporado em:

- [Dados e curadoria](dados-e-curadoria.md)
- [Curadoria e inferência](curadoria-e-inferencia.md)
- [Histórico metodológico](historico-metodologico.md)

## Ecogenicidade e fibrose

O conteúdo sobre:
- ecogenicidade cortical;
- comparação córtex–CEC;
- ROIs pareadas;
- erosão de bordas;
- mediana/IQR;
- limitações de ganho/TGC;
- diferença entre caracterização ultrassonográfica e diagnóstico de fibrose/IFTA;

foi incorporado em:

- [Ecogenicidade renal quantitativa](ecogenicidade.md)
- [Curadoria e inferência](curadoria-e-inferencia.md)
- [Roadmap e limitações](roadmap-e-limitacoes.md)

## Referências visuais e interpretação clínica

O conteúdo sobre:
- rim com aspecto preservado;
- aumento de ecogenicidade cortical;
- perda de diferenciação córtico-medular;
- referências visuais públicas;
- limites de interpretação;

foi incorporado principalmente em:

- [Ecogenicidade renal quantitativa](ecogenicidade.md)
- [Roadmap e limitações](roadmap-e-limitacoes.md)

## Regra atual

A wiki é agora a **fonte única de documentação Markdown dentro de `docs/`**.

Novas decisões, resultados e comportamentos de scripts devem ser documentados diretamente em `docs/wiki/`, evitando recriar documentação paralela solta em `docs/`.
