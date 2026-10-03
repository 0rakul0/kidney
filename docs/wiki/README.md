# Wiki — Kidney

> Estado documentado a partir da branch `main` em 03/10/2026. O projeto é experimental e de pesquisa; as saídas não constituem diagnóstico clínico.

## Visão geral

O **Kidney** evoluiu de experimentos de segmentação renal para um pipeline rastreável de visão computacional aplicado à ultrassonografia renal. O fluxo atual combina engenharia de datasets, segmentação da cápsula, consenso entre modelos, segmentação intrarrenal, curadoria human-in-the-loop, quantificação exploratória de ecogenicidade, inferência local e produção de resultados para dissertação/artigo.

```mermaid
flowchart LR
    A[Fontes] --> B[Engenharia de dataset]
    B --> C[Segmentação da cápsula]
    C --> D[Consenso e curadoria]
    D --> E[Segmentação intrarrenal]
    E --> F[Ecogenicidade / atributos]
    F --> G[Análise exploratória]
```

## Navegação

| Página | Conteúdo |
|---|---|
| [Arquitetura do pipeline](arquitetura.md) | Componentes e fluxo ponta a ponta |
| [Dados e curadoria](dados-e-curadoria.md) | kidneyUS, MONAI, pseudo-máscaras e human-in-the-loop |
| [Modelos e resultados](modelos-e-resultados.md) | Cápsula, intrarrenal, consenso e resultados consolidados |
| [Ecogenicidade](ecogenicidade.md) | Marcadores quantitativos e protocolo córtex–CEC |
| [Execução e reprodutibilidade](execucao-e-reprodutibilidade.md) | Scripts, diretórios e comandos principais |
| [Histórico metodológico](historico-metodologico.md) | Decisões que mudaram ao longo do projeto |
| [Roadmap e limitações](roadmap-e-limitacoes.md) | Pendências científicas e técnicas |

## Componentes principais

| Caminho | Papel |
|---|---|
| `src/segmentation/` | Segmentação renal, avaliação, consenso e experimentos |
| `engenharia_dataset/` | Construção, expansão, conversão e divisão de datasets |
| `curadoria_web/` | Aplicação web de revisão e inferência |
| `models/` | Metadados e checkpoints |
| `docs/` | Relatórios técnicos e decisões metodológicas |
| `artigo/SBBD_2026___Jefferson/` | Versão ativa do artigo |
| `config/` | Catálogo de fontes externas |
| `run_pipeline.py` | Runner simplificado para atributos/classificação renal |

## Estado científico resumido

- Cápsula renal: U-Net com Dice **0,9290** no teste deduplicado de 70 imagens do kidneyUS.
- Intrarrenal: U-Net multiclasse com Dice médio **0,7594** em 50 imagens de teste.
- Base externa MONAI: 4.479 imagens no recorte consolidado do artigo; 3.534 pseudo-máscaras renais candidatas.
- Consenso U-Net × DeepLabV3: usado para **priorizar revisão**, não para validar automaticamente.
- Ecogenicidade: marcador exploratório na imagem original, sem CLAHE, com comparação pareada córtex–CEC.
- Curadoria: revisões e correções por revisor, com histórico auditável.

## Regra metodológica atual

Pseudo-máscaras não devem ser tratadas como `ground truth` apenas por atingirem limiares automáticos. Confiança, morfologia e concordância entre modelos servem para organizar a fila; a incorporação a uma base curada requer revisão humana.

## Documentos técnicos relacionados

- `docs/narrativa_engenharia_dataset.md`
- `docs/organizacao_repositorio.md`
- `docs/organizacao_datasets_curadoria.md`
- `docs/reuniao_curadoria_human_in_the_loop.md`
- `docs/eixo_ecogenicidade_renal.md`
- `docs/geracao_mascaras_modelo_campeao.md`
- `docs/proximos_passos_modelo_intrarrenal.md`
- `docs/resultados_deeplab_dataset_geral_cv.md`
- `docs/resultados_expansao_pseudomascaras_medulla.md`
- `docs/revisao_medico_metodologica_fibrose.md`

A wiki funciona como mapa do projeto; os relatórios técnicos preservam os detalhes dos experimentos.
