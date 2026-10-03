# Dados e curadoria

[← Wiki](README.md)

## Objetivo desta página

Esta página explica **de onde vêm os dados, como são organizados e como uma predição automática pode evoluir até uma anotação revisada**.

O ponto central é preservar proveniência. Uma máscara manual, uma pseudo-máscara e uma máscara corrigida por revisor não devem ser tratadas como equivalentes.

## Fonte supervisionada principal — kidneyUS

O kidneyUS é a base de referência do projeto para treinamento e avaliação com anotações manuais.

### Cápsula renal

No recorte consolidado:

| Item | Quantidade |
|---|---:|
| Registros | 486 |
| Imagens únicas | 468 |
| Treino | 328 |
| Validação | 70 |
| Teste | 70 |

A deduplicação e o agrupamento por exame reduzem o risco de imagens muito semelhantes aparecerem simultaneamente em treino e teste.

### Estruturas intrarrenais

| Item | Quantidade |
|---|---:|
| Imagens | 335 |
| Treino | 235 |
| Validação | 50 |
| Teste | 50 |

Essas imagens possuem referência para estruturas internas utilizadas na tarefa multiclasse.

## Fonte externa — MONAI/NVIDIA Clinical Ultrasound Repository

A fonte MONAI foi incorporada para aumentar diversidade.

O processo de engenharia de dados incluiu:

1. obtenção dos metadados globais;
2. busca por estudos renais/retroperitoneais;
3. download incremental;
4. leitura de DICOM;
5. seleção de frames úteis;
6. conversão para PNG;
7. descarte de imagens inadequadas;
8. registro em manifesto.

O documento detalhado está em:

```text
docs/narrativa_engenharia_dataset.md
```

## Por que dados externos não entram diretamente no treino

As imagens externas podem não possuir máscara manual.

Por isso, o fluxo correto é:

```text
imagem externa
    ↓
modelo gera máscara candidata
    ↓
confiança / consenso / filtros
    ↓
fila de revisão
    ↓
humano aceita, corrige ou rejeita
    ↓
base curada
```

## Tipos de máscara

### Manual/original

Produzida por anotador humano na base supervisionada original.

### Pseudo-máscara

Gerada por modelo.

Ela pode ser útil para:

- ampliar cobertura;
- criar fila de revisão;
- experimentar pseudo-labeling;
- estudar redução de custo de anotação.

Ela não é automaticamente `ground truth`.

### Máscara revisada

Foi explicitamente avaliada por revisor.

### Máscara corrigida

A proposta automática foi modificada manualmente. A correção deve permanecer separada da versão original.

## Estados usados na curadoria

O fluxo conceitual inclui:

- `pendente`;
- `aceita`;
- `corrigir`;
- `rejeitada`;
- `indisponivel`.

Esses estados podem existir individualmente para rim, córtex, medula e CEC.

## Human-in-the-loop

O protocolo divide a validação em níveis.

### Nível 1 — rim

Pergunta principal:

> A imagem contém um rim avaliável e a máscara externa delimita corretamente o órgão?

### Nível 2 — estruturas internas

Somente após uma ROI renal confiável deve-se avaliar:

- córtex;
- medula;
- CEC.

Isso reduz o risco de validar uma estrutura interna em uma ROI externa incorreta.

## Curadoria Web

A aplicação fica em:

```text
curadoria_web/
```

Ela permite:

- selecionar revisor;
- visualizar imagem e sobreposições;
- trocar o modelo intrarrenal exibido;
- editar máscaras por polígono;
- aceitar ou rejeitar cada estrutura;
- gravar observações;
- exportar respostas;
- registrar histórico das correções.

## Persistência

No modo local, as respostas são armazenadas em:

```text
dataset_aumentado/curadoria/respostas/curadoria.sqlite3
```

Máscaras corrigidas ficam separadas em:

```text
dataset_aumentado/curadoria/respostas/mascaras_corrigidas/
```

Cada salvamento manual também mantém um histórico imutável.

## Evolução da política de pseudo-rótulos

Em etapas anteriores, alguns experimentos usaram filtros automáticos para selecionar máscaras de maior qualidade.

Essa abordagem foi importante para explorar pseudo-labeling, mas a política metodológica atual é mais conservadora.

### Antes

```text
passou limiar → entra no conjunto experimental
```

### Agora

```text
passou limiar → ganha prioridade de revisão
```

A diferença é importante para interpretar documentos históricos.

## Estrutura dos datasets derivados

```text
dataset_aumentado/
├── fontes/
├── dataset_geral_v2/
├── dataset_geral_cv/
├── dataset_intrarrenal/
│   ├── intermediario/
│   ├── supervisionado/
│   └── pseudo_expandido/
└── curadoria/
    └── respostas/
```

## O que deve acompanhar cada imagem

Idealmente, o manifesto deve permitir recuperar:

- origem;
- identificador;
- caminho da imagem;
- tipo de anotação;
- modelo que gerou a pseudo-máscara;
- checkpoint;
- confiança;
- consenso;
- status de revisão;
- revisor;
- data;
- versão corrigida.

## Princípio metodológico

> O projeto pode usar automação para reduzir a carga de anotação, mas a automação não deve apagar a distinção entre rótulo previsto e rótulo revisado.

## Próxima leitura

A página [Modelos e resultados](modelos-e-resultados.md) mostra quais modelos produzem essas máscaras e como foram avaliados.
