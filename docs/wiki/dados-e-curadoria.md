# Dados e curadoria

[← Wiki](README.md)

## kidneyUS

É a fonte supervisionada principal do projeto.

| Tarefa | Imagens |
|---|---:|
| Cápsula — registros | 486 |
| Cápsula — imagens únicas | 468 |
| Split cápsula | 328 treino / 70 validação / 70 teste |
| Intrarrenal | 335 |
| Split intrarrenal | 235 treino / 50 validação / 50 teste |

Os splits devem ser feitos por exame/paciente quando possível, reduzindo vazamento.

## MONAI / NVIDIA Clinical Ultrasound Repository

A engenharia de dataset selecionou estudos renais/retroperitoneais, converteu DICOMs úteis em PNG e descartou material inadequado. A fonte externa é usada para robustez, pseudo-rotulagem e curadoria, não como substituto automático da referência manual.

## Estados de máscara

Uma máscara deve ser interpretada pelo status e proveniência:

- manual/original;
- gerada automaticamente;
- pendente de revisão;
- aceita por revisor;
- corrigida;
- rejeitada;
- indisponível.

## Mudança metodológica importante

Versões antigas do fluxo usaram limiares automáticos em experimentos de pseudo-rotulagem. A política mais recente é:

> confiança, morfologia e consenso priorizam a revisão; não substituem a validação humana.

## Human-in-the-loop

O protocolo separa duas decisões: validar a máscara renal e, depois, validar estruturas internas dentro de uma ROI renal confiável.

## Auditoria

A Curadoria Web mantém resposta por revisor e imagem, máscaras corrigidas, histórico imutável, datas, seleção de ROIs e exportação CSV/JSON.

## Diretórios relevantes

```text
dataset_aumentado/
├── fontes/
├── dataset_geral_v2/
├── dataset_geral_cv/
├── dataset_intrarrenal/
└── curadoria/
    └── respostas/
```

Datasets derivados devem ser reproduzíveis a partir de fontes, scripts e parâmetros.
