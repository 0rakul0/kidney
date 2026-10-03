# Ecogenicidade renal quantitativa

[← Wiki](README.md)

## Escopo

O módulo é exploratório. Quantifica diferenças de intensidade em ultrassom B-mode; isoladamente, não diagnostica fibrose, DRC ou IFTA.

## Regra central

As medidas de brilho usam a imagem original, sem CLAHE e sem super-resolução.

## Medidas

Se `mu_C`, `mu_M` e `mu_CEC` são as intensidades médias de córtex, medula e CEC:

- razão córtex/CEC: `100 * mu_C / mu_CEC`;
- diferença CEC–córtex: `mu_CEC - mu_C`;
- contraste normalizado: `100 * (mu_CEC - mu_C) / (mu_CEC + mu_C)`;
- razão córtex/medula;
- mediana, IQR, cobertura e saturação.

## Protocolo pareado córtex–CEC

1. erodir máscaras para evitar bordas;
2. dividir a faixa vertical em estratos superior, médio e inferior;
3. buscar até três pares de ROIs circulares;
4. usar mesma área e profundidade aproximada;
5. calcular medidas por par;
6. usar a mediana das razões válidas como estimativa principal;
7. manter o cálculo global como descritor secundário.

## Interpretação

A redução da diferença entre córtex e CEC descreve maior semelhança de brilho. A interpretação clínica depende de validação externa com função renal, laudo ou histopatologia.

## Referências internas

- `docs/eixo_ecogenicidade_renal.md`
- `docs/revisao_medico_metodologica_fibrose.md`
