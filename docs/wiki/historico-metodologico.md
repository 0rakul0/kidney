# Histórico metodológico

[← Wiki](README.md)

## Segmentação isolada → cascata anatômica

O projeto começou com segmentação renal e comparação de arquiteturas. Depois, a cápsula passou a ser a ROI para estruturas internas e análises quantitativas.

## Expansão de dataset

A base kidneyUS foi complementada com fontes externas, principalmente MONAI, preservando proveniência.

## Pseudo-rótulos → human-in-the-loop

Experimentos antigos usaram filtros automáticos para materializar pseudo-máscaras. A política atual é mais conservadora:

- confiança: prioridade;
- consenso: prioridade;
- morfologia: prioridade;
- revisão humana: decisão final.

## Fibrose → marcador de anomalia/ecogenicidade

A documentação foi refinada para evitar inferência clínica sem referência adequada. Qualquer estudo de fibrose/IFTA exige referência clínica ou histológica.

## Ecogenicidade global → ROIs pareadas

A razão global foi mantida como descritor secundário. O protocolo principal passou a usar ROIs pareadas córtex–CEC.

## Curadoria como componente do sistema

A curadoria ganhou aplicação própria, banco SQLite, edição de máscaras, histórico, inferência e exportação.

## Como tratar resultados antigos

Classifique cada resultado como baseline, experimento histórico, resultado atual, pendente de curadoria ou não comparável por mudança de protocolo.
