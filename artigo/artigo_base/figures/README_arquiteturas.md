# Diagramas das arquiteturas

Quatro imagens autorais foram preparadas para o artigo, com os seguintes nomes esperados na pasta `figures/`:

| Arquivo | Arquitetura | Fonte original citada no artigo |
|---|---|---|
| `arquitetura_unet.png` | U-Net | `Ronneberger2015` |
| `arquitetura_unetplusplus.png` | U-Net++ | `Zhou2018UNetPP` |
| `arquitetura_deeplabv3.png` | DeepLabV3 | `Chen2017DeepLabV3` |
| `arquitetura_segformer_b0.png` | SegFormer-B0 | `Xie2021SegFormer` |

As legendas em `../main.tex` citam os artigos originais e identificam os desenhos como elaboração própria. O texto contém `\\IfFileExists`, portanto figuras ausentes ficam ocultas durante a compilação. É indispensável copiar os quatro PNGs gerados para esta pasta e versioná-los no Git antes da versão final do PDF.

**Nota técnica**: diagramas são representações conceituais das arquiteturas, não visualizações extraídas diretamente dos modelos. A configuração efetiva (por exemplo, taxas de dilatação do ASPP) deve ser conferida nos scripts de treinamento, não deduzida apenas da ilustração.
