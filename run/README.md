# Scripts PLAY

Use estes arquivos `.bat` para rodar sem digitar argumentos.

| Arquivo | O que faz |
| --- | --- |
| `PLAY_01_benchmark_capsula.bat` | Recalcula o benchmark da capsula/rim. |
| `PLAY_02_treinar_unet_interno.bat` | Treina a U-Net para cortex, medulla e CEC. |
| `PLAY_03_treinar_deeplab_interno.bat` | Treina o DeepLab para cortex, medulla e CEC. |
| `PLAY_04_gerar_rim_unet.bat` | Gera mascaras renais com U-Net, Lanczos, CLAHE e limiar 0.90. |
| `PLAY_05_gerar_interno_unet.bat` | Gera mascaras internas com U-Net. |
| `PLAY_06_gerar_interno_deeplab.bat` | Gera mascaras internas com DeepLab. |
| `PLAY_07_comparar_unet_deeplab.bat` | Compara divergencias entre U-Net e DeepLab. |
| `PLAY_08_limpar_e_atualizar_interface.bat` | Limpa mascaras, recorta regioes internas, refaz miniaturas e abre a interface. |
| `PLAY_09_abrir_interface.bat` | Apenas reinicia e abre a interface de curadoria. |
| `PLAY_pipeline_curadoria.bat` | Roda a sequencia principal completa para curadoria. |
