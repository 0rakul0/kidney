# Ferramentas de segmentacao

Organizacao atual dos scripts operacionais:

| Pasta | Uso | Scripts principais |
| --- | --- | --- |
| `bench/` | Benchmarks e comparacoes globais | `capsule.py` |
| `predict/` | Geracao de pseudo-mascaras | `missing_kidney.py`, `inner_unet.py`, `inner_deeplab.py`, `medulla_roi.py`, `inner_samples.py` |
| `post/` | Limpeza e pos-processamento | `clean_kidney.py`, `constrain_inner.py` |
| `compare/` | Divergencias entre modelos | `inner_models.py` |

Comandos mais usados:

```powershell
.\.venv\Scripts\python.exe src\segmentation\tools\bench\capsule.py
.\.venv\Scripts\python.exe src\segmentation\experiments\train_inner_unet.py
.\.venv\Scripts\python.exe src\segmentation\experiments\train_inner_deeplab.py
.\.venv\Scripts\python.exe src\segmentation\tools\predict\missing_kidney.py --model unet --clahe
.\.venv\Scripts\python.exe src\segmentation\tools\predict\inner_unet.py
.\.venv\Scripts\python.exe src\segmentation\tools\predict\inner_deeplab.py
.\.venv\Scripts\python.exe src\segmentation\tools\compare\inner_models.py
.\.venv\Scripts\python.exe src\segmentation\tools\compare\unet_preprocess.py
.\.venv\Scripts\python.exe src\segmentation\tools\post\clean_kidney.py
.\.venv\Scripts\python.exe src\segmentation\tools\post\constrain_inner.py
```
