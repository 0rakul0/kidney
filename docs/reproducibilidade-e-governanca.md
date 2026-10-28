# Reprodutibilidade e governança

## Checkpoints versionados por Git LFS

Os pesos de segmentação são armazenados por Git LFS. Um clone comum pode
conter apenas arquivos-pontes pequenos, que não são checkpoints executáveis.
Após instalar o Git LFS, prepare o conteúdo efetivo com:

```powershell
git lfs install
git lfs pull
python scripts/check_lfs_assets.py --strict
```

O script de verificação não baixa modelos. Ele apenas diferencia conteúdo
presente, ponteiro LFS e arquivo ausente. A imagem Docker da Curadoria Web não
substitui essa etapa quando a inferência local precisa carregar os checkpoints.

## Testes automatizados

O GitHub Actions executa os testes Python, verifica o ponto de entrada do
pipeline e audita o SQLite versionado. O job é deliberadamente independente da
GPU e não baixa os checkpoints durante os testes unitários. A execução de uma
inferência real continua exigindo os pesos baixados por Git LFS e o ambiente
CUDA correspondente.

## Textos acadêmicos

O documento canônico do ciclo atual é:

```text
artigo/artigo_base/main.tex
```

O diretório `artigo/SBBD_2026___Jefferson/` preserva uma versão anterior do
texto SBBD. Os arquivos em `out/pdf_text/` são extrações derivadas e não devem
ser editados como fonte. Alterações novas devem ser feitas no documento
canônico, recompiladas e comparadas com a versão anterior quando a finalidade
for publicação ou extensão do trabalho.

## SQLite de curadoria

`dataset_aumentado/curadoria/respostas/curadoria.sqlite3` é um artefato
versionado de operação e auditoria. Antes de adicionar novos registros ao
repositório público, execute:

```powershell
python scripts/audit_curadoria_sqlite.py
```

O auditor verifica padrões de e-mail, chaves OpenAI, credenciais e possíveis
identificadores clínicos sem imprimir os valores encontrados. A ausência de
um padrão não substitui a revisão humana de autorização, anonimização e
licença dos dados. Registros de pacientes, credenciais ou respostas externas
não devem ser adicionados ao repositório público.
