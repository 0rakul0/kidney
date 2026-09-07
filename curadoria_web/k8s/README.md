# Piloto local no Kubernetes do Docker Desktop

Este piloto executa uma unica replica e guarda imagens, mascaras, banco SQLite
e cache em um PVC local. Ele e adequado para validar a aplicacao na propria
maquina; nao deve ser exposto na rede antes da etapa de autenticacao.

## Preparar a imagem e o armazenamento

Na raiz do projeto, crie a imagem:

```powershell
docker build -t curadoria-web:optimized-20260829 .\curadoria_web
```

Crie o namespace e o volume persistente:

```powershell
kubectl apply -f .\curadoria_web\k8s\namespace.yaml
kubectl apply -f .\curadoria_web\k8s\pvc.yaml
kubectl apply -f .\curadoria_web\k8s\data-loader.yaml
kubectl -n curadoria wait --for=condition=Ready pod/data-loader --timeout=120s
```

Copie para o volume o conteudo necessario da raiz do projeto. Isso preserva a
estrutura de caminhos usada pela curadoria:

```powershell
kubectl -n curadoria cp .\dataset_aumentado data-loader:/data/dataset_aumentado
kubectl -n curadoria cp .\results data-loader:/data/results
```

Se forem exibidos modelos ou metadados adicionais, copie tambem as pastas
correspondentes. Em seguida remova o carregador e aplique a aplicacao:

```powershell
kubectl -n curadoria delete pod data-loader
kubectl apply -k .\curadoria_web\k8s
kubectl -n curadoria rollout status deployment/curadoria-web
kubectl -n curadoria port-forward service/curadoria-web 8765:8080
```

Abra `http://127.0.0.1:8765`. O `port-forward` limita o acesso a esta maquina.

## Persistencia e migracao futura

O PVC `curadoria-data` retém os dados enquanto o cluster Docker Desktop existir.
Para migrar, exporte as respostas e faça backup da pasta
`dataset_aumentado/curadoria/respostas/`. A proxima etapa colaborativa troca o
SQLite por PostgreSQL e adiciona login, antes de liberar acesso a outros
especialistas.
