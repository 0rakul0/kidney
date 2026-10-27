# Kidney Assistant - provedores local e nuvem

Modulo independente de pesquisa para padronizacao conservadora de observacoes e chat sobre evidencias de ultrassonografia renal. **Nao diagnostica fibrose, nao valida segmentacoes, nao gera laudos autonomos.**

O projeto oferece dois servidores FastAPI independentes, com os mesmos schemas:

- `src/kidney_assistant/cloud_app.py`: provedor opcional GPT-6 Luna, sempre exigindo consentimento explícito para processamento externo.
- `src/kidney_assistant/local_app.py`: provedor local via API compatível com OpenAI, usando Qwen2.5-VL-3B-Instruct-AWQ no vLLM/WSL2.

## Instalar e iniciar

```bash
pip install -r curadoria_web/requirements.txt
# A configuracao local fica em `.env` (fora do Git):
# OPENAI_API_KEY=sua-chave
# OPENAI_MODEL=gpt-6-luna
# O modulo carrega esse arquivo automaticamente com python-dotenv.
pytest -q tests/kidney_assistant
uvicorn src.kidney_assistant.cloud_app:app --host 127.0.0.1 --port 8010
```

O modelo padrao e `gpt-6-luna`; pode ser alterado com `OPENAI_MODEL`. O servidor nao possui autenticacao: **execute apenas em localhost**, nunca exponha diretamente na internet. Nao inclua dados identificaveis de pacientes. As chamadas externas requerem `consent_external_processing=true` e usam `store=False` (isso nao substitui revisao das politicas de retencao do provedor).

## Endpoints

- `POST /api/notes/normalize`: extracao lexical local, sem custo de API.
- `POST /api/notes/luna-suggest`: proposta de reescrita por LLM, nunca aprovada automaticamente.
- `POST /api/chat/luna`: pergunta e objeto `evidence` com metricas/mascaras descritas textualmente.
- `POST /api/chat/luna-image`: `multipart/form-data` com `question`, `evidence_json`, `consent_external_processing=true`, `image` PNG/JPEG/WebP de ate 5 MB. Envia imagem a API externa **somente quando explicitamente solicitado**.

Exemplo de proposta de normalizacao:

```bash
curl -X POST http://127.0.0.1:8010/api/notes/luna-suggest -H "Content-Type: application/json" -d '{"image_id":"US_001","observation":"Rim direito com ecogenicidade cortical aumentada.","consent_external_processing":true}'
```

## Integração automática com a Curadoria Web

A Curadoria Web agora recebe somente `image_id`, `reviewer`, pergunta e modo visual. O servidor recupera os artefatos da própria inferência ou do `CurationStore`, incluindo imagem, máscaras efetivas, `capsule_bbox`, medidor de brilho e métricas. O cliente não envia caminhos locais nem pode substituir uma máscara por outra imagem.

Endpoints locais, disponíveis apenas no servidor da Curadoria:

- `POST /api/assistant/analyze`: Qwen local ou GPT-6 Luna, conforme `provider`.
- `POST /api/assistant/compare`: executa os dois modelos com os mesmos artefatos e retorna uma lista adequada para tabela.
- `GET /api/assistant/runs`: recupera as execuções persistidas por imagem e revisor.

O campo `visual_mode` aceita `capsule` e `full`. Ambos recebem uma sobreposição visual da imagem com as máscaras de rim, córtex, medula e CEC. No modo `capsule`, a sobreposição é recortada pelo `capsule_bbox`; no modo `full`, a imagem completa preserva referências anatômicas externas. O modo externo exige `consent_external_processing=true` e o revisor é obrigatório.

Cada execução é registrada em `assistant_runs`. A resposta só entra na observação curatorial quando o especialista a copia para o formulário e salva a revisão; o `assistant_run_id` fica associado ao registro aprovado. A sugestão continua sendo uma evidência de pesquisa, nunca aprovação ou diagnóstico automático.

## Contratos Pydantic v2

`src/kidney_assistant/schemas.py` centraliza `NoteIn`, `LunaNoteIn`, `ChatIn`, `ClinicalFinding`,
`LLMNoteProposal`, `LunaNoteProposal`, `NormalizedNoteResponse` e `ChatResponse`.
Os quatro endpoints declaram `response_model`, expondo os contratos em `/docs`.
A proposta textual do LLM passa por `LLMNoteProposal.model_validate` e rejeita
campos extras; os resultados seguem pendentes de aprovacao humana. O fluxo
continua usando JSON textual na API Responses, nao o recurso nativo de Structured Outputs.

## Servidor local vLLM (Windows + WSL2, RTX 3050 8 GB)

A versao local usa `src/kidney_assistant/local_app.py` e `src/kidney_assistant/local_vllm.py` com os mesmos contratos Pydantic de `src/kidney_assistant/schemas.py`. Ela **nao chama a API externa**. O servidor vLLM oficial requer Linux; use WSL2 Ubuntu. Modelo sugerido: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` (4 bits). A capacidade real depende de versao, drivers, memoria livre e configuracao.

1. PowerShell (administrador): `wsl --install -d Ubuntu` e reinicie quando solicitado.
2. No Ubuntu WSL, confirme a GPU com `nvidia-smi`. Crie um ambiente Python compatível e instale vLLM segundo a [documentacao oficial](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).
3. Na raiz do projeto, rode `bash run/start_vllm_wsl.sh`. O script inicia exatamente o perfil recomendado: contexto 2.048, uma sequência, 82% da VRAM e uma imagem por requisição.
4. No Windows, na raiz do projeto, rode `pip install -r curadoria_web/requirements.txt` e `uvicorn src.kidney_assistant.local_app:app --host 127.0.0.1 --port 8010`. Abra `http://127.0.0.1:8010/docs`.
5. `VLLM_BASE_URL` usa `http://127.0.0.1:8000/v1` por padrão e `VLLM_MODEL` usa `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`. Se o localhost do WSL não for encaminhado, ajuste `VLLM_BASE_URL` para o endereço do servidor vLLM.

O parâmetro multimodal do launcher usa a sintaxe compatível com a imagem vLLM
0.8.x configurada no Docker:

```bash
--limit-mm-per-prompt image=1,video=0
--mm-processor-kwargs '{"max_pixels":262144}'
```

O limite de pixels reduz a memória da torre visual para permitir que o modelo
caiba na RTX 3050 de 8 GB. Se ainda houver CUDA OOM, reduza `--max-model-len`
para `1024`.

Se houver CUDA OOM, reduza `--max-model-len` para `1024`, feche outros processos da GPU e diminua `--gpu-memory-utilization`. A quantização AWQ pode caber em uma RTX 3050 de 8 GB, mas o carregamento real depende da versão do vLLM, do driver e da memória livre.

Endpoints locais: `/health`, `/api/notes/normalize`, `/api/notes/local-suggest`, `/api/chat/local`, `/api/chat/local-image`. `local-suggest` exige schema JSON gerado a partir de `LLMNoteProposal`; o resultado e revalidado por Pydantic e permanece `approved=false`. O chat recebe imagem como `image_url` base64 no protocolo OpenAI Chat Completions. Os uploads da inferência passam por validação de contêiner, formato, dimensões, pixels descomprimidos e carregamento integral antes do PyTorch. A API deve permanecer restrita a localhost e usar somente dados autorizados/anonimizados.

Exemplo de evidência para acionar o recorte da cápsula:

```json
{
  "capsule_bbox": {"x1": 120, "y1": 80, "x2": 860, "y2": 650},
  "capsule_padding_ratio": 0.12
}
```

Teste sem GPU: `pytest -q` (24 testes, incluindo validação de imagem, lateralidade local por achado e negação). O teste real da configuração atual foi executado na RTX 3050 de 8 GB: Qwen2.5-VL-3B-Instruct-AWQ no Docker, `cuda`, imagem 1024 × 768, quatro camadas e medidor de brilho disponível. O limite de contexto permanece 2048 tokens; o adaptador local usa no máximo 512 tokens de saída para reservar espaço às evidências.
