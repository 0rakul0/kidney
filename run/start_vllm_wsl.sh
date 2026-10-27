#!/usr/bin/env bash
set -euo pipefail
# WSL2 Ubuntu. The Windows app connects to this OpenAI-compatible endpoint.
# Override VLLM_MODEL when testing another compatible model.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"

# This key-value syntax is accepted by the pinned vLLM 0.8.x image. Disabling
# video avoids accepting an unused modality for this image-only service.
exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name "$MODEL" \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.82 \
  --limit-mm-per-prompt image=1,video=0 \
  --mm-processor-kwargs '{"max_pixels":262144}'
