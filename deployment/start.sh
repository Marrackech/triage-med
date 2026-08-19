#!/bin/bash
set -e

echo "🚀 Démarrage du serveur vLLM..."
python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --dtype float16 \
    --max-model-len 2048 \
    --port ${VLLM_PORT} \
    --gpu-memory-utilization 0.85 &

echo "⏳ Attente du démarrage de vLLM..."
sleep 60

echo "🚀 Démarrage du serveur FastAPI..."
uvicorn fastapi_app:app --host 0.0.0.0 --port ${API_PORT}