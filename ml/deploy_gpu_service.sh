#!/bin/bash
# Deploy Aldar Kose GPU inference microservice

set -euo pipefail

echo "🚀 Deploying Aldar Kose GPU Inference Service..."

PORT=${PORT:-8001}
WORKERS=${WORKERS:-1}

echo "📦 Building Docker image..."
docker build -f Dockerfile.service -t aldar-kose-gpu:latest .

echo "🛑 Stopping existing container..."
docker stop aldar-kose-gpu >/dev/null 2>&1 || true
docker rm aldar-kose-gpu >/dev/null 2>&1 || true

echo "▶️  Starting GPU service container..."
docker run -d \
  --name aldar-kose-gpu \
  --gpus all \
  -p "${PORT}:8001" \
  -v "$(pwd)"/outputs:/app/outputs \
  -v "$(pwd)"/configs:/app/configs:ro \
  -e OUTPUT_DIR=/app/outputs \
  -e IMAGE_BASE_URL="${IMAGE_BASE_URL:-http://localhost:8000/images}" \
  -e PORT=8001 \
  -e WORKERS="${WORKERS}" \
  --restart unless-stopped \
  aldar-kose-gpu:latest

echo "✅ GPU service deployed successfully!"
echo "📊 Service running at http://localhost:${PORT}"
echo "🏥 Health check: http://localhost:${PORT}/health"
echo "📋 Logs: docker logs -f aldar-kose-gpu"
