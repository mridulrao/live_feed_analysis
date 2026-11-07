#!/usr/bin/env bash
set -euo pipefail
source /workspace/onestreamer-env/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/outputs
output="/workspace/outputs/responses-$(date -u +%Y%m%dT%H%M%S)-$$.jsonl"
python /workspace/OneStreamer/Inference/inference.py \
  --model-path /workspace/models/OneStreamer-4B \
  --output "$output" "$@"
