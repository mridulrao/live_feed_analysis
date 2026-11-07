#!/usr/bin/env bash
set -euo pipefail
source /workspace/onestreamer-env/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export LIVE_CAMERA_ENV="${LIVE_CAMERA_ENV:-/workspace/.onestreamer-livekit.env}"
cd /workspace/live-camera
echo $$ > /workspace/live-camera/server.pid
exec python gpu_worker.py
