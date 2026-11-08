#!/usr/bin/env bash
# Resume after dependency installation without rebuilding CUDA/FlashAttention.
set -euo pipefail
source /workspace/onestreamer-env/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /workspace/models /workspace/logs /workspace/outputs
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='MCG-NJU/OneStreamer-4B', local_dir='/workspace/models/OneStreamer-4B')
PY
cp /workspace/live-camera/run-video.sh /workspace/run-onestreamer.sh
cp /workspace/live-camera/README.md /workspace/README-OneStreamer.md
chmod +x /workspace/run-onestreamer.sh /workspace/live-camera/start.sh
python -m pip freeze > /workspace/onestreamer-installed.txt
echo 'Setup complete. Configure /workspace/.onestreamer-livekit.env, then bash /workspace/live-camera/restart.sh'
