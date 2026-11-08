#!/usr/bin/env bash
# Optional setup on the Linux GPU host, after setup-runpod.sh.
set -euo pipefail
command -v nvidia-smi >/dev/null
command -v git >/dev/null
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
APP_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCATE_SOURCE="${LOCATE_ANYTHING_SOURCE:-/workspace/Eagle/Embodied}"
LOCATE_ENV="${LOCATE_ANYTHING_ENV:-/workspace/locate-anything-env}"
LOCATE_MODEL="${LOCATE_ANYTHING_MODEL:-/workspace/models/LocateAnything-3B}"
if [[ ! -f "$LOCATE_SOURCE/locateanything_worker.py" ]]; then
  if [[ "$LOCATE_SOURCE" != /workspace/Eagle/Embodied ]]; then
    echo "LOCATE_ANYTHING_SOURCE must contain NVIDIA's locateanything_worker.py" >&2
    exit 1
  fi
  git clone https://github.com/NVlabs/Eagle.git /workspace/Eagle
fi
if [[ ! -x "$LOCATE_ENV/bin/python" ]]; then
  /workspace/onestreamer-env/bin/python -m venv "$LOCATE_ENV"
fi
"$LOCATE_ENV/bin/python" -m pip install --upgrade pip setuptools wheel packaging ninja
"$LOCATE_ENV/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 "$LOCATE_ENV/bin/python" -m pip install -e "$LOCATE_SOURCE"
"$LOCATE_ENV/bin/python" - "$LOCATE_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download('nvidia/LocateAnything-3B', local_dir=sys.argv[1])
PY
# OpenCV wheels share the cv2 namespace. Remove competing variants before
# reinstalling contrib so an existing ByteTrack installation cannot mask CSRT.
/workspace/onestreamer-env/bin/python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
/workspace/onestreamer-env/bin/python -m pip install --force-reinstall --no-deps opencv-contrib-python-headless==4.11.0.86
/workspace/onestreamer-env/bin/python -m pip install -r "$APP_SOURCE/requirements-worker.txt"
"$LOCATE_ENV/bin/python" -m pip freeze > /workspace/locate-anything-installed.txt
echo 'Setup finished. Add these values to /workspace/.onestreamer-livekit.env:'
echo 'TRACKING_MODELS_ENABLED=true'
echo 'DETECTION_PLUGIN=locate_anything'
echo 'TRACKING_PLUGIN=csrt'
echo 'LOCALIZATION_RETRY_SECONDS=1'
printf 'LOCATE_ANYTHING_PYTHON=%s/bin/python\nLOCATE_ANYTHING_SOURCE=%s\nLOCATE_ANYTHING_MODEL=%s\nLOCATE_ANYTHING_BATCH_RUNTIME=0\n' "$LOCATE_ENV" "$LOCATE_SOURCE" "$LOCATE_MODEL"
echo 'Then restart the GPU worker. Models are loaded before the worker connects.'
