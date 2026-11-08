#!/usr/bin/env bash
# Run from any clone location on the GPU instance.
set -euo pipefail
APP_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_REV=338e5fe77383fc8d691bd6226c0358bcb2c83704
command -v git >/dev/null || { echo 'Install Git before running setup.' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'Install Python 3 before running setup.' >&2; exit 1; }
command -v nvcc >/dev/null || { echo 'Use a CUDA development image containing nvcc for FlashAttention.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'This setup needs an NVIDIA GPU runtime.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
export UV_PYTHON_INSTALL_DIR=/workspace/.python
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
# GPU templates can enable an optional downloader absent from our isolated venv.
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p /workspace/logs /workspace/models /workspace/outputs
if ! command -v uv >/dev/null; then
  python3 -m venv /workspace/uv-bootstrap
  /workspace/uv-bootstrap/bin/python -m pip install uv
  export PATH="/workspace/uv-bootstrap/bin:$PATH"
fi
export ONESTREAMER_APP_SOURCE="$APP_SOURCE"
python3 - <<'PY'
import os
from pathlib import Path
import shutil
source = Path(os.environ['ONESTREAMER_APP_SOURCE']).resolve()
destination = Path('/workspace/live-camera')
if source != destination.resolve():
    def ignore_runtime(directory, names):
        ignored = set(shutil.ignore_patterns('__pycache__', '*.pyc', '._*',
                      'server.pid', 'node_modules')(directory, names))
        ignored.update(name for name in names if name.startswith('.env')
                       and name not in ('.env.example', '.env.worker.example'))
        return ignored
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=ignore_runtime)
PY
if [[ ! -d /workspace/OneStreamer/.git ]]; then
  git clone https://github.com/MCG-NJU/OneStreamer.git /workspace/OneStreamer
  git -C /workspace/OneStreamer checkout --detach "$UPSTREAM_REV"
elif [[ "$(git -C /workspace/OneStreamer rev-parse HEAD)" != "$UPSTREAM_REV" ]]; then
  echo "Existing /workspace/OneStreamer is not at $UPSTREAM_REV. Preserve that checkout and move it aside before setup." >&2
  exit 1
fi
uv python install 3.10
if [[ ! -x /workspace/onestreamer-env/bin/python ]]; then
  uv venv --python 3.10 --seed /workspace/onestreamer-env
fi
source /workspace/onestreamer-env/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install packaging==25.0 psutil==7.2.2 ninja setuptools wheel
MAX_JOBS=4 python -m pip install --no-build-isolation -r /workspace/OneStreamer/Eval/requirements.txt
# Worker requirements install the contrib OpenCV wheel used by CSRT.
python -m pip install -r /workspace/live-camera/requirements-worker.txt
bash /workspace/live-camera/finish-setup.sh
