#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
import os
import signal
import time
path = Path('/workspace/live-camera/server.pid')
if path.exists():
    pid = int(path.read_text().strip())
    process = Path(f'/proc/{pid}')
    if process.exists():
        command = (process / 'cmdline').read_bytes()
        cwd = (process / 'cwd').resolve()
        legacy = b'uvicorn' in command and b'server:app' in command
        worker = b'gpu_worker.py' in command
        if not (legacy or worker) or cwd != Path('/workspace/live-camera'):
            raise SystemExit('PID belongs to a different process; refusing to stop it.')
        os.kill(pid, signal.SIGTERM)
        for _ in range(90):
            if not process.exists() or (process / 'stat').read_text().split()[2] == 'Z':
                break
            time.sleep(.5)
        else:
            raise SystemExit('Server is still shutting down. Check its logs before restarting.')
PY
nohup bash /workspace/live-camera/start.sh > /workspace/logs/camera-server.log 2>&1 < /dev/null &
echo "GPU worker starting (PID $!). Check /workspace/logs/camera-server.log for model readiness."
